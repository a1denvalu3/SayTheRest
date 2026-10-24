use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Body,
    extract::{Path as RoutePath, State},
    http::{Request, StatusCode, header},
    middleware::{self, Next},
    response::Response,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
};
use sayit_core::{
    BenchmarkRunner, EngineConfig, InferenceRuntime, ManagedEngine, SynthesisRequest,
    TextCleaningOptions, TtsEngine, clean_text, wav_duration_seconds,
};
use sayit_protocol::{
    Health, HistoryItem, HuggingFaceModelImportRequest, InferenceDeviceStatus, JobState,
    LocalModelImportRequest, ModelBenchmark, ModelDescriptor, PROTOCOL_VERSION, PinRequest,
    PlaybackSnapshot, PlaybackTextChunk, QueuePolicy, ServiceEvent, ServiceSettings,
    ServiceSettingsUpdate, ServiceSnapshot, SpeechJob, SpeechSource, SpeechSubmission,
    VoiceCloneRequest, VoiceProfile, VoiceQuality, VoiceRenameRequest, VoiceSynthesisMode,
    VoiceTranscriptRequest, VoiceTuningRequest,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    sync::{Notify, RwLock, broadcast},
};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use uuid::Uuid;

mod model_manager;
mod phonon_voices;
mod player;
pub mod qwen_runtime;
mod speech_blocks;
mod stretch;
mod voice_discovery;
mod voice_modes;
use model_manager::ModelManager;
use player::{PlaybackController, PlayerStatus};

#[derive(Clone)]
pub struct ServiceState {
    inner: Arc<RwLock<InnerState>>,
    notify: Arc<Notify>,
    store_path: PathBuf,
    audio_dir: PathBuf,
    voices_dir: PathBuf,
    engine_config_path: Arc<RwLock<Option<PathBuf>>>,
    models: ModelManager,
    player: PlaybackController,
    api_token: Arc<String>,
    api_read_token: Arc<String>,
    rate_limits: Arc<Mutex<HashMap<&'static str, VecDeque<Instant>>>>,
    events: broadcast::Sender<ServiceEvent>,
    event_sequence: Arc<AtomicU64>,
    resident_engine: Arc<Mutex<Option<ResidentEngineSlot>>>,
    inference_runtime: InferenceRuntime,
    last_inference_device: Arc<Mutex<Option<InferenceDeviceStatus>>>,
    synthesis_cancelled: Arc<AtomicBool>,
    voice_studio: Arc<Mutex<Option<voice_discovery::Studio>>>,
    voice_studio_save: Arc<tokio::sync::Mutex<()>>,
}

struct ResidentEngineSlot {
    config_path: PathBuf,
    config_bytes: Vec<u8>,
    engine: ManagedEngine,
    last_used: Instant,
}

fn inference_device(engine: &ManagedEngine) -> InferenceDeviceStatus {
    let device = engine.device();
    InferenceDeviceStatus {
        provider: device.provider,
        device: device.device,
        dtype: device.dtype,
        fallback_reason: device.fallback_reason,
        backend: device.backend,
        quantization: device.quantization,
    }
}

#[derive(Default, Deserialize, Serialize)]
struct InnerState {
    jobs: VecDeque<SpeechJob>,
    history: Vec<HistoryItem>,
    settings: ServiceSettings,
    #[serde(default)]
    voices: Vec<VoiceProfile>,
    #[serde(default)]
    benchmarks: HashMap<String, ModelBenchmark>,
    #[serde(default)]
    recovery_attempts: HashMap<Uuid, u8>,
}

impl Default for ServiceState {
    fn default() -> Self {
        let root = std::env::temp_dir().join(format!("sayit-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("temporary test directory");
        let model = root.join("test-model.onnx");
        let tokens = root.join("test-tokens.txt");
        fs::write(&model, b"test").expect("temporary test model");
        fs::write(&tokens, b"test").expect("temporary test tokens");
        let config_path = root.join("test-config.json");
        let config = EngineConfig::SherpaOnnxVits(sayit_core::VitsConfig {
            executable: "test-tts".into(),
            model,
            tokens,
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 1,
            speaker_id: 0,
        });
        fs::write(&config_path, serde_json::to_vec(&config).unwrap())
            .expect("temporary test config");
        Self::open(root, Some(config_path)).expect("temporary service state")
    }
}

impl ServiceState {
    pub fn open(data_dir: PathBuf, mut engine_config_path: Option<PathBuf>) -> Result<Self> {
        fs::create_dir_all(&data_dir)
            .with_context(|| format!("failed to create {}", data_dir.display()))?;
        let audio_dir = data_dir.join("history");
        fs::create_dir_all(&audio_dir)?;
        let voices_dir = data_dir.join("voices");
        fs::create_dir_all(&voices_dir)?;
        let store_path = data_dir.join("service-state.json");
        let api_token = load_or_create_api_token(&data_dir)?;
        let api_read_token = load_or_create_named_token(&data_dir, "api-read-token")?;
        let (mut inner, recovered_from_backup) = load_persisted_state(&store_path)?;
        let mut state_changed = recover_interrupted_jobs(&mut inner, &audio_dir);
        let migration_model = inner
            .settings
            .active_model_id
            .clone()
            .unwrap_or_else(|| "pocket-tts-int8".into());
        for voice in &mut inner.voices {
            if voice.model_id.is_empty() {
                voice.model_id = migration_model.clone();
                state_changed = true;
            }
            let reference = PathBuf::from(&voice.reference_audio_path);
            if !reference.is_file()
                && let Some(name) = reference.file_name()
            {
                let migrated = voices_dir.join(name);
                if migrated.is_file() {
                    voice.reference_audio_path = migrated.display().to_string();
                    state_changed = true;
                }
            }
        }
        if inner.settings.voice_profile_by_model.is_empty() {
            inner.settings.voice_profile_by_model.insert(
                migration_model.clone(),
                inner.settings.active_voice_profile_id,
            );
        }
        let models = ModelManager::new(data_dir.join("models"), engine_config_path.as_deref())?;
        let configured_model_id = engine_config_path
            .as_deref()
            .and_then(|path| EngineConfig::from_path(path).ok())
            .map(|config| match config {
                EngineConfig::SherpaOnnxVits(_) => "piper-en-us-lessac-medium",
                EngineConfig::SherpaOnnxKokoro(_) => "kokoro-int8-multi-lang-v1-1",
                EngineConfig::SherpaOnnxKitten(config) => {
                    if config
                        .model
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.contains("int8"))
                    {
                        "kitten-nano-en-v0-8-int8"
                    } else {
                        "kitten-mini-en-v0-8"
                    }
                }
                EngineConfig::SherpaOnnxPocket(_) => "pocket-tts-int8",
                EngineConfig::Qwen3Tts(_) => "qwen3-tts",
                EngineConfig::Qwen3TtsOnnx(_) => "qwen3-tts-06b-base",
                EngineConfig::Phonon(_) => "phonon-31m-en",
            });
        if let (Some(id), Some(source)) = (configured_model_id, engine_config_path.as_deref()) {
            let managed = models.config_path(id);
            if !managed.is_file() {
                if let Some(parent) = managed.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(source, &managed)?;
            }
        }
        if let Some(selected) = inner.settings.active_model_id.as_deref() {
            let selected_config = models.config_path(selected);
            if selected_config.is_file() {
                engine_config_path = Some(selected_config);
            } else if let Some(id) = configured_model_id {
                inner.settings.active_model_id = Some(id.into());
                engine_config_path = Some(models.config_path(id));
            }
        } else if let Some(id) = configured_model_id {
            inner.settings.active_model_id = Some(id.into());
            engine_config_path = Some(models.config_path(id));
        }
        if let Some(model_id) = inner.settings.active_model_id.as_deref() {
            let selected = inner
                .settings
                .voice_profile_by_model
                .get(model_id)
                .copied()
                .flatten()
                .filter(|id| {
                    inner
                        .voices
                        .iter()
                        .any(|voice| voice.id == *id && voice.model_id == model_id)
                });
            inner.settings.active_voice_profile_id = selected;
        }
        state_changed |= enforce_history_policy_inner(&mut inner, &audio_dir, chrono::Utc::now());
        if state_changed || recovered_from_backup {
            write_persisted_state(&store_path, &inner, !recovered_from_backup)?;
        }
        let (events, _) = broadcast::channel(256);
        Ok(Self {
            inference_runtime: InferenceRuntime::new(&data_dir),
            last_inference_device: Arc::new(Mutex::new(None)),
            synthesis_cancelled: Arc::new(AtomicBool::new(false)),
            voice_studio: Arc::new(Mutex::new(None)),
            voice_studio_save: Arc::new(tokio::sync::Mutex::new(())),
            inner: Arc::new(RwLock::new(inner)),
            notify: Arc::new(Notify::new()),
            store_path,
            audio_dir,
            voices_dir,
            engine_config_path: Arc::new(RwLock::new(engine_config_path)),
            models,
            player: PlaybackController::new(),
            api_token: Arc::new(api_token),
            api_read_token: Arc::new(api_read_token),
            rate_limits: Arc::new(Mutex::new(HashMap::new())),
            events,
            event_sequence: Arc::new(AtomicU64::new(0)),
            resident_engine: Arc::new(Mutex::new(None)),
        })
    }

    pub fn start_worker(&self) {
        let state = self.clone();
        tokio::spawn(async move { state.worker_loop().await });
        let state = self.clone();
        tokio::spawn(async move { state.model_unload_loop().await });
        let state = self.clone();
        tokio::spawn(async move {
            // Two snapshots/second leave room for tray polling and management
            // requests under the shared read-token limit of 240/minute. The UI
            // interpolates position between these authoritative updates.
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            let mut was_active = false;
            loop {
                interval.tick().await;
                let playback = state.player.snapshot();
                let active = playback.job_id.is_some()
                    && (!playback.generation_complete
                        || matches!(
                            playback.status,
                            PlayerStatus::Playing | PlayerStatus::Buffering
                        ));
                if active || was_active {
                    state.emit("playback");
                }
                was_active = active;
            }
        });
        self.notify.notify_one();
    }

    async fn model_unload_loop(self) {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            let timeout =
                Duration::from_secs(self.inner.read().await.settings.model_unload_delay_seconds);
            let Ok(mut resident) = self.resident_engine.try_lock() else {
                continue;
            };
            if resident
                .as_ref()
                .is_some_and(|slot| slot.last_used.elapsed() >= timeout)
            {
                *resident = None;
                drop(resident);
                self.emit("models");
            }
        }
    }

    async fn resolve_engine_config(&self) -> Option<PathBuf> {
        if let Some(path) = self.engine_config_path.read().await.clone()
            && EngineConfig::from_path(&path).is_ok()
        {
            return Some(path);
        }
        let active_model = self.inner.read().await.settings.active_model_id.clone()?;
        let path = self.models.config_path(&active_model);
        if EngineConfig::from_path(&path).is_err() {
            return None;
        }
        *self.engine_config_path.write().await = Some(path.clone());
        Some(path)
    }

    async fn worker_loop(self) {
        loop {
            while let Some(job) = self.take_next_job().await {
                self.process_job(job).await;
            }
            self.notify.notified().await;
        }
    }

    async fn take_next_job(&self) -> Option<SpeechJob> {
        let mut inner = self.inner.write().await;
        let job = inner
            .jobs
            .iter_mut()
            .find(|job| job.state == JobState::Queued)?;
        job.state = JobState::Synthesizing;
        self.synthesis_cancelled.store(false, Ordering::Relaxed);
        let result = job.clone();
        drop(inner);
        let _ = self.persist().await;
        Some(result)
    }

    async fn process_job(&self, job: SpeechJob) {
        let config_path = match job.model_id.as_deref() {
            Some(id) => Some(self.models.config_path(id)),
            None => self.resolve_engine_config().await,
        };
        let Some(config_path) = config_path else {
            self.finish_job(
                job.id,
                Err(anyhow::anyhow!("no inference configuration installed")),
            )
            .await;
            return;
        };
        let output = self.audio_dir.join(format!("{}.wav", job.id));
        let (
            reference_audio,
            reference_text,
            speaking_pace,
            refinement_steps,
            generation,
            prefer_gpu,
            rate,
            volume,
        ) = {
            let inner = self.inner.read().await;
            let voice = job
                .voice_profile_id
                .and_then(|id| inner.voices.iter().find(|voice| voice.id == id));
            if job.voice_profile_id.is_some() && voice.is_none() {
                drop(inner);
                self.finish_job(
                    job.id,
                    Err(anyhow::anyhow!(
                        "the voice profile for this job was deleted"
                    )),
                )
                .await;
                return;
            }
            (
                voice.map(|voice| PathBuf::from(&voice.reference_audio_path)),
                voice.and_then(|voice| voice.reference_text.clone()),
                job.speaking_pace.unwrap_or(inner.settings.speaking_pace),
                voice.map(|voice| voice.refinement_steps),
                voice
                    .and_then(|voice| voice.generation_tuning)
                    .map(|tuning| sayit_core::QwenGenerationOptions {
                        seed: None,
                        max_new_tokens: None,
                        sampling: voice_discovery::sampling(tuning),
                    }),
                inner.settings.compute_device != "cpu",
                inner.settings.playback_rate,
                inner.settings.volume,
            )
        };
        let generation = generation.or_else(|| voice_modes::generation_defaults(job.voice_mode));
        let text = job.text.clone();
        let resident_engine = self.resident_engine.clone();
        let runtime = self.inference_runtime.clone();
        let last_device = self.last_inference_device.clone();
        let cancelled = self.synthesis_cancelled.clone();
        if cancelled.load(Ordering::Relaxed) {
            return;
        }
        if let Err(error) = self.player.begin_stream(job.id, rate, volume) {
            self.finish_job(job.id, Err(error)).await;
            return;
        }
        let player = self.player.clone();
        let synthesis = tokio::task::spawn_blocking(move || -> Result<(PathBuf, f64)> {
            let mut config = EngineConfig::from_path(&config_path)?;
            if let Some(language) = &job.language {
                config.set_language(language)?;
            }
            let mut reading_voice = voice_modes::ReadingVoice::new(job.voice_mode, &config)?;
            if let (EngineConfig::SherpaOnnxPocket(pocket), Some(steps)) =
                (&mut config, refinement_steps)
            {
                pocket.num_steps = steps as usize;
            }
            let mut config_bytes = serde_json::to_vec(&config)?;
            // Model/voice-pack revisions can change while the engine settings
            // retain the same paths. Reload on the next request after an update.
            if let Some(directory) = config_path.parent()
                && let Ok(manifest) = fs::read(directory.join("catalog-install.json"))
            {
                config_bytes.extend_from_slice(b"\0catalog\0");
                config_bytes.extend_from_slice(&manifest);
            }
            config_bytes.extend_from_slice(&refinement_steps.unwrap_or_default().to_le_bytes());
            config_bytes.push(u8::from(prefer_gpu));
            let mut resident = resident_engine.lock().unwrap();
            let reload = resident.as_ref().is_none_or(|slot| {
                slot.config_path != config_path || slot.config_bytes != config_bytes
            });
            if reload {
                // Release the previous model before the replacement allocates
                // weights; assignment alone keeps both alive during loading.
                *resident = None;
                *resident = Some(ResidentEngineSlot {
                    config_path: config_path.clone(),
                    config_bytes,
                    engine: ManagedEngine::load(&config, &runtime, prefer_gpu)?,
                    last_used: Instant::now(),
                });
            }
            let slot = resident
                .as_mut()
                .context("resident engine was not loaded")?;
            let result = (|| -> Result<()> {
                let pending =
                    tempfile::NamedTempFile::new_in(output.parent().unwrap())?.into_temp_path();
                let mut archive = None;
                let mut archive_rate = None;
                let blocks = if job.voice_mode == VoiceSynthesisMode::RandomPerParagraph {
                    speech_blocks::paragraph_blocks(&text, 160)
                } else {
                    speech_blocks::speech_blocks(&text, 160)
                };
                for block in blocks {
                    anyhow::ensure!(
                        !cancelled.load(Ordering::Relaxed),
                        "speech generation cancelled"
                    );
                    reading_voice.prepare(&slot.engine, block.paragraph, &cancelled)?;
                    let generated_reference = reading_voice.reference();
                    player.text_block(job.id, block.start, block.end, false)?;
                    let part =
                        tempfile::NamedTempFile::new_in(output.parent().unwrap())?.into_temp_path();
                    slot.engine.synthesize_streaming(
                        &SynthesisRequest {
                            generation: generation.as_ref(),
                            text: block.text,
                            output: part.as_ref(),
                            reference_audio: generated_reference
                                .map(|(audio, _)| audio)
                                .or(reference_audio.as_deref()),
                            reference_text: generated_reference
                                .map(|(_, text)| text)
                                .or(reference_text.as_deref()),
                            speaking_pace,
                        },
                        Some(&cancelled),
                        &mut |samples, sample_rate| {
                            anyhow::ensure!(
                                !cancelled.load(Ordering::Relaxed),
                                "speech generation cancelled"
                            );
                            anyhow::ensure!(
                                archive_rate.is_none_or(|rate| rate == sample_rate),
                                "sample rate changed between speech blocks"
                            );
                            if archive.is_none() {
                                archive = Some(hound::WavWriter::create(
                                    &pending,
                                    hound::WavSpec {
                                        channels: 1,
                                        sample_rate,
                                        bits_per_sample: 16,
                                        sample_format: hound::SampleFormat::Int,
                                    },
                                )?);
                                archive_rate = Some(sample_rate);
                            }
                            for sample in samples {
                                archive.as_mut().unwrap().write_sample(
                                    (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16,
                                )?;
                            }
                            player.append(job.id, sample_rate, samples)
                        },
                    )?;
                    player.text_block(job.id, block.start, block.end, true)?;
                }
                archive
                    .context("inference generated empty audio")?
                    .finalize()?;
                anyhow::ensure!(
                    !cancelled.load(Ordering::Relaxed),
                    "speech generation cancelled"
                );
                pending.persist(&output).map_err(|error| error.error)?;
                Ok(())
            })();
            *last_device.lock().unwrap() = Some(inference_device(&slot.engine));
            if cancelled.load(Ordering::Relaxed) {
                *resident = None;
                anyhow::bail!("speech generation cancelled");
            }
            result?;
            slot.last_used = Instant::now();
            let duration = wav_duration_seconds(&output)?;
            Ok((output, duration))
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("synthesis worker crashed: {error}")));
        match synthesis {
            Ok((path, duration)) => {
                if let Err(error) = self.player.finish_stream(job.id) {
                    let _ = self.player.abort_stream(job.id);
                    let _ = fs::remove_file(path);
                    self.finish_job(job.id, Err(error)).await;
                    return;
                }
                {
                    let mut inner = self.inner.write().await;
                    if let Some(active) = inner.jobs.iter_mut().find(|entry| entry.id == job.id)
                        && active.state == JobState::Synthesizing
                    {
                        active.state = JobState::Playing;
                    }
                }
                self.emit("jobs");
                self.emit("playback");
                self.wait_for_playback(job.id, path, duration).await;
            }
            Err(error) => {
                let _ = self.player.abort_stream(job.id);
                self.finish_job(job.id, Err(error)).await;
            }
        }
    }

    async fn wait_for_playback(&self, id: Uuid, path: PathBuf, duration: f64) {
        loop {
            let snapshot = self.player.snapshot();
            if let Some(error) = snapshot.error {
                let _ = self.player.abort_stream(id);
                let _ = fs::remove_file(&path);
                self.finish_job(id, Err(anyhow::anyhow!(error))).await;
                return;
            }
            if snapshot.job_id != Some(id)
                || matches!(snapshot.status, PlayerStatus::Finished | PlayerStatus::Idle)
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        if self
            .inner
            .read()
            .await
            .jobs
            .iter()
            .find(|job| job.id == id)
            .is_some_and(|job| job.state == JobState::Cancelled)
        {
            let _ = self.persist().await;
            self.emit("jobs");
            self.emit("playback");
            return;
        }
        self.finish_job(id, Ok((path, duration))).await;
    }

    async fn finish_job(&self, id: Uuid, result: Result<(PathBuf, f64)>) {
        let mut inner = self.inner.write().await;
        let Some(index) = inner.jobs.iter().position(|job| job.id == id) else {
            return;
        };
        if inner.jobs[index].state == JobState::Cancelled {
            if let Ok((path, _)) = result {
                let _ = fs::remove_file(path);
            }
            drop(inner);
            let _ = self.persist().await;
            self.emit("jobs");
            self.emit("playback");
            return;
        }
        let (audio_path, duration, error) = match result {
            Ok((path, duration)) => (Some(path.display().to_string()), Some(duration), None),
            Err(error) => (None, None, Some(error.to_string())),
        };
        if let Some(error) = &error {
            eprintln!("speech job {id} failed: {error}");
        }
        inner.jobs[index].state = if error.is_some() {
            JobState::Failed
        } else {
            JobState::Completed
        };
        inner.jobs[index].error = error;
        let completed = inner.jobs[index].clone();
        inner.recovery_attempts.remove(&id);
        let playback = self.player.snapshot();
        let spoken_chunks = if playback.job_id == Some(id) {
            playback.spoken_chunks
        } else {
            Vec::new()
        };
        inner.history.insert(
            0,
            HistoryItem {
                spoken_chunks,
                job: completed,
                audio_path,
                duration_seconds: duration,
                pinned: false,
            },
        );
        drop(inner);
        let _ = self.enforce_history_policy().await;
        let _ = self.persist().await;
        self.emit("history");
        self.emit("jobs");
        self.emit("playback");
    }

    async fn enforce_history_policy(&self) -> bool {
        let mut inner = self.inner.write().await;
        enforce_history_policy_inner(&mut inner, &self.audio_dir, chrono::Utc::now())
    }

    async fn persist(&self) -> Result<()> {
        write_persisted_state(&self.store_path, &*self.inner.read().await, true)
    }

    fn record_api_request(&self, scope: &'static str) -> Result<(), StatusCode> {
        let now = Instant::now();
        let mut limits = self
            .rate_limits
            .lock()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let attempts = limits.entry(scope).or_default();
        while attempts
            .front()
            .is_some_and(|instant| now.duration_since(*instant) > Duration::from_secs(60))
        {
            attempts.pop_front();
        }
        if attempts.len() >= 240 {
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
        attempts.push_back(now);
        Ok(())
    }

    fn emit(&self, resource: &str) {
        let _ = self.events.send(ServiceEvent {
            sequence: self.event_sequence.fetch_add(1, Ordering::Relaxed) + 1,
            resource: resource.to_owned(),
            occurred_at: chrono::Utc::now(),
        });
    }
}

fn load_persisted_state(store_path: &Path) -> Result<(InnerState, bool)> {
    if !store_path.is_file() {
        if recovery_state_exists(store_path) {
            return load_recovery_state(store_path, "primary journal is missing");
        }
        return Ok((InnerState::default(), false));
    }
    let primary = fs::read(store_path)
        .with_context(|| format!("failed to read service state at {}", store_path.display()))?;
    match serde_json::from_slice(&primary) {
        Ok(state) => Ok((state, false)),
        Err(primary_error) => load_recovery_state(
            store_path,
            &format!("primary journal is invalid: {primary_error}"),
        ),
    }
}

fn recovery_state_exists(store_path: &Path) -> bool {
    store_path.with_extension("json.tmp").is_file()
        || store_path.with_extension("json.bak").is_file()
}

fn load_recovery_state(store_path: &Path, primary_problem: &str) -> Result<(InnerState, bool)> {
    let candidates = [
        store_path.with_extension("json.tmp"),
        store_path.with_extension("json.bak"),
    ];
    let mut failures = Vec::new();
    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }
        match fs::read(&candidate)
            .with_context(|| format!("failed to read {}", candidate.display()))
            .and_then(|bytes| {
                serde_json::from_slice(&bytes)
                    .with_context(|| format!("invalid state at {}", candidate.display()))
            }) {
            Ok(state) => return Ok((state, true)),
            Err(error) => failures.push(error.to_string()),
        }
    }
    Err(anyhow::anyhow!(
        "unable to recover service state: {primary_problem}; {}",
        if failures.is_empty() {
            "no recovery journal exists".into()
        } else {
            failures.join("; ")
        }
    ))
}

fn recover_interrupted_jobs(inner: &mut InnerState, audio_dir: &Path) -> bool {
    let mut changed = false;
    let mut repeatedly_interrupted = Vec::new();
    for job in &mut inner.jobs {
        if !matches!(
            job.state,
            JobState::Synthesizing | JobState::Playing | JobState::Paused
        ) {
            continue;
        }
        let output = audio_dir.join(format!("{}.wav", job.id));
        if output.is_file() {
            let _ = fs::remove_file(output);
        }
        let attempts = inner.recovery_attempts.entry(job.id).or_default();
        if *attempts == 0 {
            *attempts = 1;
            job.state = JobState::Queued;
            job.error = None;
        } else {
            job.state = JobState::Failed;
            job.error = Some(
                "job was interrupted again after automatic crash recovery; retry it manually"
                    .into(),
            );
            repeatedly_interrupted.push(job.clone());
        }
        changed = true;
    }
    for job in repeatedly_interrupted {
        inner.recovery_attempts.remove(&job.id);
        if !inner.history.iter().any(|item| item.job.id == job.id) {
            inner.history.insert(
                0,
                HistoryItem {
                    spoken_chunks: Vec::new(),
                    job,
                    audio_path: None,
                    duration_seconds: None,
                    pinned: false,
                },
            );
        }
    }
    changed
}

fn write_persisted_state(store_path: &Path, state: &InnerState, rotate_backup: bool) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(state)?;
    let temporary = store_path.with_extension("json.tmp");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);

    if rotate_backup && store_path.is_file() {
        let backup = store_path.with_extension("json.bak");
        let backup_temporary = store_path.with_extension("json.bak.tmp");
        fs::copy(store_path, &backup_temporary)?;
        // Windows requires a writable handle for FlushFileBuffers, which is
        // what File::sync_all uses. Opening the copied backup read-only works
        // on Unix but fails with ERROR_ACCESS_DENIED on Windows.
        fs::OpenOptions::new()
            .write(true)
            .open(&backup_temporary)?
            .sync_all()?;
        replace_file(&backup_temporary, &backup)?;
    }
    replace_file(&temporary, store_path)?;
    sync_parent_directory(store_path)?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    if fs::rename(source, destination).is_ok() {
        return Ok(());
    }
    // Windows does not let std::fs::rename replace an existing file and
    // virus scanners/indexers can hold a newly synced journal briefly.
    // Retry the bounded replace while retaining the synced source file.
    let mut last_error = None;
    for attempt in 0..8 {
        if destination.is_file() {
            let _ = fs::remove_file(destination);
        }
        match fs::rename(source, destination) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        std::thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1)));
    }
    Err(last_error.expect("replacement attempted").into())
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn enforce_history_policy_inner(
    inner: &mut InnerState,
    audio_dir: &std::path::Path,
    now: chrono::DateTime<chrono::Utc>,
) -> bool {
    let mut changed = false;
    let retention_days = inner.settings.history_retention_days;
    if retention_days > 0 {
        let cutoff = now - chrono::Duration::days(i64::from(retention_days));
        let mut index = 0;
        while index < inner.history.len() {
            if !inner.history[index].pinned && inner.history[index].job.created_at < cutoff {
                let item = inner.history.remove(index);
                remove_archived_audio(&item, audio_dir);
                changed = true;
            } else {
                index += 1;
            }
        }
    }

    let quota = inner.settings.history_quota_bytes;
    let mut used = inner
        .history
        .iter()
        .filter_map(|item| item.audio_path.as_deref())
        .filter_map(|path| fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum::<u64>();
    while used > quota {
        let Some(index) = inner.history.iter().rposition(|item| !item.pinned) else {
            break;
        };
        let item = inner.history.remove(index);
        if let Some(path) = item.audio_path.as_deref() {
            let path = PathBuf::from(path);
            if path.starts_with(audio_dir)
                && let Ok(metadata) = fs::metadata(&path)
            {
                used = used.saturating_sub(metadata.len());
            }
        }
        remove_archived_audio(&item, audio_dir);
        changed = true;
    }
    changed
}

fn remove_archived_audio(item: &HistoryItem, audio_dir: &std::path::Path) {
    if let Some(path) = item.audio_path.as_deref() {
        let path = PathBuf::from(path);
        if path.starts_with(audio_dir) && path.is_file() {
            let _ = fs::remove_file(path);
        }
    }
}

pub fn router(state: ServiceState) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/models/{id}/backend", post(set_model_backend))
        .route("/v1/openapi.json", get(openapi))
        .route("/v1/events", get(events))
        .route("/v1/state", get(snapshot))
        .route("/v1/jobs", get(jobs).post(submit_job))
        .route("/v1/playback/pause", post(pause))
        .route("/v1/playback/resume", post(resume))
        .route("/v1/playback/stop", post(stop))
        .route("/v1/playback/seek", post(seek))
        .route("/v1/playback/rate", post(set_rate))
        .route("/v1/playback/volume", post(set_volume))
        .route("/v1/models", get(models))
        .route("/v1/models/imports/local", post(import_local_model))
        .route(
            "/v1/models/imports/huggingface",
            post(import_hugging_face_model),
        )
        .route("/v1/models/{id}/install", post(install_model))
        .route("/v1/models/{id}/language", post(select_model_language))
        .route("/v1/models/{id}/voice-mode", post(voice_modes::select))
        .route("/v1/models/{id}/update", post(install_model))
        .route("/v1/models/{id}/cancel", post(cancel_model_install))
        .route("/v1/models/{id}/select", post(select_model))
        .route("/v1/models/{id}/unload", post(unload_model))
        .route(
            "/v1/models/{id}/voices/{voice_id}/select",
            post(select_preset_voice),
        )
        .route(
            "/v1/models/{id}/voices/{voice_id}/preview",
            post(preview_preset_voice),
        )
        .route("/v1/models/{id}/benchmark", post(benchmark_model))
        .route("/v1/models/{id}", axum::routing::delete(remove_model))
        .route("/v1/voices", get(voices).post(create_voice))
        .route("/v1/voices/import", post(phonon_voices::import_voice))
        .route(
            "/v1/voices/discovery",
            get(voice_discovery::snapshot)
                .post(voice_discovery::start)
                .delete(voice_discovery::clear),
        )
        .route("/v1/voices/discovery/cancel", post(voice_discovery::cancel))
        .route(
            "/v1/voices/discovery/{id}/preview",
            post(voice_discovery::preview),
        )
        .route(
            "/v1/voices/discovery/{id}/save",
            post(voice_discovery::save),
        )
        .route("/v1/voices/{id}/generation", post(voice_discovery::tune))
        .route("/v1/voices/default/select", post(select_default_voice))
        .route("/v1/voices/{id}/select", post(select_voice))
        .route("/v1/voices/{id}/preview", post(preview_voice))
        .route("/v1/voices/{id}/rename", post(rename_voice))
        .route("/v1/voices/{id}/transcript", post(update_voice_transcript))
        .route("/v1/voices/{id}/tune", post(tune_voice))
        .route("/v1/voices/{id}", axum::routing::delete(delete_voice))
        .route("/v1/history", get(history))
        .route("/v1/history/{id}/pin", post(pin_history))
        .route("/v1/history/{id}/replay", post(replay_history))
        .route("/v1/history/{id}/regenerate", post(regenerate_history))
        .route("/v1/history/{id}", axum::routing::delete(delete_history))
        .route("/v1/settings", get(settings).post(update_settings))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .with_state(state)
}

async fn require_auth(
    State(state): State<ServiceState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    if request.uri().path() == "/v1/health" {
        return Ok(next.run(request).await);
    }
    let supplied = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    let control = format!("Bearer {}", state.api_token);
    let read = format!("Bearer {}", state.api_read_token);
    let scope = if supplied == Some(control.as_str()) {
        "control"
    } else if supplied == Some(read.as_str()) && request.method() == axum::http::Method::GET {
        "read"
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    state.record_api_request(scope)?;
    Ok(next.run(request).await)
}

fn load_or_create_api_token(data_dir: &std::path::Path) -> Result<String> {
    load_or_create_named_token(data_dir, "api-token")
}

fn load_or_create_named_token(data_dir: &std::path::Path, filename: &str) -> Result<String> {
    let path = data_dir.join(filename);
    if path.is_file() {
        let token = fs::read_to_string(&path)?.trim().to_owned();
        anyhow::ensure!(token.len() >= 32, "stored API token is invalid");
        return Ok(token);
    }
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    fs::write(&path, &token)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(token)
}

pub async fn serve(listener: TcpListener, state: ServiceState) -> Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        protocol_version: PROTOCOL_VERSION,
    })
}

async fn openapi() -> Json<serde_json::Value> {
    Json(
        serde_json::from_str(include_str!("../../../docs/openapi.json"))
            .expect("embedded OpenAPI document must be valid JSON"),
    )
}

async fn events(
    State(state): State<ServiceState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream =
        BroadcastStream::new(state.events.subscribe()).filter_map(|message| match message {
            Ok(event) => Some(Ok(Event::default()
                .id(event.sequence.to_string())
                .event("state_changed")
                .json_data(event)
                .expect("service event is serializable"))),
            Err(_) => None,
        });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn snapshot(State(state): State<ServiceState>) -> Json<ServiceSnapshot> {
    let player = state.player.snapshot();
    let inner = state.inner.read().await;
    let processing = inner
        .jobs
        .iter()
        .find(|job| job.state == JobState::Synthesizing);
    let (playback_state, active_job_id) = match player.status {
        PlayerStatus::Playing => (Some(JobState::Playing), player.job_id),
        PlayerStatus::Buffering => (
            Some(if player.duration_seconds > 0.0 {
                JobState::Playing
            } else {
                JobState::Synthesizing
            }),
            player.job_id,
        ),
        PlayerStatus::Paused => (Some(JobState::Paused), player.job_id),
        PlayerStatus::Finished => {
            // Completion includes publication to history. The audio callback
            // can finish shortly before the job monitor journals the result.
            // Untracked preset previews have no history entry to wait for.
            let archived = inner
                .jobs
                .iter()
                .find(|job| Some(job.id) == player.job_id)
                .is_none_or(|job| job.state == JobState::Completed);
            (
                Some(if archived {
                    JobState::Completed
                } else {
                    JobState::Playing
                }),
                player.job_id,
            )
        }
        PlayerStatus::Idle | PlayerStatus::Failed => (
            processing.map(|_| JobState::Synthesizing),
            processing.map(|job| job.id),
        ),
    };
    let spoken_text = active_job_id
        .and_then(|id| inner.jobs.iter().find(|job| job.id == id))
        .map(|job| job.text.clone())
        .or(player.preview_text)
        .unwrap_or_default();
    let current_title = speech_title(&spoken_text);
    let spoken_chunks = if player.spoken_chunks.is_empty() && player.generation_complete {
        playback_text_chunks(&spoken_text, player.duration_seconds)
    } else {
        player.spoken_chunks
    };
    Json(ServiceSnapshot {
        inference_device: state.last_inference_device.lock().unwrap().clone(),
        protocol_version: PROTOCOL_VERSION,
        active_model_id: inner.settings.active_model_id.clone(),
        queue_depth: inner
            .jobs
            .iter()
            .filter(|job| job.state == JobState::Queued)
            .count(),
        playback: PlaybackSnapshot {
            state: playback_state,
            job_id: active_job_id,
            position_seconds: player.position_seconds,
            duration_seconds: player.duration_seconds,
            rate: player.rate,
            volume: player.volume,
            buffering: player.status == PlayerStatus::Buffering,
            generation_complete: player.generation_complete,
            amplitudes: player.amplitudes,
            current_title,
            spoken_text,
            spoken_chunks,
        },
    })
}

fn speech_title(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let Some((readable_start, _)) = text
        .char_indices()
        .find(|(_, value)| value.is_alphanumeric())
    else {
        return "Untitled Recording".into();
    };
    let readable = text[readable_start..].trim();
    let first_line = readable.lines().next().unwrap_or(readable).trim();
    let sentence_end = first_line
        .char_indices()
        .find(|(_, value)| matches!(value, '.' | '!' | '?'))
        .map(|(index, value)| index + value.len_utf8())
        .unwrap_or(first_line.len());
    let normalized = first_line[..sentence_end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        return "Untitled Recording".into();
    }
    let characters = normalized.chars().collect::<Vec<_>>();
    if characters.len() <= 80 {
        normalized
    } else {
        let prefix = characters[..77]
            .iter()
            .collect::<String>()
            .trim_end()
            .to_owned();
        format!("{prefix}…")
    }
}

fn playback_text_chunks(text: &str, duration_seconds: f64) -> Vec<PlaybackTextChunk> {
    let characters = text.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < characters.len() {
        while start < characters.len() && characters[start].is_whitespace() {
            start += 1;
        }
        if start == characters.len() {
            break;
        }
        let hard_end = (start + 1_000).min(characters.len());
        let target_end = (start + 650).min(hard_end);
        let mut end = if hard_end == characters.len() {
            hard_end
        } else {
            (target_end..hard_end)
                .find(|index| matches!(characters[*index], '.' | '!' | '?' | '\n'))
                .map(|index| index + 1)
                .or_else(|| {
                    (start + 1..=target_end)
                        .rev()
                        .find(|index| characters[*index - 1].is_whitespace())
                })
                .unwrap_or(hard_end)
        };
        while end > start && characters[end - 1].is_whitespace() {
            end -= 1;
        }
        ranges.push((start, end));
        start = end;
    }
    let total_characters = characters.len() as f64;
    ranges
        .into_iter()
        .map(|(text_start, text_end)| PlaybackTextChunk {
            text_start,
            text_end,
            audio_end_seconds: Some(duration_seconds * text_end as f64 / total_characters),
            audio_start_seconds: if duration_seconds > 0.0 {
                duration_seconds * text_start as f64 / total_characters
            } else {
                0.0
            },
        })
        .collect()
}

async fn jobs(State(state): State<ServiceState>) -> Json<Vec<SpeechJob>> {
    Json(state.inner.read().await.jobs.iter().cloned().collect())
}

async fn history(State(state): State<ServiceState>) -> Json<Vec<HistoryItem>> {
    Json(state.inner.read().await.history.clone())
}

async fn pin_history(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
    Json(request): Json<PinRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut inner = state.inner.write().await;
    let Some(item) = inner.history.iter_mut().find(|item| item.job.id == id) else {
        return Err((StatusCode::NOT_FOUND, "history item not found".into()));
    };
    item.pinned = request.pinned;
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.emit("history");
    Ok(StatusCode::NO_CONTENT)
}

async fn regenerate_history(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<(StatusCode, Json<SpeechJob>), (StatusCode, String)> {
    let item = state
        .inner
        .read()
        .await
        .history
        .iter()
        .find(|item| item.job.id == id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, "history item not found".into()))?;
    // Older Qwen entries could retain a pace that the backend ignored.
    // Regenerate those at the natural pace actually used by that model.
    let model_config = match item.job.model_id.as_deref() {
        Some(id) => Some(state.models.config_path(id)),
        None => state.resolve_engine_config().await,
    };
    let speaking_pace = if let Some(path) = model_config
        && !EngineConfig::from_path(&path)
            .map_err(unprocessable)?
            .supports_native_speaking_pace()
    {
        Some(1.0)
    } else {
        item.job.speaking_pace
    };
    submit_job(
        State(state),
        Json(SpeechSubmission {
            voice_mode: Some(item.job.voice_mode),
            model_id: item.job.model_id,
            language: item.job.language,
            text: item.job.text,
            voice_profile_id: item.job.voice_profile_id,
            source: SpeechSource::History,
            queue_policy: QueuePolicy::Replace,
            confirmed_long_text: true,
            speaking_pace,
        }),
    )
    .await
}

async fn replay_history(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let (path, rate, volume, chunks) = {
        let inner = state.inner.read().await;
        let item = inner
            .history
            .iter()
            .find(|item| item.job.id == id)
            .ok_or((StatusCode::NOT_FOUND, "history item not found".into()))?;
        let path = item
            .audio_path
            .as_deref()
            .map(PathBuf::from)
            .filter(|path| path.starts_with(&state.audio_dir) && path.is_file())
            .ok_or((
                StatusCode::NOT_FOUND,
                "archived audio is unavailable".into(),
            ))?;
        (
            path,
            inner.settings.playback_rate,
            inner.settings.volume,
            item.spoken_chunks.clone(),
        )
    };
    let previously_active = state.player.snapshot().job_id;
    if previously_active.is_some_and(|active| active != id) {
        state.synthesis_cancelled.store(true, Ordering::Relaxed);
    }
    state.player.stop().map_err(internal_error)?;
    state
        .player
        .play(id, path, rate, volume)
        .map_err(internal_error)?;
    state
        .player
        .restore_chunks(id, chunks)
        .map_err(internal_error)?;
    {
        let mut inner = state.inner.write().await;
        mark_replay_started(&mut inner, id, previously_active);
    }
    state.persist().await.map_err(internal_error)?;
    monitor_replay(state.clone(), id);
    state.emit("jobs");
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

fn monitor_replay(replay_state: ServiceState, id: Uuid) {
    tokio::spawn(async move {
        loop {
            let player = replay_state.player.snapshot();
            if player.job_id != Some(id)
                || matches!(
                    player.status,
                    PlayerStatus::Finished | PlayerStatus::Idle | PlayerStatus::Failed
                )
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        {
            let mut inner = replay_state.inner.write().await;
            mark_replay_finished(&mut inner, id);
        }
        let _ = replay_state.persist().await;
        replay_state.emit("jobs");
        replay_state.emit("playback");
    });
}

fn mark_replay_started(inner: &mut InnerState, replay_id: Uuid, active_id: Option<Uuid>) {
    if let Some(active) = active_id.filter(|active| *active != replay_id)
        && let Some(job) = inner.jobs.iter_mut().find(|job| job.id == active)
        && matches!(
            job.state,
            JobState::Synthesizing | JobState::Playing | JobState::Paused
        )
    {
        job.state = JobState::Cancelled;
    }
    if let Some(job) = inner.jobs.iter_mut().find(|job| job.id == replay_id) {
        job.state = JobState::Playing;
    }
}

fn mark_replay_finished(inner: &mut InnerState, replay_id: Uuid) {
    if let Some(job) = inner.jobs.iter_mut().find(|job| job.id == replay_id)
        && job.state == JobState::Playing
    {
        job.state = JobState::Completed;
    }
}

async fn delete_history(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut inner = state.inner.write().await;
    let Some(index) = inner.history.iter().position(|item| item.job.id == id) else {
        return Err((StatusCode::NOT_FOUND, "history item not found".into()));
    };
    let item = inner.history.remove(index);
    drop(inner);
    if let Some(path) = item.audio_path {
        let path = PathBuf::from(path);
        if path.starts_with(&state.audio_dir) && path.is_file() {
            fs::remove_file(path).map_err(|error| internal_error(error.into()))?;
        }
    }
    state.persist().await.map_err(internal_error)?;
    state.emit("history");
    Ok(StatusCode::NO_CONTENT)
}

async fn settings(State(state): State<ServiceState>) -> Json<ServiceSettings> {
    Json(state.inner.read().await.settings.clone())
}

async fn update_settings(
    State(state): State<ServiceState>,
    Json(update): Json<ServiceSettingsUpdate>,
) -> Result<Json<ServiceSettings>, (StatusCode, String)> {
    if update
        .compute_device
        .as_deref()
        .is_some_and(|device| !matches!(device, "auto" | "cpu"))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "compute device must be auto or cpu".into(),
        ));
    }
    if update
        .playback_rate
        .is_some_and(|value| !(0.5..=3.0).contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "playback rate must be between 0.5 and 3.0".into(),
        ));
    }
    if update
        .speaking_pace
        .is_some_and(|value| ![0.75, 0.9, 1.0, 1.1, 1.25, 1.5].contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "speaking pace must be one of 0.75, 0.9, 1.0, 1.1, 1.25, or 1.5".into(),
        ));
    }
    if update
        .volume
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "volume must be between 0.0 and 1.0".into(),
        ));
    }
    if update
        .model_unload_delay_seconds
        .is_some_and(|value| !(30..=86_400).contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "model timeout must be between 30 seconds and 24 hours".into(),
        ));
    }
    if update
        .history_quota_bytes
        .is_some_and(|value| !(16_777_216..=107_374_182_400).contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "history quota must be between 16 MB and 100 GB".into(),
        ));
    }
    if update
        .history_retention_days
        .is_some_and(|value| value > 3_650)
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "history retention must be zero (forever) or at most 3,650 days".into(),
        ));
    }
    if update
        .long_text_confirmation_characters
        .is_some_and(|value| !(500..=1_000_000).contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "long-text threshold must be between 500 and 1,000,000 characters".into(),
        ));
    }
    let mut inner = state.inner.write().await;
    let compute_changed = update
        .compute_device
        .as_ref()
        .is_some_and(|value| *value != inner.settings.compute_device);
    if let Some(value) = update.compute_device {
        inner.settings.compute_device = value;
    }
    if let Some(value) = update.playback_rate {
        inner.settings.playback_rate = value;
    }
    if let Some(value) = update.speaking_pace {
        inner.settings.speaking_pace = value;
    }
    if let Some(value) = update.volume {
        inner.settings.volume = value;
    }
    if let Some(value) = update.model_unload_delay_seconds {
        inner.settings.model_unload_delay_seconds = value;
    }
    if let Some(value) = update.history_quota_bytes {
        inner.settings.history_quota_bytes = value;
    }
    if let Some(value) = update.history_retention_days {
        inner.settings.history_retention_days = value;
    }
    if let Some(value) = update.long_text_confirmation_characters {
        inner.settings.long_text_confirmation_characters = value;
    }
    if let Some(value) = update.text_cleaning {
        inner.settings.text_cleaning = value;
    }
    let settings = inner.settings.clone();
    drop(inner);
    if compute_changed {
        if let Ok(mut resident) = state.resident_engine.try_lock() {
            *resident = None;
        }
        *state.last_inference_device.lock().unwrap() = None;
        state.inference_runtime.reset_failures();
    }
    state
        .player
        .set_rate(settings.playback_rate)
        .map_err(unprocessable)?;
    state
        .player
        .set_volume(settings.volume)
        .map_err(unprocessable)?;
    let history_changed = state.enforce_history_policy().await;
    state.persist().await.map_err(internal_error)?;
    state.emit("settings");
    if history_changed {
        state.emit("history");
    }
    Ok(Json(settings))
}

async fn voices(State(state): State<ServiceState>) -> Json<Vec<VoiceProfile>> {
    let inner = state.inner.read().await;
    let active_model = inner.settings.active_model_id.as_deref();
    Json(
        inner
            .voices
            .iter()
            .filter(|voice| Some(voice.model_id.as_str()) == active_model)
            .cloned()
            .collect(),
    )
}

async fn select_default_voice(
    State(state): State<ServiceState>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut inner = state.inner.write().await;
    let model_id = inner
        .settings
        .active_model_id
        .clone()
        .ok_or((StatusCode::CONFLICT, "select a model first".into()))?;
    inner.settings.active_voice_profile_id = None;
    inner
        .settings
        .voice_profile_by_model
        .insert(model_id.clone(), None);
    inner.settings.voice_mode_by_model.remove(&model_id);
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn select_voice(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut inner = state.inner.write().await;
    let model_id = inner
        .settings
        .active_model_id
        .clone()
        .ok_or((StatusCode::CONFLICT, "select a model first".into()))?;
    let Some(voice) = inner.voices.iter().find(|voice| voice.id == id) else {
        return Err((StatusCode::NOT_FOUND, "voice profile not found".into()));
    };
    if voice.model_id != model_id {
        return Err((
            StatusCode::CONFLICT,
            "voice profile belongs to a different model".into(),
        ));
    }
    if let Some(language) = voice.language.clone() {
        state
            .models
            .select_language(&model_id, &language)
            .map_err(unprocessable)?;
    }
    inner.settings.voice_mode_by_model.remove(&model_id);
    inner.settings.active_voice_profile_id = Some(id);
    inner
        .settings
        .voice_profile_by_model
        .insert(model_id, Some(id));
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.emit("voices");
    state.emit("models");
    Ok(StatusCode::NO_CONTENT)
}

async fn preview_voice(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<(StatusCode, Json<SpeechJob>), (StatusCode, String)> {
    let voice = state
        .inner
        .read()
        .await
        .voices
        .iter()
        .find(|voice| voice.id == id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, "voice profile not found".into()))?;
    if state.inner.read().await.settings.active_model_id.as_deref() != Some(voice.model_id.as_str())
    {
        return Err((
            StatusCode::CONFLICT,
            "select the voice profile's model before previewing it".into(),
        ));
    }
    submit_job(
        State(state),
        Json(SpeechSubmission {
            voice_mode: None,
            model_id: None,
            language: None,
            text: format!("This is a preview of the voice {}.", voice.name),
            voice_profile_id: Some(id),
            source: SpeechSource::Desktop,
            queue_policy: QueuePolicy::Replace,
            confirmed_long_text: true,
            speaking_pace: None,
        }),
    )
    .await
}

async fn rename_voice(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
    Json(request): Json<VoiceRenameRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let name = request.name.trim();
    if name.is_empty() || name.chars().count() > 80 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "voice name must contain 1 to 80 characters".into(),
        ));
    }
    let mut inner = state.inner.write().await;
    let Some(voice) = inner.voices.iter_mut().find(|voice| voice.id == id) else {
        return Err((StatusCode::NOT_FOUND, "voice profile not found".into()));
    };
    voice.name = name.to_owned();
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn validate_reference_text(value: Option<&str>) -> Result<Option<String>, (StatusCode, String)> {
    let text = value.map(str::trim).filter(|text| !text.is_empty());
    if text.is_some_and(|text| text.chars().count() > 2000) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "reference transcript must contain at most 2000 characters".into(),
        ));
    }
    Ok(text.map(str::to_owned))
}

async fn update_voice_transcript(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
    Json(request): Json<VoiceTranscriptRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let reference_text = validate_reference_text(Some(&request.reference_text))?.ok_or((
        StatusCode::UNPROCESSABLE_ENTITY,
        "enter the exact words spoken in the reference recording".into(),
    ))?;
    let mut inner = state.inner.write().await;
    let voice = inner
        .voices
        .iter_mut()
        .find(|voice| voice.id == id)
        .ok_or((StatusCode::NOT_FOUND, "voice profile not found".into()))?;
    if voice.reference_kind != sayit_protocol::VoiceReferenceKind::Audio {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "A Phonon voice file does not use a reference transcript".into(),
        ));
    }
    voice.reference_text = Some(reference_text);
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.emit("voices");
    Ok(StatusCode::NO_CONTENT)
}

async fn tune_voice(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
    Json(request): Json<VoiceTuningRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    if !(3..=8).contains(&request.refinement_steps) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "PocketTTS refinement steps must be between 3 and 8".into(),
        ));
    }
    let mut inner = state.inner.write().await;
    let Some(voice) = inner.voices.iter_mut().find(|voice| voice.id == id) else {
        return Err((StatusCode::NOT_FOUND, "voice profile not found".into()));
    };
    if voice.reference_kind != sayit_protocol::VoiceReferenceKind::Audio {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "A Phonon voice file does not use PocketTTS refinement".into(),
        ));
    }
    voice.refinement_steps = request.refinement_steps;
    drop(inner);
    if let Ok(mut resident) = state.resident_engine.try_lock() {
        *resident = None;
    }
    state.persist().await.map_err(internal_error)?;
    state.emit("voices");
    Ok(StatusCode::NO_CONTENT)
}

async fn delete_voice(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut inner = state.inner.write().await;
    if inner
        .settings
        .voice_profile_by_model
        .values()
        .any(|selected| *selected == Some(id))
    {
        return Err((
            StatusCode::CONFLICT,
            "select another voice before deleting this profile".into(),
        ));
    }
    let Some(index) = inner.voices.iter().position(|voice| voice.id == id) else {
        return Err((StatusCode::NOT_FOUND, "voice profile not found".into()));
    };
    let voice = inner.voices.remove(index);
    drop(inner);
    let path = PathBuf::from(voice.reference_audio_path);
    if path.starts_with(&state.voices_dir) && path.is_file() {
        fs::remove_file(path).map_err(|error| internal_error(error.into()))?;
    }
    state.persist().await.map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn pause(State(state): State<ServiceState>) -> Result<StatusCode, (StatusCode, String)> {
    if matches!(
        state.player.snapshot().status,
        PlayerStatus::Playing | PlayerStatus::Buffering
    ) {
        state.player.pause().map_err(internal_error)?;
        set_active_job_state(&state, JobState::Paused).await;
        state.emit("playback");
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn resume(State(state): State<ServiceState>) -> Result<StatusCode, (StatusCode, String)> {
    let player = state.player.snapshot();
    if player.job_id.is_none() {
        return Ok(StatusCode::NO_CONTENT);
    }
    let completed = state
        .inner
        .read()
        .await
        .jobs
        .iter()
        .any(|job| Some(job.id) == player.job_id && job.state == JobState::Completed);
    state.player.resume().map_err(internal_error)?;
    set_active_job_state(&state, JobState::Playing).await;
    if completed && let Some(id) = player.job_id {
        monitor_replay(state.clone(), id);
    }
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn stop(State(state): State<ServiceState>) -> Result<StatusCode, (StatusCode, String)> {
    voice_discovery::cancel_active(&state);
    let active = state.player.snapshot().job_id;
    state.player.stop().map_err(internal_error)?;
    if let Some(job) = state.inner.write().await.jobs.iter_mut().find(|job| {
        (Some(job.id) == active || job.state == JobState::Synthesizing)
            && matches!(
                job.state,
                JobState::Synthesizing | JobState::Playing | JobState::Paused
            )
    }) {
        job.state = JobState::Cancelled;
        state.synthesis_cancelled.store(true, Ordering::Relaxed);
    }
    state.persist().await.map_err(internal_error)?;
    state.emit("jobs");
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn seek(
    State(state): State<ServiceState>,
    Json(request): Json<sayit_protocol::SecondsRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    if !request.seconds.is_finite() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "seconds must be finite".into(),
        ));
    }
    state.player.seek(request.seconds).map_err(internal_error)?;
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn set_rate(
    State(state): State<ServiceState>,
    Json(request): Json<sayit_protocol::PlaybackRateRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    state.player.set_rate(request.rate).map_err(unprocessable)?;
    state.inner.write().await.settings.playback_rate = request.rate;
    state.persist().await.map_err(internal_error)?;
    state.emit("settings");
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn set_volume(
    State(state): State<ServiceState>,
    Json(request): Json<sayit_protocol::VolumeRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .player
        .set_volume(request.volume)
        .map_err(unprocessable)?;
    state.inner.write().await.settings.volume = request.volume;
    state.persist().await.map_err(internal_error)?;
    state.emit("settings");
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn set_active_job_state(state: &ServiceState, job_state: JobState) {
    let active = state.player.snapshot().job_id;
    if let Some(job) = state
        .inner
        .write()
        .await
        .jobs
        .iter_mut()
        .find(|job| Some(job.id) == active)
    {
        job.state = job_state;
    }
    let _ = state.persist().await;
}

async fn create_voice(
    State(state): State<ServiceState>,
    Json(request): Json<VoiceCloneRequest>,
) -> Result<(StatusCode, Json<VoiceProfile>), (StatusCode, String)> {
    if !request.speaker_permission_confirmed {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "speaker permission must be confirmed before cloning".into(),
        ));
    }
    let (model_id, config_path) = {
        let inner = state.inner.read().await;
        let model_id = inner
            .settings
            .active_model_id
            .clone()
            .ok_or((StatusCode::CONFLICT, "select a model first".into()))?;
        (
            model_id,
            state
                .models
                .config_path(inner.settings.active_model_id.as_deref().unwrap()),
        )
    };
    let config = EngineConfig::from_path(&config_path).map_err(unprocessable)?;
    if !config.supports_voice_cloning() {
        return Err((
            StatusCode::CONFLICT,
            "the selected model does not support voice cloning".into(),
        ));
    }
    let reference_text = validate_reference_text(request.reference_text.as_deref())?;
    if config.is_qwen() && reference_text.is_none() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Qwen voice cloning requires the exact transcript of the reference recording".into(),
        ));
    }
    let name = request.name.trim();
    if name.is_empty() || name.chars().count() > 80 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "voice name must contain 1 to 80 characters".into(),
        ));
    }
    let source = PathBuf::from(&request.reference_audio_path);
    let id = Uuid::new_v4();
    let destination = state.voices_dir.join(format!("{id}.wav"));
    let maximum_seconds = if config.is_qwen() { 45.0 } else { 15.0 };
    let quality = analyze_and_trim_voice(&source, &destination, maximum_seconds)
        .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))?;
    let profile = VoiceProfile {
        generated: false,
        language: config
            .language()
            .and_then(|value| sayit_core::language::qwen_language(value).ok())
            .map(|language| language.code.to_owned()),
        generation_tuning: None,
        reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
        id,
        name: name.to_owned(),
        model_id,
        reference_audio_path: destination.display().to_string(),
        reference_text,
        reference_duration_seconds: quality.trimmed_duration_seconds,
        created_at: chrono::Utc::now(),
        quality,
        refinement_steps: 5,
    };
    state.inner.write().await.voices.push(profile.clone());
    state.persist().await.map_err(internal_error)?;
    state.emit("voices");
    Ok((StatusCode::CREATED, Json(profile)))
}

async fn models(State(state): State<ServiceState>) -> Json<Vec<ModelDescriptor>> {
    let inner = state.inner.read().await;
    let mut models = state
        .models
        .descriptors(inner.settings.active_model_id.as_deref(), &inner.benchmarks);
    drop(inner);
    let resident = state.resident_engine.try_lock().ok();
    for model in &mut models {
        if let Some(slot) = resident.as_deref().and_then(Option::as_ref)
            && slot.config_path == state.models.config_path(&model.id)
        {
            model.resident = true;
            model.inference_device = Some(inference_device(&slot.engine));
        }
    }
    Json(models)
}

async fn import_local_model(
    State(state): State<ServiceState>,
    Json(request): Json<LocalModelImportRequest>,
) -> Result<(StatusCode, Json<ModelDescriptor>), (StatusCode, String)> {
    if !request.untested_model_and_license_review_confirmed {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "confirm that community models are untested and that you reviewed the source license"
                .into(),
        ));
    }
    let id = state
        .models
        .import_local(
            Path::new(&request.directory),
            &request.display_name,
            &request.license,
            &request.license_url,
        )
        .map_err(unprocessable)?;
    let inner = state.inner.read().await;
    let descriptor = state
        .models
        .descriptors(inner.settings.active_model_id.as_deref(), &inner.benchmarks)
        .into_iter()
        .find(|model| model.id == id)
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "imported model was not cataloged".into(),
        ))?;
    drop(inner);
    state.emit("models");
    Ok((StatusCode::CREATED, Json(descriptor)))
}

async fn import_hugging_face_model(
    State(state): State<ServiceState>,
    Json(request): Json<HuggingFaceModelImportRequest>,
) -> Result<(StatusCode, Json<ModelDescriptor>), (StatusCode, String)> {
    if !request.untested_model_and_license_review_confirmed {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "confirm that community models are untested and that you reviewed the source license"
                .into(),
        ));
    }
    let manager = state.models.clone();
    let id = tokio::task::spawn_blocking(move || {
        manager.import_hugging_face(
            &request.repository,
            request.revision.as_deref(),
            &request.display_name,
            &request.license,
            &request.license_url,
            request.access_token.as_deref(),
        )
    })
    .await
    .map_err(|error| internal_error(error.into()))?
    .map_err(unprocessable)?;
    let inner = state.inner.read().await;
    let descriptor = state
        .models
        .descriptors(inner.settings.active_model_id.as_deref(), &inner.benchmarks)
        .into_iter()
        .find(|model| model.id == id)
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "imported model was not cataloged".into(),
        ))?;
    drop(inner);
    state.emit("models");
    Ok((StatusCode::CREATED, Json(descriptor)))
}

async fn benchmark_model(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<Json<ModelBenchmark>, (StatusCode, String)> {
    let config_path = state.models.config_path(&id);
    let config = EngineConfig::from_path(&config_path).map_err(unprocessable)?;
    let (reference, reference_text, prefer_gpu) = {
        let inner = state.inner.read().await;
        let voice_id = inner
            .settings
            .voice_profile_by_model
            .get(&id)
            .copied()
            .flatten()
            .or_else(|| {
                (inner.settings.active_model_id.as_deref() == Some(id.as_str()))
                    .then_some(inner.settings.active_voice_profile_id)
                    .flatten()
            });
        let voice = voice_id.and_then(|voice_id| {
            inner
                .voices
                .iter()
                .find(|voice| voice.id == voice_id && voice.model_id == id)
        });
        (
            voice.map(|voice| PathBuf::from(&voice.reference_audio_path)),
            voice.and_then(|voice| voice.reference_text.clone()),
            inner.settings.compute_device != "cpu",
        )
    };
    let requires_reference = config.requires_reference();
    let automatic_voice = reference.is_none() && voice_modes::supported(&config);
    if requires_reference && reference.is_none() && !automatic_voice {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "select a cloned voice before benchmarking this model".into(),
        ));
    }
    let output = state.audio_dir.join(format!("benchmark-{id}.wav"));
    let runtime = state.inference_runtime.clone();
    let last_device = state.last_inference_device.clone();
    let resident_engine = state.resident_engine.clone();
    let report = tokio::task::spawn_blocking(move || {
        let mut resident = resident_engine.lock().unwrap();
        *resident = None;
        let engine = ManagedEngine::load(&config, &runtime, prefer_gpu)?;
        let mut reading_voice = voice_modes::ReadingVoice::new(
            if automatic_voice {
                VoiceSynthesisMode::AutomaticStable
            } else {
                VoiceSynthesisMode::Standard
            },
            &config,
        )?;
        let voice_preparation = Instant::now();
        reading_voice.prepare(&engine, 0, &AtomicBool::new(false))?;
        let preparation_ms = voice_preparation.elapsed().as_secs_f64() * 1000.0;
        let generated_reference = reading_voice.reference();
        let result = BenchmarkRunner::run_with_reference(
            &engine,
            "The quick brown fox jumps over the lazy dog.",
            &output,
            3,
            generated_reference
                .map(|(audio, _)| audio)
                .or(reference.as_deref()),
            generated_reference
                .map(|(_, text)| text)
                .or(reference_text.as_deref()),
        )
        .map(|mut report| {
            report.cold_start_ms += preparation_ms;
            report
        });
        *last_device.lock().unwrap() = Some(inference_device(&engine));
        let _ = fs::remove_file(&output);
        result
    })
    .await
    .map_err(|error| internal_error(error.into()))?
    .map_err(internal_error)?;
    let benchmark = ModelBenchmark {
        iterations: report.iterations,
        cold_start_ms: report.cold_start_ms,
        p50_ms: report.p50_ms,
        p95_ms: report.p95_ms,
        p99_ms: report.p99_ms,
        mean_rtf: report.mean_rtf,
        audio_seconds: report.audio_seconds,
    };
    state
        .inner
        .write()
        .await
        .benchmarks
        .insert(id, benchmark.clone());
    state.persist().await.map_err(internal_error)?;
    state.emit("models");
    Ok(Json(benchmark))
}

async fn install_model(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    state.models.start_install(&id).map_err(unprocessable)?;
    Ok(StatusCode::ACCEPTED)
}

async fn select_model_language(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
    Json(request): Json<sayit_protocol::ModelLanguageRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .models
        .select_language(&id, &request.language)
        .map_err(unprocessable)?;
    state.emit("models");
    state.emit("settings");
    Ok(StatusCode::NO_CONTENT)
}

async fn cancel_model_install(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    state.models.cancel(&id).map_err(unprocessable)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn select_model(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let config = state.models.config_path(&id);
    EngineConfig::from_path(&config).map_err(unprocessable)?;
    voice_discovery::cancel_active(&state);
    *state.engine_config_path.write().await = Some(config);
    if let Ok(mut resident) = state.resident_engine.try_lock() {
        *resident = None;
    }
    let mut inner = state.inner.write().await;
    inner.settings.active_model_id = Some(id.clone());
    let selected = inner
        .settings
        .voice_profile_by_model
        .get(&id)
        .copied()
        .flatten()
        .filter(|voice_id| {
            inner
                .voices
                .iter()
                .any(|voice| voice.id == *voice_id && voice.model_id == id)
        });
    inner.settings.active_voice_profile_id = selected;
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn unload_model(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let config = state.models.config_path(&id);
    let mut resident = state.resident_engine.try_lock().map_err(|_| {
        (
            StatusCode::CONFLICT,
            "inference is busy; stop playback or wait before unloading".into(),
        )
    })?;
    if resident
        .as_ref()
        .is_some_and(|slot| slot.config_path == config)
    {
        *resident = None;
        drop(resident);
        state.emit("models");
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn select_preset_voice(
    State(state): State<ServiceState>,
    RoutePath((id, voice_id)): RoutePath<(String, String)>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .models
        .select_preset_voice(&id, &voice_id)
        .map_err(unprocessable)?;
    let config_path = state.models.config_path(&id);
    {
        if let Ok(mut resident) = state.resident_engine.try_lock()
            && resident
                .as_ref()
                .is_some_and(|slot| slot.config_path == config_path)
        {
            *resident = None;
        }
    }
    let mut inner = state.inner.write().await;
    if inner.settings.active_model_id.as_deref() == Some(id.as_str()) {
        inner.settings.active_voice_profile_id = None;
        inner.settings.voice_profile_by_model.insert(id, None);
    }
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.emit("models");
    state.emit("settings");
    Ok(StatusCode::NO_CONTENT)
}

async fn preview_preset_voice(
    State(state): State<ServiceState>,
    RoutePath((id, voice_id)): RoutePath<(String, String)>,
) -> Result<StatusCode, (StatusCode, String)> {
    if state.inner.read().await.settings.active_model_id.as_deref() != Some(id.as_str()) {
        return Err((
            StatusCode::CONFLICT,
            "select this voice's model before previewing it".into(),
        ));
    }
    let voice = state
        .models
        .descriptors(Some(&id), &HashMap::new())
        .into_iter()
        .find(|model| model.id == id)
        .and_then(|model| {
            model
                .preset_voices
                .into_iter()
                .find(|voice| voice.id == voice_id)
        })
        .ok_or((StatusCode::NOT_FOUND, "preset voice not found".into()))?;
    let config = state
        .models
        .preset_voice_config(&id, &voice_id)
        .map_err(unprocessable)?;
    let preview_id = Uuid::new_v4();
    let output = state.audio_dir.join(format!("preview-{preview_id}.wav"));
    let text = format!(
        "Hello! This is a preview of {}. Choose the voice that makes listening feel effortless.",
        voice.name
    );
    let preview_text = text.clone();
    let speaking_pace = if !config.supports_native_speaking_pace() {
        1.0
    } else {
        state.inner.read().await.settings.speaking_pace
    };
    let prefer_gpu = state.inner.read().await.settings.compute_device != "cpu";
    let runtime = state.inference_runtime.clone();
    let last_device = state.last_inference_device.clone();
    let resident_engine = state.resident_engine.clone();
    let synthesis_output = output.clone();
    tokio::task::spawn_blocking(move || {
        let mut resident = resident_engine.lock().unwrap();
        *resident = None;
        let engine = ManagedEngine::load(&config, &runtime, prefer_gpu)?;
        let result = engine.synthesize(&SynthesisRequest {
            generation: None,
            text: &text,
            output: &synthesis_output,
            reference_audio: None,
            reference_text: None,
            speaking_pace,
        });
        *last_device.lock().unwrap() = Some(inference_device(&engine));
        result
    })
    .await
    .map_err(|error| internal_error(error.into()))?
    .map_err(internal_error)?;
    let (rate, volume) = {
        let inner = state.inner.read().await;
        (inner.settings.playback_rate, inner.settings.volume)
    };
    let playback = state
        .player
        .play(preview_id, output, rate, volume)
        .map_err(unprocessable);
    let _ = fs::remove_file(state.audio_dir.join(format!("preview-{preview_id}.wav")));
    playback?;
    state
        .player
        .set_preview_text(preview_id, preview_text)
        .map_err(unprocessable)?;
    state.emit("state");
    Ok(StatusCode::ACCEPTED)
}

async fn remove_model(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let inner = state.inner.read().await;
    if inner.settings.active_model_id.as_deref() == Some(id.as_str()) {
        return Err((
            StatusCode::CONFLICT,
            "select another model before removing this one".into(),
        ));
    }
    if inner.jobs.iter().any(|job| {
        job.model_id.as_deref() == Some(id.as_str())
            && matches!(
                job.state,
                JobState::Queued | JobState::Synthesizing | JobState::Playing | JobState::Paused
            )
    }) {
        return Err((
            StatusCode::CONFLICT,
            "finish or cancel this model's queued readings before removing it".into(),
        ));
    }
    state.models.remove(&id).map_err(unprocessable)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn submit_job(
    State(state): State<ServiceState>,
    Json(submission): Json<SpeechSubmission>,
) -> Result<(StatusCode, Json<SpeechJob>), (StatusCode, String)> {
    let model_id =
        submission
            .model_id
            .clone()
            .or(state.inner.read().await.settings.active_model_id.clone());
    if model_id.as_deref().is_some_and(|id| {
        id.is_empty()
            || id.len() > 120
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    }) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid model identifier".into(),
        ));
    }
    let config_path = model_id
        .as_deref()
        .map(|id| state.models.config_path(id))
        .filter(|path| path.is_file())
        .ok_or((
            StatusCode::CONFLICT,
            "download and select a speech model before reading text aloud".into(),
        ))?;
    let config = EngineConfig::from_path(&config_path).map_err(unprocessable)?;
    let requires_voice_profile = config.requires_reference();
    let language = match submission.language.as_deref().or(config.language()) {
        Some(language) => {
            let mut checked = config.clone();
            checked.set_language(language).map_err(unprocessable)?;
            Some(
                sayit_core::language::qwen_language(language)
                    .map_err(unprocessable)?
                    .code
                    .to_owned(),
            )
        }
        None => None,
    };
    let phonon = matches!(config, EngineConfig::Phonon(_));
    let natural_pace = !config.supports_native_speaking_pace();
    if natural_pace && submission.speaking_pace.is_some_and(|pace| pace != 1.0) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "The selected model uses natural speaking pace; adjust playback speed to listen faster or slower"
                .into(),
        ));
    }
    let cleaning = state.inner.read().await.settings.text_cleaning;
    let cleaned = clean_text(
        &submission.text,
        TextCleaningOptions {
            enabled: cleaning.enabled,
            strip_markdown: cleaning.strip_markdown,
            strip_html: cleaning.strip_html,
            strip_code_blocks: cleaning.strip_code_blocks,
            strip_special_characters: cleaning.strip_special_characters,
            normalize_whitespace: cleaning.normalize_whitespace,
        },
    );
    let text = cleaned.text;
    if text.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "text must not be empty".into(),
        ));
    }
    if submission
        .speaking_pace
        .is_some_and(|value| ![0.75, 0.9, 1.0, 1.1, 1.25, 1.5].contains(&value))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "speaking pace must be one of 0.75, 0.9, 1.0, 1.1, 1.25, or 1.5".into(),
        ));
    }
    let mut inner = state.inner.write().await;
    if !config_path.is_file() {
        return Err((
            StatusCode::CONFLICT,
            "the requested model was removed before this reading could be queued".into(),
        ));
    }
    let character_count = text.chars().count();
    if character_count > 200_000 {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "cleaned text exceeds SayIt's 200,000-character maximum".into(),
        ));
    }
    if character_count > inner.settings.long_text_confirmation_characters
        && !submission.confirmed_long_text
    {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "text contains {character_count} characters; explicit confirmation is required above {}",
                inner.settings.long_text_confirmation_characters
            ),
        ));
    }
    let mut voice_profile_id = submission.voice_profile_id.or_else(|| {
        if model_id == inner.settings.active_model_id {
            inner.settings.active_voice_profile_id
        } else {
            model_id.as_ref().and_then(|id| {
                inner
                    .settings
                    .voice_profile_by_model
                    .get(id)
                    .copied()
                    .flatten()
            })
        }
    });
    let voice_mode = submission.voice_mode.unwrap_or_else(|| {
        if voice_profile_id.is_some() || !voice_modes::supported(&config) {
            VoiceSynthesisMode::Standard
        } else {
            model_id
                .as_ref()
                .and_then(|id| inner.settings.voice_mode_by_model.get(id).copied())
                .unwrap_or(VoiceSynthesisMode::AutomaticStable)
        }
    });
    if voice_mode != VoiceSynthesisMode::Standard {
        if !voice_modes::supported(&config) || submission.voice_profile_id.is_some() {
            return Err((StatusCode::UNPROCESSABLE_ENTITY,"Random reading requires native Qwen Base and cannot be combined with a saved profile override".into()));
        }
        voice_profile_id = None;
    }
    if requires_voice_profile
        && voice_profile_id.is_none()
        && voice_mode == VoiceSynthesisMode::Standard
    {
        return Err((
            StatusCode::CONFLICT,
            "The selected model requires a cloned voice; open Voices and create or select a voice profile first"
                .into(),
        ));
    }
    if let Some(voice_id) = voice_profile_id {
        let active_model = model_id.as_deref();
        let compatible = inner.voices.iter().any(|voice| {
            voice.id == voice_id
                && Some(voice.model_id.as_str()) == active_model
                && (voice.reference_kind == sayit_protocol::VoiceReferenceKind::PhononEmbedding)
                    == phonon
        });
        if !compatible {
            return Err((
                StatusCode::CONFLICT,
                "voice profile is not compatible with the requested model".into(),
            ));
        }
    }
    // Validate the complete request before replacement/interrupt side effects.
    if submission.queue_policy == QueuePolicy::Interrupt {
        let running = |job: &SpeechJob| {
            matches!(
                job.state,
                JobState::Synthesizing | JobState::Playing | JobState::Paused
            )
        };
        if state.player.snapshot().job_id.is_some() || inner.jobs.iter().any(running) {
            state.synthesis_cancelled.store(true, Ordering::Relaxed);
            state.player.stop().map_err(internal_error)?;
        }
        for job in inner.jobs.iter_mut().filter(|job| running(job)) {
            job.state = JobState::Cancelled;
        }
    }
    if matches!(
        submission.queue_policy,
        QueuePolicy::Replace | QueuePolicy::Interrupt
    ) {
        for queued in inner
            .jobs
            .iter_mut()
            .filter(|job| job.state == JobState::Queued)
        {
            queued.state = JobState::Cancelled;
        }
    }
    voice_discovery::cancel_active(&state);
    let job = SpeechJob {
        voice_mode,
        model_id,
        language,
        id: Uuid::new_v4(),
        text,
        source: submission.source,
        state: JobState::Queued,
        created_at: chrono::Utc::now(),
        error: None,
        voice_profile_id,
        speaking_pace: if natural_pace {
            Some(1.0)
        } else {
            submission.speaking_pace
        },
    };
    inner.jobs.push_back(job.clone());
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.notify.notify_one();
    state.emit("jobs");
    Ok((StatusCode::ACCEPTED, Json(job)))
}

async fn set_model_backend(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
    Json(request): Json<sayit_protocol::ModelBackendRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    state
        .models
        .start_backend(&id, &request.backend, request.precision.as_deref())
        .map_err(unprocessable)?;
    voice_discovery::cancel_active(&state);
    state.emit("models");
    Ok(StatusCode::ACCEPTED)
}

fn analyze_and_trim_voice(
    source: &std::path::Path,
    destination: &std::path::Path,
    maximum_seconds: f64,
) -> Result<VoiceQuality> {
    let mut reader = hound::WavReader::open(source).context("cannot read reference WAV")?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16,
        "reference WAV must contain 16-bit PCM audio"
    );
    anyhow::ensure!(spec.channels > 0, "reference WAV has no channels");
    let interleaved = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(!interleaved.is_empty(), "reference WAV is empty");
    let channels = spec.channels as usize;
    let mono = interleaved
        .chunks_exact(channels)
        .map(|frame| {
            let sum = frame.iter().map(|sample| *sample as i32).sum::<i32>();
            (sum / channels as i32) as i16
        })
        .collect::<Vec<_>>();
    let original_duration = mono.len() as f64 / spec.sample_rate as f64;
    let silence_threshold = (i16::MAX as f64 * 0.01) as i16;
    let first_signal = mono
        .iter()
        .position(|sample| sample.unsigned_abs() >= silence_threshold as u16)
        .context("reference WAV contains only silence")?;
    let last_signal = mono
        .iter()
        .rposition(|sample| sample.unsigned_abs() >= silence_threshold as u16)
        .unwrap();
    let padding = (spec.sample_rate / 10) as usize;
    let start = first_signal.saturating_sub(padding);
    let end = (last_signal + padding + 1).min(mono.len());
    let trimmed = &mono[start..end];
    let trimmed_duration = trimmed.len() as f64 / spec.sample_rate as f64;
    anyhow::ensure!(
        (3.0..=maximum_seconds).contains(&trimmed_duration),
        "usable speech after silence trimming must be 3 to {maximum_seconds:.0} seconds; received {trimmed_duration:.2} seconds"
    );
    let scale = i16::MAX as f64;
    let peak = trimmed
        .iter()
        .map(|sample| sample.unsigned_abs() as f64 / scale)
        .fold(0.0, f64::max);
    let rms = (trimmed
        .iter()
        .map(|sample| (*sample as f64 / scale).powi(2))
        .sum::<f64>()
        / trimmed.len() as f64)
        .sqrt();
    let clipped = trimmed
        .iter()
        .filter(|sample| sample.unsigned_abs() as f64 / scale >= 0.98)
        .count() as f64
        / trimmed.len() as f64
        * 100.0;
    let peak_dbfs = 20.0 * peak.max(1e-6).log10();
    let rms_dbfs = 20.0 * rms.max(1e-6).log10();
    let mut issues = Vec::new();
    let mut score = 100u8;
    if spec.sample_rate < 16_000 {
        issues.push("Sample rate is below 16 kHz; record at 24 kHz or higher.".into());
        score = score.saturating_sub(25);
    }
    if rms_dbfs < -35.0 {
        issues.push("Speech is very quiet; move closer to the microphone.".into());
        score = score.saturating_sub(20);
    }
    if clipped > 0.1 {
        issues.push("Audio contains clipped samples; lower the recording gain.".into());
        score = score.saturating_sub(30);
    }
    if original_duration - trimmed_duration > 2.0 {
        issues.push("More than two seconds of leading or trailing silence were removed.".into());
        score = score.saturating_sub(10);
    }
    let mut writer = hound::WavWriter::create(
        destination,
        hound::WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in trimmed {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(VoiceQuality {
        sample_rate: spec.sample_rate,
        original_duration_seconds: original_duration,
        trimmed_duration_seconds: trimmed_duration,
        peak_dbfs,
        rms_dbfs,
        clipped_sample_percent: clipped,
        quality_score: score,
        issues,
    })
}

fn internal_error(error: anyhow::Error) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

fn unprocessable(error: anyhow::Error) -> (StatusCode, String) {
    (StatusCode::UNPROCESSABLE_ENTITY, error.to_string())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn api_requires_the_per_user_bearer_token() {
        let state = ServiceState::default();
        let token = state.api_token.to_string();
        let read_token = state.api_read_token.to_string();
        let app = router(state);
        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/state")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let authorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/state")
                    .header(header::AUTHORIZATION, format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
        let read_only_mutation = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/settings")
                    .header(header::AUTHORIZATION, format!("Bearer {read_token}"))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(read_only_mutation.status(), StatusCode::UNAUTHORIZED);
        let health = app
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_rate_limits_each_token_scope() {
        let state = ServiceState::default();
        let token = state.api_token.to_string();
        state.rate_limits.lock().unwrap().insert(
            "control",
            std::iter::repeat_n(Instant::now(), 240).collect(),
        );
        let response = router(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/state")
                    .header(header::AUTHORIZATION, format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn event_bus_sequences_resource_changes() {
        let state = ServiceState::default();
        let mut receiver = state.events.subscribe();
        state.emit("jobs");
        state.emit("playback");
        let first = receiver.recv().await.unwrap();
        let second = receiver.recv().await.unwrap();
        assert_eq!(first.sequence, 1);
        assert_eq!(first.resource, "jobs");
        assert_eq!(second.sequence, 2);
        assert_eq!(second.resource, "playback");
    }

    #[tokio::test]
    async fn openapi_document_identifies_v1_and_every_resource_family() {
        let Json(document) = openapi().await;
        assert_eq!(document["openapi"], "3.1.0");
        assert_eq!(document["info"]["version"], "1.0.0");
        let paths = document["paths"].as_object().unwrap();
        for path in [
            "/health",
            "/events",
            "/state",
            "/jobs",
            "/playback/{action}",
            "/models",
            "/models/imports/local",
            "/models/imports/huggingface",
            "/voices",
            "/history",
            "/settings",
        ] {
            assert!(paths.contains_key(path), "OpenAPI is missing {path}");
        }
        assert!(document["components"]["securitySchemes"]["readToken"].is_object());
        assert!(document["components"]["securitySchemes"]["controlToken"].is_object());
    }

    #[test]
    fn playback_metadata_matches_sayit_title_and_unicode_chunk_contract() {
        let title = speech_title(
            "*** A first sentence with a useful title. A second sentence should not appear.",
        );
        assert_eq!(title, "A first sentence with a useful title.");
        assert_eq!(speech_title("!!!"), "Untitled Recording");
        let long_title = speech_title(&"é".repeat(100));
        assert_eq!(long_title.chars().count(), 78);
        assert!(long_title.ends_with('…'));

        let text = format!("{} {}", "🙂word ".repeat(120), "ending".repeat(90));
        let chunks = playback_text_chunks(&text, 24.0);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].text_start, 0);
        assert!(chunks.iter().all(|chunk| chunk.text_end > chunk.text_start));
        assert!(
            chunks
                .windows(2)
                .all(|pair| pair[0].text_end <= pair[1].text_start
                    && pair[0].audio_start_seconds < pair[1].audio_start_seconds)
        );
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.text_end - chunk.text_start <= 1_000)
        );
    }

    #[tokio::test]
    async fn snapshot_exposes_the_job_while_speech_is_being_generated() {
        let state = ServiceState::default();
        let job_id = Uuid::new_v4();
        state.inner.write().await.jobs.push_back(SpeechJob {
            voice_mode: Default::default(),
            model_id: None,
            language: None,
            id: job_id,
            text: "Visible while synthesis is running.".into(),
            source: SpeechSource::Selection,
            state: JobState::Synthesizing,
            created_at: chrono::Utc::now(),
            error: None,
            voice_profile_id: None,
            speaking_pace: None,
        });
        let Json(snapshot) = snapshot(State(state)).await;
        assert_eq!(snapshot.playback.state, Some(JobState::Synthesizing));
        assert_eq!(snapshot.playback.job_id, Some(job_id));
        assert_eq!(
            snapshot.playback.spoken_text,
            "Visible while synthesis is running."
        );
    }

    #[test]
    fn replay_cancels_previous_audio_and_never_overwrites_an_explicit_stop() {
        let active_id = Uuid::new_v4();
        let replay_id = Uuid::new_v4();
        let make_job = |id, state| SpeechJob {
            voice_mode: Default::default(),
            model_id: None,
            language: None,
            id,
            text: id.to_string(),
            source: SpeechSource::History,
            state,
            created_at: chrono::Utc::now(),
            error: None,
            voice_profile_id: None,
            speaking_pace: None,
        };
        let mut inner = InnerState::default();
        inner.jobs.push_back(make_job(active_id, JobState::Playing));
        inner
            .jobs
            .push_back(make_job(replay_id, JobState::Completed));

        mark_replay_started(&mut inner, replay_id, Some(active_id));
        assert_eq!(inner.jobs[0].state, JobState::Cancelled);
        assert_eq!(inner.jobs[1].state, JobState::Playing);

        inner.jobs[1].state = JobState::Cancelled;
        mark_replay_finished(&mut inner, replay_id);
        assert_eq!(inner.jobs[1].state, JobState::Cancelled);

        inner.jobs[1].state = JobState::Playing;
        mark_replay_finished(&mut inner, replay_id);
        assert_eq!(inner.jobs[1].state, JobState::Completed);

        inner.jobs[0].state = JobState::Synthesizing;
        mark_replay_started(&mut inner, replay_id, Some(active_id));
        assert_eq!(inner.jobs[0].state, JobState::Cancelled);
    }

    #[tokio::test]
    async fn state_is_persisted_across_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let submission = SpeechSubmission {
            voice_mode: None,
            model_id: None,
            language: None,
            text: "persist me".into(),
            voice_profile_id: None,
            source: Default::default(),
            queue_policy: QueuePolicy::Append,
            confirmed_long_text: false,
            speaking_pace: None,
        };
        let job = SpeechJob {
            voice_mode: Default::default(),
            model_id: None,
            language: None,
            id: Uuid::new_v4(),
            text: submission.text,
            source: submission.source,
            state: JobState::Queued,
            created_at: chrono::Utc::now(),
            error: None,
            voice_profile_id: submission.voice_profile_id,
            speaking_pace: submission.speaking_pace,
        };
        state.inner.write().await.jobs.push_back(job);
        state.persist().await.unwrap();
        let reopened = ServiceState::open(directory.path().to_owned(), None).unwrap();
        assert_eq!(reopened.inner.read().await.jobs.len(), 1);
    }

    #[tokio::test]
    async fn startup_requeues_interrupted_jobs_once_and_removes_partial_audio() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let job_id = Uuid::new_v4();
        state.inner.write().await.jobs.push_back(SpeechJob {
            voice_mode: Default::default(),
            model_id: None,
            language: None,
            id: job_id,
            text: "recover after a crash".into(),
            source: SpeechSource::Clipboard,
            state: JobState::Playing,
            created_at: chrono::Utc::now(),
            error: Some("stale error".into()),
            voice_profile_id: None,
            speaking_pace: None,
        });
        let partial = directory
            .path()
            .join("history")
            .join(format!("{job_id}.wav"));
        fs::write(&partial, b"partial output").unwrap();
        state.persist().await.unwrap();

        let recovered = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let inner = recovered.inner.read().await;
        assert_eq!(inner.jobs[0].state, JobState::Queued);
        assert_eq!(inner.jobs[0].error, None);
        assert_eq!(inner.recovery_attempts.get(&job_id), Some(&1));
        assert!(!partial.exists());
        drop(inner);

        recovered.inner.write().await.jobs[0].state = JobState::Synthesizing;
        recovered.persist().await.unwrap();
        let stopped_retrying = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let inner = stopped_retrying.inner.read().await;
        assert_eq!(inner.jobs[0].state, JobState::Failed);
        assert!(
            inner.jobs[0]
                .error
                .as_deref()
                .unwrap()
                .contains("interrupted again")
        );
        assert_eq!(inner.history.len(), 1);
        assert_eq!(inner.history[0].job.id, job_id);
        assert!(!inner.recovery_attempts.contains_key(&job_id));
    }

    #[tokio::test]
    async fn startup_recovers_from_the_last_valid_state_backup() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        state.inner.write().await.jobs.push_back(SpeechJob {
            voice_mode: Default::default(),
            model_id: None,
            language: None,
            id: Uuid::new_v4(),
            text: "survives journal corruption".into(),
            source: SpeechSource::Api,
            state: JobState::Queued,
            created_at: chrono::Utc::now(),
            error: None,
            voice_profile_id: None,
            speaking_pace: None,
        });
        state.persist().await.unwrap();
        state.inner.write().await.settings.volume = 0.75;
        state.persist().await.unwrap();
        state.inner.write().await.settings.volume = 0.5;
        let newest_state = serde_json::to_vec_pretty(&*state.inner.read().await).unwrap();
        fs::write(
            directory.path().join("service-state.json.tmp"),
            newest_state,
        )
        .unwrap();
        fs::write(directory.path().join("service-state.json"), b"{torn").unwrap();

        let recovered = ServiceState::open(directory.path().to_owned(), None).unwrap();
        assert_eq!(recovered.inner.read().await.jobs.len(), 1);
        assert_eq!(recovered.inner.read().await.settings.volume, 0.5);
        let primary = directory.path().join("service-state.json");
        let bytes = fs::read(&primary).unwrap();
        serde_json::from_slice::<InnerState>(&bytes).unwrap();

        fs::remove_file(&primary).unwrap();
        let recovered_without_primary =
            ServiceState::open(directory.path().to_owned(), None).unwrap();
        assert_eq!(recovered_without_primary.inner.read().await.jobs.len(), 1);
        assert!(primary.is_file());
    }

    #[tokio::test]
    async fn voice_cloning_requires_speaker_permission() {
        let state = ServiceState::default();
        let request = VoiceCloneRequest {
            name: "Someone else".into(),
            reference_audio_path: "missing.wav".into(),
            reference_text: None,
            speaker_permission_confirmed: false,
        };
        let error = create_voice(State(state), Json(request)).await.unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(error.1.contains("permission"));
    }

    fn qwen_voice_test_state(directory: &Path) -> ServiceState {
        let worker = directory.join("worker.py");
        fs::write(
            &worker,
            r#"import json, pathlib, sys, wave
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    output = pathlib.Path(request["output"])
    pathlib.Path(sys.argv[sys.argv.index("--model") + 1], "last-request.json").write_text(json.dumps(request))
    with wave.open(str(output), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "NONE"))
        wav.writeframes(b"\0\0" * 160)
    print(json.dumps({"ok": True}), flush=True)
"#,
        )
        .unwrap();
        let config_path = directory.join("qwen.json");
        let config = EngineConfig::Qwen3Tts(sayit_core::Qwen3TtsConfig {
            python: if cfg!(target_os = "linux") {
                "/usr/bin/python3".into()
            } else {
                worker.clone()
            },
            worker,
            model: directory.into(),
            mode: sayit_core::Qwen3TtsMode::VoiceClone,
            device: "cpu".into(),
            dtype: "float32".into(),
            language: "Auto".into(),
            speaker: None,
            voice_description: None,
        });
        fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let state = ServiceState::open(directory.join("data"), Some(config_path)).unwrap();
        state.inner.try_write().unwrap().settings.active_model_id = Some("qwen3-tts".into());
        write_persisted_state(&state.store_path, &state.inner.try_read().unwrap(), false).unwrap();
        state
    }

    #[tokio::test]
    async fn model_is_captured_and_history_regenerates_with_it_after_selection_changes() {
        let state = ServiceState::default();
        let primary = "piper-en-us-lessac-medium";
        let alternate = "alternate-model";
        let path = state.models.config_path(alternate);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::copy(state.models.config_path(primary), path).unwrap();
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"A queued reading.","queue_policy":"append"}),
        )
        .unwrap();
        let (_, Json(first)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        assert_eq!(first.model_id.as_deref(), Some(primary));
        select_model(State(state.clone()), RoutePath(alternate.into()))
            .await
            .unwrap();
        let (_, Json(second)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        assert_eq!(second.model_id.as_deref(), Some(alternate));
        assert_eq!(
            remove_model(State(state.clone()), RoutePath(primary.into()))
                .await
                .unwrap_err()
                .0,
            StatusCode::CONFLICT
        );
        assert!(state.models.config_path(primary).is_file());
        let (_, Json(explicit)) = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                model_id: Some(primary.into()),
                ..request
            }),
        )
        .await
        .unwrap();
        assert_eq!(explicit.model_id.as_deref(), Some(primary));
        assert_eq!(
            state.inner.read().await.settings.active_model_id.as_deref(),
            Some(alternate)
        );
        state.inner.write().await.history.push(HistoryItem {
            job: SpeechJob {
                state: JobState::Completed,
                ..first.clone()
            },
            audio_path: None,
            duration_seconds: None,
            pinned: false,
            spoken_chunks: Vec::new(),
        });
        let (_, Json(regenerated)) = regenerate_history(State(state.clone()), RoutePath(first.id))
            .await
            .unwrap();
        assert_eq!(regenerated.model_id.as_deref(), Some(primary));
        assert_eq!(
            state.inner.read().await.settings.active_model_id.as_deref(),
            Some(alternate)
        );
        state.persist().await.unwrap();
        let reopened = ServiceState::open(state.store_path.parent().unwrap().into(), None).unwrap();
        let inner = reopened.inner.read().await;
        assert_eq!(inner.history[0].job.model_id.as_deref(), Some(primary));
        assert_eq!(
            inner
                .jobs
                .iter()
                .find(|job| job.id == regenerated.id)
                .unwrap()
                .model_id
                .as_deref(),
            Some(primary)
        );
    }

    #[tokio::test]
    async fn rejected_model_or_profile_overrides_do_not_interrupt_or_replace_existing_work() {
        let state = ServiceState::default();
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"An existing reading.","queue_policy":"append"}),
        )
        .unwrap();
        let (_, Json(queued)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        let active = Uuid::new_v4();
        state.inner.write().await.jobs.push_back(SpeechJob {
            id: active,
            state: JobState::Synthesizing,
            ..queued.clone()
        });
        for policy in [QueuePolicy::Replace, QueuePolicy::Interrupt] {
            for (model_id, voice_profile_id, expected) in [
                (
                    Some("../invalid".into()),
                    None,
                    StatusCode::UNPROCESSABLE_ENTITY,
                ),
                (Some("uninstalled-model".into()), None, StatusCode::CONFLICT),
                (None, Some(Uuid::new_v4()), StatusCode::CONFLICT),
            ] {
                let error = submit_job(
                    State(state.clone()),
                    Json(SpeechSubmission {
                        model_id,
                        voice_profile_id,
                        queue_policy: policy.clone(),
                        ..request.clone()
                    }),
                )
                .await
                .unwrap_err();
                assert_eq!(error.0, expected);
                let inner = state.inner.read().await;
                assert_eq!(inner.jobs.len(), 2);
                assert_eq!(inner.jobs[0].state, JobState::Queued);
                assert_eq!(inner.jobs[1].state, JobState::Synthesizing);
                assert!(!state.synthesis_cancelled.load(Ordering::Relaxed));
            }
        }
    }

    #[tokio::test]
    async fn interrupt_preserves_completed_records_and_cancels_only_unfinished_work() {
        let state = ServiceState::default();
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"A completed reading.","queue_policy":"append"}),
        )
        .unwrap();
        let (_, Json(completed)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        {
            let mut inner = state.inner.write().await;
            inner.jobs[0].state = JobState::Completed;
            inner.jobs.push_back(SpeechJob {
                id: Uuid::new_v4(),
                state: JobState::Synthesizing,
                ..completed.clone()
            });
            inner.jobs.push_back(SpeechJob {
                id: Uuid::new_v4(),
                state: JobState::Paused,
                ..completed.clone()
            });
        }
        let (_, Json(next)) = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                queue_policy: QueuePolicy::Interrupt,
                ..request
            }),
        )
        .await
        .unwrap();
        let inner = state.inner.read().await;
        assert_eq!(inner.jobs[0].state, JobState::Completed);
        assert_eq!(inner.jobs[1].state, JobState::Cancelled);
        assert_eq!(inner.jobs[2].state, JobState::Cancelled);
        assert_eq!(inner.jobs[3].id, next.id);
        assert_eq!(inner.jobs[3].state, JobState::Queued);
    }

    #[tokio::test]
    async fn concurrent_removal_before_enqueue_rejects_the_job_without_queue_side_effects() {
        let state = ServiceState::default();
        let primary = "piper-en-us-lessac-medium";
        let alternate = state.models.config_path("alternate-model");
        fs::create_dir_all(alternate.parent().unwrap()).unwrap();
        fs::copy(state.models.config_path(primary), alternate).unwrap();
        select_model(State(state.clone()), RoutePath("alternate-model".into()))
            .await
            .unwrap();
        // A remover holds the same read lock while it checks in-flight work.
        let removing = state.inner.read().await;
        let background = state.clone();
        let queued = tokio::spawn(async move {
            submit_job(
                State(background),
                Json(
                    serde_json::from_value(
                        serde_json::json!({"text":"A racing submission.","model_id":primary}),
                    )
                    .unwrap(),
                ),
            )
            .await
        });
        // Tokio's fair lock makes a later reader wait once submission reaches
        // the enqueue write lock; no production timing hooks are required.
        let start = Instant::now();
        loop {
            if tokio::time::timeout(std::time::Duration::from_millis(5), state.inner.read())
                .await
                .is_err()
            {
                break;
            }
            assert!(start.elapsed() < std::time::Duration::from_secs(5));
            tokio::task::yield_now().await;
        }
        fs::remove_file(state.models.config_path(primary)).unwrap();
        drop(removing);
        assert_eq!(queued.await.unwrap().unwrap_err().0, StatusCode::CONFLICT);
        assert!(state.inner.read().await.jobs.is_empty());
    }

    #[tokio::test]
    async fn model_override_uses_its_saved_profile_and_language_without_changing_active_selection()
    {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let voice = Uuid::new_v4();
        state.inner.write().await.voices.push(VoiceProfile {
            id: voice,
            name: "Saved narrator".into(),
            model_id: "qwen3-tts".into(),
            generated: false,
            language: None,
            generation_tuning: None,
            reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
            reference_audio_path: "reference.wav".into(),
            reference_text: Some("Exact reference text.".into()),
            reference_duration_seconds: 4.0,
            created_at: chrono::Utc::now(),
            quality: VoiceQuality::default(),
            refinement_steps: 5,
        });
        select_voice(State(state.clone()), RoutePath(voice))
            .await
            .unwrap();
        state.models.select_language("qwen3-tts", "es").unwrap();
        let piper = ServiceState::default();
        let path = state.models.config_path("piper-en-us-lessac-medium");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::copy(piper.models.config_path("piper-en-us-lessac-medium"), path).unwrap();
        select_model(
            State(state.clone()),
            RoutePath("piper-en-us-lessac-medium".into()),
        )
        .await
        .unwrap();
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"Buenos días.","model_id":"qwen3-tts"}),
        )
        .unwrap();
        let (_, Json(job)) = submit_job(State(state.clone()), Json(request))
            .await
            .unwrap();
        assert_eq!(job.model_id.as_deref(), Some("qwen3-tts"));
        assert_eq!(job.language.as_deref(), Some("es"));
        assert_eq!(job.voice_profile_id, Some(voice));
        let inner = state.inner.read().await;
        assert_eq!(
            inner.settings.active_model_id.as_deref(),
            Some("piper-en-us-lessac-medium")
        );
        assert!(inner.settings.active_voice_profile_id.is_none());
    }

    pub(super) fn native_voice_test_state(directory: &Path) -> ServiceState {
        let model = directory.join("native-model");
        fs::create_dir_all(model.join("int4")).unwrap();
        for file in sayit_core::onnx_runtime::model_assets().files {
            if file.path.starts_with("fp32/") {
                continue;
            }
            fs::write(model.join(file.path), b"test fixture").unwrap();
        }
        let config = EngineConfig::Qwen3TtsOnnx(sayit_core::QwenOnnxConfig {
            worker: directory.join("worker"),
            model,
            runtime: None,
            device: "auto".into(),
            precision: "int4".into(),
            num_threads: 4,
            language: "Auto".into(),
        });
        let path = directory.join("native.json");
        fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        ServiceState::open(directory.join("data"), Some(path)).unwrap()
    }

    #[tokio::test]
    async fn native_config_bootstraps_and_preserves_long_voice_profiles_across_restart() {
        let directory = tempfile::tempdir().unwrap();
        let state = native_voice_test_state(directory.path());
        assert_eq!(
            state.inner.read().await.settings.active_model_id.as_deref(),
            Some("qwen3-tts-06b-base")
        );
        let source = directory.path().join("reference.wav");
        let mut writer = hound::WavWriter::create(
            &source,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..30 * 16000 {
            writer.write_sample(1000i16).unwrap();
        }
        writer.finalize().unwrap();
        let request = VoiceCloneRequest {
            name: "Native reference".into(),
            reference_audio_path: source.display().to_string(),
            reference_text: Some("The reference transcript.".into()),
            speaker_permission_confirmed: true,
        };
        let error = create_voice(
            State(state.clone()),
            Json(VoiceCloneRequest {
                reference_text: None,
                ..request.clone()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        let (_, Json(voice)) = create_voice(State(state.clone()), Json(request))
            .await
            .unwrap();
        assert!(voice.reference_duration_seconds > 15.0);
        select_voice(State(state.clone()), RoutePath(voice.id))
            .await
            .unwrap();
        let descriptor = state
            .models
            .descriptors(None, &HashMap::new())
            .into_iter()
            .find(|model| model.id == voice.model_id)
            .unwrap();
        assert_eq!(descriptor.inference_backend.as_deref(), Some("onnx"));
        assert_eq!(descriptor.weight_precision.as_deref(), Some("int4"));
        let reopened = ServiceState::open(directory.path().join("data"), None).unwrap();
        assert_eq!(
            reopened.inner.read().await.settings.active_voice_profile_id,
            Some(voice.id)
        );
        let Json(profiles) = voices(State(reopened)).await;
        assert_eq!(profiles[0].id, voice.id);
        assert_eq!(profiles[0].reference_text, voice.reference_text);
    }

    #[tokio::test]
    async fn pytorch_benchmark_requires_a_voice_before_runtime_preparation() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let error = benchmark_model(State(state), RoutePath("qwen3-tts".into()))
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(!directory.path().join("data/runtimes").exists());
    }

    #[tokio::test]
    async fn invalid_native_backend_requests_fail_before_downloads() {
        let state = ServiceState::default();
        for (id, backend, precision) in [
            ("pocket-tts-int8", "onnx", Some("int4")),
            ("qwen3-tts-06b-base", "unknown", None),
            ("qwen3-tts-06b-base", "onnx", Some("invalid")),
        ] {
            let result = set_model_backend(
                State(state.clone()),
                RoutePath(id.into()),
                Json(sayit_protocol::ModelBackendRequest {
                    backend: backend.into(),
                    precision: precision.map(str::to_owned),
                }),
            )
            .await;
            assert_eq!(result.unwrap_err().0, StatusCode::UNPROCESSABLE_ENTITY);
        }
    }

    #[tokio::test]
    async fn stopping_synthesis_preserves_cancelled_state_and_discards_output() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let id = Uuid::new_v4();
        state.inner.write().await.jobs.push_back(SpeechJob {
            voice_mode: Default::default(),
            model_id: None,
            language: None,
            id,
            text: "hello".into(),
            source: SpeechSource::Cli,
            state: JobState::Synthesizing,
            created_at: chrono::Utc::now(),
            error: None,
            voice_profile_id: None,
            speaking_pace: None,
        });
        stop(State(state.clone())).await.unwrap();
        assert!(state.synthesis_cancelled.load(Ordering::Relaxed));
        let output = state.audio_dir.join("cancelled.wav");
        fs::write(&output, b"discarded result").unwrap();
        state.finish_job(id, Ok((output.clone(), 1.0))).await;
        assert!(!output.exists());
        let inner = state.inner.read().await;
        assert_eq!(inner.jobs[0].state, JobState::Cancelled);
        assert!(inner.history.is_empty());
    }

    #[tokio::test]
    async fn qwen_voice_creation_requires_a_bounded_nonempty_transcript() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        // Older clients omit reference_text. They get an actionable validation
        // error rather than silently creating a less faithful Qwen profile.
        let request: VoiceCloneRequest = serde_json::from_value(serde_json::json!({
            "name": "My voice", "reference_audio_path": "missing.wav",
            "speaker_permission_confirmed": true
        }))
        .unwrap();
        for reference_text in [None, Some(" \n ".into()), Some("字".repeat(2001))] {
            let error = create_voice(
                State(state.clone()),
                Json(VoiceCloneRequest {
                    reference_text,
                    ..request.clone()
                }),
            )
            .await
            .unwrap_err();
            assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
            assert!(error.1.contains("transcript"), "{}", error.1);
        }
        assert!(state.inner.read().await.voices.is_empty());
        assert_eq!(fs::read_dir(&state.voices_dir).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn language_is_captured_when_queued_and_invalid_overrides_do_not_enter_the_queue() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let voice = Uuid::new_v4();
        state.inner.write().await.voices.push(VoiceProfile {
            generated: false,
            language: None,
            generation_tuning: None,
            id: voice,
            name: "Reference".into(),
            model_id: "qwen3-tts".into(),
            reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
            reference_audio_path: "reference.wav".into(),
            reference_text: Some("Reference text.".into()),
            reference_duration_seconds: 4.0,
            created_at: chrono::Utc::now(),
            quality: VoiceQuality::default(),
            refinement_steps: 5,
        });
        state.models.select_language("qwen3-tts", "es").unwrap();
        state.inner.write().await.settings.speaking_pace = 1.25;
        let submission = SpeechSubmission {
            voice_mode: None,
            model_id: None,
            text: "Buenos días.".into(),
            language: None,
            voice_profile_id: Some(voice),
            source: SpeechSource::Api,
            queue_policy: QueuePolicy::Append,
            confirmed_long_text: false,
            speaking_pace: None,
        };
        let (_, Json(first)) = submit_job(State(state.clone()), Json(submission.clone()))
            .await
            .unwrap();
        assert_eq!(first.language.as_deref(), Some("es"));
        assert_eq!(first.speaking_pace, Some(1.0));
        state.models.select_language("qwen3-tts", "fr").unwrap();
        let (_, Json(second)) = submit_job(State(state.clone()), Json(submission.clone()))
            .await
            .unwrap();
        assert_eq!(second.language.as_deref(), Some("fr"));
        assert_eq!(
            state.inner.read().await.jobs[0].language.as_deref(),
            Some("es")
        );
        let count = state.inner.read().await.jobs.len();
        let error = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                language: Some("unknown".into()),
                ..submission
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(state.inner.read().await.jobs.len(), count);
        let error = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                speaking_pace: Some(1.25),
                text: "Unsupported pace.".into(),
                voice_profile_id: Some(voice),
                source: SpeechSource::Api,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: false,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(state.inner.read().await.jobs.len(), count);
        state.inner.write().await.history.push(HistoryItem {
            job: SpeechJob {
                speaking_pace: Some(1.25),
                state: JobState::Completed,
                ..first.clone()
            },
            audio_path: None,
            duration_seconds: None,
            pinned: false,
            spoken_chunks: Vec::new(),
        });
        let (_, Json(regenerated)) = regenerate_history(State(state.clone()), RoutePath(first.id))
            .await
            .unwrap();
        assert_eq!(regenerated.language.as_deref(), Some("es"));
        assert_eq!(regenerated.speaking_pace, Some(1.0));
        assert_eq!(
            EngineConfig::from_path(&state.models.config_path("qwen3-tts"))
                .unwrap()
                .language(),
            Some("French")
        );
        state.persist().await.unwrap();
        let restored = ServiceState::open(directory.path().join("data"), None).unwrap();
        assert_eq!(
            restored.inner.read().await.jobs[0].language.as_deref(),
            Some("es")
        );
    }

    #[tokio::test]
    async fn voice_transcripts_survive_creation_restart_and_editing_legacy_profiles() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let source = directory.path().join("reference.wav");
        let mut writer = hound::WavWriter::create(
            &source,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..64_000 {
            writer.write_sample(1000i16).unwrap();
        }
        writer.finalize().unwrap();
        let (_, Json(voice)) = create_voice(
            State(state.clone()),
            Json(VoiceCloneRequest {
                name: "My voice".into(),
                reference_audio_path: source.display().to_string(),
                reference_text: Some("  Buenos días. Esta es mi voz.\n ".into()),
                speaker_permission_confirmed: true,
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            voice.reference_text.as_deref(),
            Some("Buenos días. Esta es mi voz.")
        );
        let reopened = ServiceState::open(directory.path().join("data"), None).unwrap();
        assert_eq!(
            reopened.inner.read().await.voices[0].reference_text,
            voice.reference_text
        );

        // Existing installations have profiles without the new field.
        let mut legacy = serde_json::to_value(&*reopened.inner.read().await).unwrap();
        legacy["voices"][0]
            .as_object_mut()
            .unwrap()
            .remove("reference_text");
        fs::write(&state.store_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        let legacy_state = ServiceState::open(directory.path().join("data"), None).unwrap();
        assert!(
            legacy_state.inner.read().await.voices[0]
                .reference_text
                .is_none()
        );
        for text in [" \n ".into(), "字".repeat(2001)] {
            let error = update_voice_transcript(
                State(legacy_state.clone()),
                RoutePath(voice.id),
                Json(VoiceTranscriptRequest {
                    reference_text: text,
                }),
            )
            .await
            .unwrap_err();
            assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        }
        let response = router(legacy_state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/voices/{}/transcript", voice.id))
                    .header(
                        header::AUTHORIZATION,
                        format!("Bearer {}", legacy_state.api_token),
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"reference_text":"  Esta es mi voz original.  "}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        let edited = ServiceState::open(directory.path().join("data"), None).unwrap();
        assert_eq!(
            edited.inner.read().await.voices[0]
                .reference_text
                .as_deref(),
            Some("Esta es mi voz original.")
        );
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn saved_voice_transcript_reaches_qwen_worker_and_does_not_leak_between_profiles() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let id = Uuid::new_v4();
        state.inner.write().await.voices.push(VoiceProfile {
            generated: false,
            language: None,
            generation_tuning: None,
            reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
            id,
            name: "My voice".into(),
            model_id: "qwen3-tts".into(),
            reference_audio_path: "reference.wav".into(),
            reference_text: Some("Mi voz, con su transcripción exacta.".into()),
            reference_duration_seconds: 4.0,
            created_at: chrono::Utc::now(),
            quality: VoiceQuality::default(),
            refinement_steps: 5,
        });
        state.persist().await.unwrap();
        let state = ServiceState::open(directory.path().join("data"), None).unwrap();
        // Startup rebinds packaged runtimes. Restore the test worker instead
        // of starting the production worker or loading any model weights.
        fs::copy(
            directory.path().join("qwen.json"),
            state.models.config_path("qwen3-tts"),
        )
        .unwrap();
        for transcript in [Some("Mi voz, con su transcripción exacta."), None] {
            state.inner.write().await.voices[0].reference_text = transcript.map(str::to_owned);
            let job = SpeechJob {
                voice_mode: Default::default(),
                model_id: None,
                language: Some("es".into()),
                id: Uuid::new_v4(),
                text: "Text to generate, different from the reference transcript.".into(),
                source: SpeechSource::Desktop,
                state: JobState::Synthesizing,
                created_at: chrono::Utc::now(),
                error: None,
                voice_profile_id: Some(id),
                speaking_pace: None,
            };
            // Keep this synthesis out of the playback queue so the test never
            // opens an audio device. The normal synthesis path still runs.
            state.process_job(job.clone()).await;
            let payload: serde_json::Value = serde_json::from_slice(
                &fs::read(directory.path().join("last-request.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(payload["text"], job.text);
            assert_eq!(payload["language"], "Spanish");
            assert_eq!(payload["reference_audio"], "reference.wav");
            assert_eq!(
                payload.get("reference_text").and_then(|v| v.as_str()),
                transcript
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn benchmark_uses_its_models_saved_reference_without_switching_active_model() {
        let directory = tempfile::tempdir().unwrap();
        let state = qwen_voice_test_state(directory.path());
        let id = Uuid::new_v4();
        {
            let mut inner = state.inner.write().await;
            inner.settings.active_model_id = Some("piper-en-us-lessac-medium".into());
            inner.settings.active_voice_profile_id = None;
            inner
                .settings
                .voice_profile_by_model
                .insert("qwen3-tts".into(), Some(id));
            inner.voices.push(VoiceProfile {
                generated: false,
                language: None,
                generation_tuning: None,
                reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
                id,
                name: "Reference".into(),
                model_id: "qwen3-tts".into(),
                reference_audio_path: "reference.wav".into(),
                reference_text: Some("Exact reference words.".into()),
                reference_duration_seconds: 4.0,
                created_at: chrono::Utc::now(),
                quality: VoiceQuality::default(),
                refinement_steps: 5,
            });
        }
        let Json(report) = benchmark_model(State(state.clone()), RoutePath("qwen3-tts".into()))
            .await
            .unwrap();
        assert_eq!(report.iterations, 3);
        let request: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.path().join("last-request.json")).unwrap())
                .unwrap();
        assert_eq!(request["reference_audio"], "reference.wav");
        assert_eq!(request["reference_text"], "Exact reference words.");
        assert_eq!(
            state.inner.read().await.settings.active_model_id.as_deref(),
            Some("piper-en-us-lessac-medium")
        );
        assert_eq!(
            state
                .last_inference_device
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .provider,
            "cpu"
        );
    }

    #[tokio::test]
    async fn pocket_requires_a_profile_only_until_a_default_reference_is_installed() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("pocket.json");
        let asset = directory.path().join("unused");
        fs::write(&asset, []).unwrap();
        let mut config = EngineConfig::SherpaOnnxPocket(sayit_core::PocketConfig {
            reference_audio: None,
            preset_voice: None,
            executable: asset.clone(),
            lm_flow: asset.clone(),
            lm_main: asset.clone(),
            encoder: asset.clone(),
            decoder: asset.clone(),
            text_conditioner: asset.clone(),
            vocab_json: asset.clone(),
            token_scores_json: asset,
            provider: "cpu".into(),
            num_threads: 1,
            num_steps: 5,
        });
        fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let state =
            ServiceState::open(directory.path().join("data"), Some(config_path.clone())).unwrap();

        let error = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                text: "This must not enter the queue.".into(),
                voice_profile_id: None,
                source: SpeechSource::Desktop,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: false,
                speaking_pace: None,
            }),
        )
        .await
        .unwrap_err();

        assert_eq!(error.0, StatusCode::CONFLICT);
        assert!(error.1.contains("requires a cloned voice"));
        assert!(state.inner.read().await.jobs.is_empty());

        let EngineConfig::SherpaOnnxPocket(pocket) = &mut config else {
            unreachable!()
        };
        pocket.reference_audio = Some(pocket.encoder.clone());
        pocket.preset_voice = Some("alba".into());
        // Both the active copy and original config are deliberately updated;
        // the service resolves the managed copy after bootstrap.
        for path in [config_path, state.models.config_path("pocket-tts-int8")] {
            fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
        }
        state.inner.write().await.settings.speaking_pace = 1.25;
        let submission = |pace| {
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                text: "A preset is ready without a saved profile.".into(),
                voice_profile_id: None,
                source: SpeechSource::Desktop,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: false,
                speaking_pace: pace,
            })
        };
        let (_, Json(job)) = submit_job(State(state.clone()), submission(None))
            .await
            .unwrap();
        assert_eq!(job.voice_profile_id, None);
        assert_eq!(job.speaking_pace, Some(1.0));
        let error = submit_job(State(state), submission(Some(1.25)))
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(error.1.contains("natural speaking pace"));
    }

    #[tokio::test]
    async fn cloned_voice_refinement_is_validated_and_persisted() {
        let state = ServiceState::default();
        let id = Uuid::new_v4();
        state.inner.write().await.voices.push(VoiceProfile {
            generated: false,
            language: None,
            generation_tuning: None,
            reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
            id,
            name: "Tunable".into(),
            model_id: "pocket-tts-int8".into(),
            reference_audio_path: "voice.wav".into(),
            reference_text: None,
            reference_duration_seconds: 4.0,
            created_at: chrono::Utc::now(),
            quality: VoiceQuality::default(),
            refinement_steps: 5,
        });
        let invalid = tune_voice(
            State(state.clone()),
            RoutePath(id),
            Json(VoiceTuningRequest {
                refinement_steps: 9,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(invalid.0, StatusCode::UNPROCESSABLE_ENTITY);
        tune_voice(
            State(state.clone()),
            RoutePath(id),
            Json(VoiceTuningRequest {
                refinement_steps: 7,
            }),
        )
        .await
        .unwrap();
        assert_eq!(state.inner.read().await.voices[0].refinement_steps, 7);
    }

    #[test]
    fn voice_import_trims_silence_and_reports_signal_quality() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.wav");
        let destination = directory.path().join("trimmed.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&source, spec).unwrap();
        for index in 0..80_000 {
            let sample = if (8_000..72_000).contains(&index) {
                if index % 2 == 0 { 4_000 } else { -4_000 }
            } else {
                0
            };
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let quality = analyze_and_trim_voice(&source, &destination, 15.0).unwrap();
        assert!(destination.is_file());
        assert!((4.1..=4.3).contains(&quality.trimmed_duration_seconds));
        assert_eq!(quality.sample_rate, 16_000);
        assert_eq!(quality.clipped_sample_percent, 0.0);
        assert!(quality.quality_score >= 90);
    }

    #[tokio::test]
    async fn long_text_requires_explicit_confirmation() {
        let state = ServiceState::default();
        state
            .inner
            .write()
            .await
            .settings
            .long_text_confirmation_characters = 500;
        let submission = SpeechSubmission {
            voice_mode: None,
            model_id: None,
            language: None,
            text: "x".repeat(501),
            voice_profile_id: None,
            source: SpeechSource::Selection,
            queue_policy: QueuePolicy::Replace,
            confirmed_long_text: false,
            speaking_pace: None,
        };
        let error = submit_job(State(state.clone()), Json(submission.clone()))
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::PAYLOAD_TOO_LARGE);
        let confirmed = SpeechSubmission {
            language: None,
            confirmed_long_text: true,
            ..submission
        };
        assert!(submit_job(State(state), Json(confirmed)).await.is_ok());
    }

    #[tokio::test]
    async fn submission_without_an_installed_model_is_rejected_before_history() {
        let storage = tempfile::tempdir().unwrap();
        let state = ServiceState::open(storage.path().into(), None).unwrap();
        let error = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                text: "This must not become a failed history item.".into(),
                voice_profile_id: None,
                source: SpeechSource::Clipboard,
                queue_policy: QueuePolicy::Replace,
                confirmed_long_text: false,
                speaking_pace: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::CONFLICT);
        assert!(state.inner.read().await.jobs.is_empty());
        assert!(state.inner.read().await.history.is_empty());
    }

    #[tokio::test]
    async fn queue_policy_append_and_replace_have_distinct_behavior() {
        let state = ServiceState::default();
        for text in ["first", "second"] {
            let _ = submit_job(
                State(state.clone()),
                Json(SpeechSubmission {
                    voice_mode: None,
                    model_id: None,
                    language: None,
                    text: text.into(),
                    voice_profile_id: None,
                    source: SpeechSource::Api,
                    queue_policy: QueuePolicy::Append,
                    confirmed_long_text: false,
                    speaking_pace: None,
                }),
            )
            .await
            .unwrap();
        }
        let _ = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                text: "replacement".into(),
                voice_profile_id: None,
                source: SpeechSource::Api,
                queue_policy: QueuePolicy::Replace,
                confirmed_long_text: false,
                speaking_pace: None,
            }),
        )
        .await
        .unwrap();
        let inner = state.inner.read().await;
        assert_eq!(
            inner
                .jobs
                .iter()
                .filter(|job| job.state == JobState::Cancelled)
                .count(),
            2
        );
        assert_eq!(
            inner
                .jobs
                .iter()
                .filter(|job| job.state == JobState::Queued)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn settings_update_is_validated_and_persisted() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let update = ServiceSettingsUpdate {
            compute_device: Some("cpu".into()),
            history_quota_bytes: Some(64 * 1_048_576),
            history_retention_days: Some(90),
            long_text_confirmation_characters: Some(4_000),
            speaking_pace: Some(1.25),
            text_cleaning: Some(sayit_protocol::TextCleaningSettings {
                strip_code_blocks: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        let _ = update_settings(State(state), Json(update)).await.unwrap();
        let reopened = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let settings = &reopened.inner.read().await.settings;
        assert_eq!(settings.history_quota_bytes, 64 * 1_048_576);
        assert_eq!(settings.history_retention_days, 90);
        assert_eq!(settings.long_text_confirmation_characters, 4_000);
        assert_eq!(settings.speaking_pace, 1.25);
        assert_eq!(settings.compute_device, "cpu");
        assert!(!settings.text_cleaning.strip_code_blocks);
    }

    #[tokio::test]
    async fn invalid_compute_device_is_rejected_without_changing_policy() {
        let state = ServiceState::default();
        let error = update_settings(
            State(state.clone()),
            Json(ServiceSettingsUpdate {
                compute_device: Some("invented-gpu".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(state.inner.read().await.settings.compute_device, "auto");
    }

    #[test]
    fn longer_qwen_references_preserve_pocket_limits_and_remain_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("reference.wav");
        let destination = directory.path().join("trimmed.wav");
        for seconds in [31, 46] {
            let mut writer = hound::WavWriter::create(
                &source,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for _ in 0..16000 * seconds {
                writer.write_sample(1000i16).unwrap();
            }
            writer.finalize().unwrap();
            let result = analyze_and_trim_voice(&source, &destination, 45.0);
            if seconds == 31 {
                assert_eq!(result.unwrap().trimmed_duration_seconds, 31.0);
            } else {
                assert!(result.unwrap_err().to_string().contains("3 to 45"));
            }
            assert!(
                analyze_and_trim_voice(&source, &destination, 15.0)
                    .unwrap_err()
                    .to_string()
                    .contains("3 to 15")
            );
        }
    }

    #[tokio::test]
    async fn submitted_text_is_cleaned_before_confirmation_queue_and_history() {
        let state = ServiceState::default();
        let fence = char::from(96).to_string().repeat(3);
        let raw = format!(
            "# Readable heading\nA **useful** [link](https://example.com).\u{200b}\n{fence}\n{}\n{fence}",
            "ignored code ".repeat(100)
        );
        state
            .inner
            .write()
            .await
            .settings
            .long_text_confirmation_characters = 500;
        let (_, Json(job)) = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                text: raw,
                voice_profile_id: None,
                source: SpeechSource::Selection,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: false,
                speaking_pace: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(job.text, "Readable heading\nA useful link.");
        assert_eq!(state.inner.read().await.jobs.back().unwrap().text, job.text);
    }

    #[tokio::test]
    async fn text_cleaning_can_be_disabled_without_skipping_safety_limits() {
        let state = ServiceState::default();
        {
            let mut inner = state.inner.write().await;
            inner.settings.text_cleaning.enabled = false;
            inner.settings.long_text_confirmation_characters = 500;
        }
        let raw = "  # Keep **formatting**  ";
        let (_, Json(job)) = submit_job(
            State(state),
            Json(SpeechSubmission {
                voice_mode: None,
                model_id: None,
                language: None,
                text: raw.into(),
                voice_profile_id: None,
                source: SpeechSource::Api,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: false,
                speaking_pace: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(job.text, "# Keep **formatting**");
    }

    #[tokio::test]
    async fn startup_expires_only_old_unpinned_history_and_its_managed_audio() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let now = chrono::Utc::now();
        let expired_path = state.audio_dir.join("expired.wav");
        let pinned_path = state.audio_dir.join("pinned.wav");
        let fresh_path = state.audio_dir.join("fresh.wav");
        fs::write(&expired_path, b"expired").unwrap();
        fs::write(&pinned_path, b"pinned").unwrap();
        fs::write(&fresh_path, b"fresh").unwrap();
        let make_item = |text: &str, created_at, path: &std::path::Path, pinned| HistoryItem {
            spoken_chunks: Vec::new(),
            job: SpeechJob {
                voice_mode: Default::default(),
                model_id: None,
                language: None,
                id: Uuid::new_v4(),
                text: text.into(),
                source: SpeechSource::Desktop,
                state: JobState::Completed,
                created_at,
                error: None,
                voice_profile_id: None,
                speaking_pace: None,
            },
            audio_path: Some(path.display().to_string()),
            duration_seconds: Some(1.0),
            pinned,
        };
        {
            let mut inner = state.inner.write().await;
            inner.settings.history_retention_days = 30;
            inner.history = vec![
                make_item("fresh", now - chrono::Duration::days(2), &fresh_path, false),
                make_item(
                    "pinned",
                    now - chrono::Duration::days(90),
                    &pinned_path,
                    true,
                ),
                make_item(
                    "expired",
                    now - chrono::Duration::days(31),
                    &expired_path,
                    false,
                ),
            ];
        }
        state.persist().await.unwrap();
        drop(state);

        let reopened = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let inner = reopened.inner.read().await;
        assert_eq!(
            inner
                .history
                .iter()
                .map(|item| item.job.text.as_str())
                .collect::<Vec<_>>(),
            vec!["fresh", "pinned"]
        );
        assert!(!expired_path.exists());
        assert!(pinned_path.exists());
        assert!(fresh_path.exists());
        drop(inner);

        let persisted: InnerState =
            serde_json::from_slice(&fs::read(directory.path().join("service-state.json")).unwrap())
                .unwrap();
        assert_eq!(persisted.history.len(), 2);
    }

    #[tokio::test]
    async fn changing_retention_applies_immediately() {
        let state = ServiceState::default();
        state.inner.write().await.history.push(HistoryItem {
            spoken_chunks: Vec::new(),
            job: SpeechJob {
                voice_mode: Default::default(),
                model_id: None,
                language: None,
                id: Uuid::new_v4(),
                text: "old".into(),
                source: SpeechSource::Desktop,
                state: JobState::Completed,
                created_at: chrono::Utc::now() - chrono::Duration::days(10),
                error: None,
                voice_profile_id: None,
                speaking_pace: None,
            },
            audio_path: None,
            duration_seconds: None,
            pinned: false,
        });
        let _ = update_settings(
            State(state.clone()),
            Json(ServiceSettingsUpdate {
                history_retention_days: Some(7),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert!(state.inner.read().await.history.is_empty());
    }

    #[tokio::test]
    async fn selected_model_survives_service_restart() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let model_dir = directory.path().join("fake-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.onnx"), []).unwrap();
        fs::write(model_dir.join("tokens.txt"), []).unwrap();
        let config_path = state.models.config_path("piper-en-us-lessac-medium");
        fs::create_dir_all(config_path.parent().unwrap()).unwrap();
        let config = EngineConfig::SherpaOnnxVits(sayit_core::VitsConfig {
            executable: "tts".into(),
            model: model_dir.join("model.onnx"),
            tokens: model_dir.join("tokens.txt"),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 1,
            speaker_id: 0,
        });
        fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        select_model(State(state), RoutePath("piper-en-us-lessac-medium".into()))
            .await
            .unwrap();
        let reopened = ServiceState::open(directory.path().to_owned(), None).unwrap();
        assert_eq!(*reopened.engine_config_path.read().await, Some(config_path));
    }

    #[tokio::test]
    async fn model_switch_restores_its_voice_selection_across_restart() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let assets = directory.path().join("voice-model-assets");
        fs::create_dir_all(&assets).unwrap();
        for name in [
            "model", "tokens", "flow", "main", "encoder", "decoder", "text", "vocab", "scores",
        ] {
            fs::write(assets.join(name), []).unwrap();
        }
        let piper_path = state.models.config_path("piper-en-us-lessac-medium");
        let pocket_path = state.models.config_path("pocket-tts-int8");
        fs::create_dir_all(piper_path.parent().unwrap()).unwrap();
        fs::create_dir_all(pocket_path.parent().unwrap()).unwrap();
        let piper = EngineConfig::SherpaOnnxVits(sayit_core::VitsConfig {
            executable: "tts".into(),
            model: assets.join("model"),
            tokens: assets.join("tokens"),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 1,
            speaker_id: 0,
        });
        let pocket = EngineConfig::SherpaOnnxPocket(sayit_core::PocketConfig {
            reference_audio: None,
            preset_voice: None,
            executable: "tts".into(),
            lm_flow: assets.join("flow"),
            lm_main: assets.join("main"),
            encoder: assets.join("encoder"),
            decoder: assets.join("decoder"),
            text_conditioner: assets.join("text"),
            vocab_json: assets.join("vocab"),
            token_scores_json: assets.join("scores"),
            provider: "cpu".into(),
            num_threads: 1,
            num_steps: 5,
        });
        fs::write(&piper_path, serde_json::to_vec(&piper).unwrap()).unwrap();
        fs::write(&pocket_path, serde_json::to_vec(&pocket).unwrap()).unwrap();
        let voice_id = Uuid::new_v4();
        {
            let mut inner = state.inner.write().await;
            inner.voices.push(VoiceProfile {
                generated: false,
                language: None,
                generation_tuning: None,
                reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
                id: voice_id,
                name: "Remembered".into(),
                model_id: "pocket-tts-int8".into(),
                reference_audio_path: directory.path().join("voice.wav").display().to_string(),
                reference_text: None,
                reference_duration_seconds: 4.0,
                created_at: chrono::Utc::now(),
                quality: VoiceQuality::default(),
                refinement_steps: 5,
            });
            inner
                .settings
                .voice_profile_by_model
                .insert("pocket-tts-int8".into(), Some(voice_id));
        }
        select_model(State(state.clone()), RoutePath("pocket-tts-int8".into()))
            .await
            .unwrap();
        assert_eq!(
            state.inner.read().await.settings.active_voice_profile_id,
            Some(voice_id)
        );
        select_model(
            State(state.clone()),
            RoutePath("piper-en-us-lessac-medium".into()),
        )
        .await
        .unwrap();
        assert_eq!(
            state.inner.read().await.settings.active_voice_profile_id,
            None
        );
        select_model(State(state), RoutePath("pocket-tts-int8".into()))
            .await
            .unwrap();
        let reopened = ServiceState::open(directory.path().to_owned(), None).unwrap();
        assert_eq!(
            reopened.inner.read().await.settings.active_voice_profile_id,
            Some(voice_id)
        );
    }
}
