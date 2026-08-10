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
use say_the_rest_core::{
    BenchmarkRunner, EngineConfig, ResidentSherpaEngine, SherpaOnnxEngine, SynthesisRequest,
    TextCleaningOptions, clean_text, wav_duration_seconds,
};
use say_the_rest_protocol::{
    Health, HistoryItem, HuggingFaceModelImportRequest, JobState, LocalModelImportRequest,
    ModelBenchmark, ModelDescriptor, PROTOCOL_VERSION, PinRequest, PlaybackSnapshot,
    PlaybackTextChunk, QueuePolicy, ServiceEvent, ServiceSettings, ServiceSettingsUpdate,
    ServiceSnapshot, SpeechJob, SpeechSource, SpeechSubmission, VoiceCloneRequest, VoiceProfile,
    VoiceQuality, VoiceRenameRequest,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
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
mod player;
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
}

struct ResidentEngineSlot {
    config_path: PathBuf,
    config_bytes: Vec<u8>,
    engine: ResidentSherpaEngine,
    last_used: Instant,
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
        let root = std::env::temp_dir().join(format!("say-the-rest-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("temporary test directory");
        let model = root.join("test-model.onnx");
        let tokens = root.join("test-tokens.txt");
        fs::write(&model, b"test").expect("temporary test model");
        fs::write(&tokens, b"test").expect("temporary test tokens");
        let config_path = root.join("test-config.json");
        let config = EngineConfig::SherpaOnnxVits(say_the_rest_core::VitsConfig {
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
                EngineConfig::SherpaOnnxPocket(_) => "pocket-tts-int8",
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
        self.notify.notify_one();
    }

    async fn model_unload_loop(self) {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            let timeout =
                Duration::from_secs(self.inner.read().await.settings.model_unload_delay_seconds);
            let mut resident = self.resident_engine.lock().unwrap();
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
        let result = job.clone();
        drop(inner);
        let _ = self.persist().await;
        Some(result)
    }

    async fn process_job(&self, job: SpeechJob) {
        let Some(config_path) = self.resolve_engine_config().await else {
            self.finish_job(
                job.id,
                Err(anyhow::anyhow!("no inference configuration installed")),
            )
            .await;
            return;
        };
        let output = self.audio_dir.join(format!("{}.wav", job.id));
        let (reference_audio, speaking_pace) = {
            let inner = self.inner.read().await;
            (
                job.voice_profile_id
                    .and_then(|id| inner.voices.iter().find(|voice| voice.id == id))
                    .map(|voice| PathBuf::from(&voice.reference_audio_path)),
                job.speaking_pace.unwrap_or(inner.settings.speaking_pace),
            )
        };
        let text = job.text.clone();
        let resident_engine = self.resident_engine.clone();
        let synthesis = tokio::task::spawn_blocking(move || -> Result<(PathBuf, f64)> {
            let config = EngineConfig::from_path(&config_path)?;
            let config_bytes = fs::read(&config_path)?;
            let mut resident = resident_engine.lock().unwrap();
            let reload = resident.as_ref().is_none_or(|slot| {
                slot.config_path != config_path || slot.config_bytes != config_bytes
            });
            if reload {
                *resident = Some(ResidentEngineSlot {
                    config_path: config_path.clone(),
                    config_bytes,
                    engine: ResidentSherpaEngine::load(&config)?,
                    last_used: Instant::now(),
                });
            }
            let slot = resident
                .as_mut()
                .context("resident engine was not loaded")?;
            slot.engine.synthesize(&SynthesisRequest {
                text: &text,
                output: &output,
                reference_audio: reference_audio.as_deref(),
                speaking_pace,
            })?;
            slot.last_used = Instant::now();
            let duration = wav_duration_seconds(&output)?;
            Ok((output, duration))
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("synthesis worker crashed: {error}")));
        match synthesis {
            Ok((path, duration)) => self.play_job(job.id, path, duration).await,
            Err(error) => self.finish_job(job.id, Err(error)).await,
        }
    }

    async fn play_job(&self, id: Uuid, path: PathBuf, duration: f64) {
        let (rate, volume) = {
            let mut inner = self.inner.write().await;
            let Some(job) = inner.jobs.iter_mut().find(|job| job.id == id) else {
                return;
            };
            job.state = JobState::Playing;
            (inner.settings.playback_rate, inner.settings.volume)
        };
        if let Err(error) = self.player.play(id, path.clone(), rate, volume) {
            self.finish_job(id, Err(error)).await;
            return;
        }
        loop {
            let snapshot = self.player.snapshot();
            if matches!(snapshot.status, PlayerStatus::Finished | PlayerStatus::Idle) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let _ = self.player.stop();
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
        let (audio_path, duration, error) = match result {
            Ok((path, duration)) => (Some(path.display().to_string()), Some(duration), None),
            Err(error) => (None, None, Some(error.to_string())),
        };
        inner.jobs[index].state = if error.is_some() {
            JobState::Failed
        } else {
            JobState::Completed
        };
        inner.jobs[index].error = error;
        let completed = inner.jobs[index].clone();
        inner.recovery_attempts.remove(&id);
        inner.history.insert(
            0,
            HistoryItem {
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
        fs::File::open(&backup_temporary)?.sync_all()?;
        replace_file(&backup_temporary, &backup)?;
    }
    replace_file(&temporary, store_path)?;
    sync_parent_directory(store_path)?;
    Ok(())
}

fn replace_file(source: &Path, destination: &Path) -> Result<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        #[cfg(windows)]
        Err(_) if destination.is_file() => {
            fs::remove_file(destination)?;
            fs::rename(source, destination)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
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
        .route("/v1/models/{id}/update", post(install_model))
        .route("/v1/models/{id}/cancel", post(cancel_model_install))
        .route("/v1/models/{id}/select", post(select_model))
        .route("/v1/models/{id}/unload", post(unload_model))
        .route(
            "/v1/models/{id}/voices/{voice_id}/select",
            post(select_preset_voice),
        )
        .route("/v1/models/{id}/benchmark", post(benchmark_model))
        .route("/v1/models/{id}", axum::routing::delete(remove_model))
        .route("/v1/voices", get(voices).post(create_voice))
        .route("/v1/voices/default/select", post(select_default_voice))
        .route("/v1/voices/{id}/select", post(select_voice))
        .route("/v1/voices/{id}/preview", post(preview_voice))
        .route("/v1/voices/{id}/rename", post(rename_voice))
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
        PlayerStatus::Paused => (Some(JobState::Paused), player.job_id),
        PlayerStatus::Idle | PlayerStatus::Finished => (
            processing.map(|_| JobState::Synthesizing),
            processing.map(|job| job.id),
        ),
    };
    let spoken_text = active_job_id
        .and_then(|id| inner.jobs.iter().find(|job| job.id == id))
        .map(|job| job.text.clone())
        .unwrap_or_default();
    let current_title = speech_title(&spoken_text);
    let spoken_chunks = playback_text_chunks(&spoken_text, player.duration_seconds);
    Json(ServiceSnapshot {
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
    submit_job(
        State(state),
        Json(SpeechSubmission {
            text: item.job.text,
            voice_profile_id: item.job.voice_profile_id,
            source: SpeechSource::History,
            queue_policy: QueuePolicy::Replace,
            confirmed_long_text: true,
            speaking_pace: item.job.speaking_pace,
        }),
    )
    .await
}

async fn replay_history(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    let (path, rate, volume) = {
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
        (path, inner.settings.playback_rate, inner.settings.volume)
    };
    let previously_active = state.player.snapshot().job_id;
    state.player.stop().map_err(internal_error)?;
    state
        .player
        .play(id, path, rate, volume)
        .map_err(internal_error)?;
    {
        let mut inner = state.inner.write().await;
        mark_replay_started(&mut inner, id, previously_active);
    }
    state.persist().await.map_err(internal_error)?;
    let replay_state = state.clone();
    tokio::spawn(async move {
        loop {
            if matches!(
                replay_state.player.snapshot().status,
                PlayerStatus::Finished | PlayerStatus::Idle
            ) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let _ = replay_state.player.stop();
        {
            let mut inner = replay_state.inner.write().await;
            mark_replay_finished(&mut inner, id);
        }
        let _ = replay_state.persist().await;
        replay_state.emit("jobs");
        replay_state.emit("playback");
    });
    state.emit("jobs");
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

fn mark_replay_started(inner: &mut InnerState, replay_id: Uuid, active_id: Option<Uuid>) {
    if let Some(active) = active_id.filter(|active| *active != replay_id)
        && let Some(job) = inner.jobs.iter_mut().find(|job| job.id == active)
        && matches!(job.state, JobState::Playing | JobState::Paused)
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
    inner.settings.voice_profile_by_model.insert(model_id, None);
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
    inner.settings.active_voice_profile_id = Some(id);
    inner
        .settings
        .voice_profile_by_model
        .insert(model_id, Some(id));
    drop(inner);
    state.persist().await.map_err(internal_error)?;
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
    state.player.pause().map_err(internal_error)?;
    set_active_job_state(&state, JobState::Paused).await;
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn resume(State(state): State<ServiceState>) -> Result<StatusCode, (StatusCode, String)> {
    state.player.resume().map_err(internal_error)?;
    set_active_job_state(&state, JobState::Playing).await;
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn stop(State(state): State<ServiceState>) -> Result<StatusCode, (StatusCode, String)> {
    let active = state.player.snapshot().job_id;
    state.player.stop().map_err(internal_error)?;
    if let Some(job) = state
        .inner
        .write()
        .await
        .jobs
        .iter_mut()
        .find(|job| Some(job.id) == active)
    {
        job.state = JobState::Cancelled;
    }
    state.persist().await.map_err(internal_error)?;
    state.emit("jobs");
    state.emit("playback");
    Ok(StatusCode::NO_CONTENT)
}

async fn seek(
    State(state): State<ServiceState>,
    Json(request): Json<say_the_rest_protocol::SecondsRequest>,
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
    Json(request): Json<say_the_rest_protocol::PlaybackRateRequest>,
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
    Json(request): Json<say_the_rest_protocol::VolumeRequest>,
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
    if !matches!(config, EngineConfig::SherpaOnnxPocket(_)) {
        return Err((
            StatusCode::CONFLICT,
            "the selected model does not support voice cloning".into(),
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
    let quality = analyze_and_trim_voice(&source, &destination)
        .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))?;
    let profile = VoiceProfile {
        id,
        name: name.to_owned(),
        model_id,
        reference_audio_path: destination.display().to_string(),
        reference_duration_seconds: quality.trimmed_duration_seconds,
        created_at: chrono::Utc::now(),
        quality,
    };
    state.inner.write().await.voices.push(profile.clone());
    state.persist().await.map_err(internal_error)?;
    Ok((StatusCode::CREATED, Json(profile)))
}

async fn models(State(state): State<ServiceState>) -> Json<Vec<ModelDescriptor>> {
    let inner = state.inner.read().await;
    let mut models = state
        .models
        .descriptors(inner.settings.active_model_id.as_deref(), &inner.benchmarks);
    drop(inner);
    let resident_path = state
        .resident_engine
        .lock()
        .unwrap()
        .as_ref()
        .map(|slot| slot.config_path.clone());
    for model in &mut models {
        model.resident = resident_path.as_ref() == Some(&state.models.config_path(&model.id));
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
    let reference = {
        let inner = state.inner.read().await;
        inner
            .settings
            .active_voice_profile_id
            .and_then(|voice_id| inner.voices.iter().find(|voice| voice.id == voice_id))
            .map(|voice| PathBuf::from(&voice.reference_audio_path))
    };
    if matches!(&config, EngineConfig::SherpaOnnxPocket(_)) && reference.is_none() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "select a cloned voice before benchmarking PocketTTS".into(),
        ));
    }
    let output = state.audio_dir.join(format!("benchmark-{id}.wav"));
    let report = tokio::task::spawn_blocking(move || {
        let engine = SherpaOnnxEngine::from_config(config);
        let result = BenchmarkRunner::run_with_reference(
            &engine,
            "The quick brown fox jumps over the lazy dog.",
            &output,
            3,
            reference.as_deref(),
        );
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
    *state.engine_config_path.write().await = Some(config);
    *state.resident_engine.lock().unwrap() = None;
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
    let mut resident = state.resident_engine.lock().unwrap();
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

async fn remove_model(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    if state.inner.read().await.settings.active_model_id.as_deref() == Some(id.as_str()) {
        return Err((
            StatusCode::CONFLICT,
            "select another model before removing this one".into(),
        ));
    }
    state.models.remove(&id).map_err(unprocessable)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn submit_job(
    State(state): State<ServiceState>,
    Json(submission): Json<SpeechSubmission>,
) -> Result<(StatusCode, Json<SpeechJob>), (StatusCode, String)> {
    if state.resolve_engine_config().await.is_none() {
        return Err((
            StatusCode::CONFLICT,
            "download and select a speech model before reading text aloud".into(),
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
    let active = if submission.queue_policy == QueuePolicy::Interrupt {
        state.player.snapshot().job_id
    } else {
        None
    };
    if active.is_some() {
        state.player.stop().map_err(internal_error)?;
    }
    let mut inner = state.inner.write().await;
    if let Some(active) = active
        && let Some(job) = inner.jobs.iter_mut().find(|job| job.id == active)
    {
        job.state = JobState::Cancelled;
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
    let voice_profile_id = submission
        .voice_profile_id
        .or(inner.settings.active_voice_profile_id);
    if let Some(voice_id) = voice_profile_id {
        let active_model = inner.settings.active_model_id.as_deref();
        let compatible = inner
            .voices
            .iter()
            .any(|voice| voice.id == voice_id && Some(voice.model_id.as_str()) == active_model);
        if !compatible {
            return Err((
                StatusCode::CONFLICT,
                "voice profile is not compatible with the active model".into(),
            ));
        }
    }
    let job = SpeechJob {
        id: Uuid::new_v4(),
        text,
        source: submission.source,
        state: JobState::Queued,
        created_at: chrono::Utc::now(),
        error: None,
        voice_profile_id,
        speaking_pace: submission.speaking_pace,
    };
    inner.jobs.push_back(job.clone());
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.notify.notify_one();
    state.emit("jobs");
    Ok((StatusCode::ACCEPTED, Json(job)))
}

fn analyze_and_trim_voice(
    source: &std::path::Path,
    destination: &std::path::Path,
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
        (3.0..=15.0).contains(&trimmed_duration),
        "usable speech after silence trimming must be 3 to 15 seconds; received {trimmed_duration:.2} seconds"
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
    }

    #[tokio::test]
    async fn state_is_persisted_across_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let state = ServiceState::open(directory.path().to_owned(), None).unwrap();
        let submission = SpeechSubmission {
            text: "persist me".into(),
            voice_profile_id: None,
            source: Default::default(),
            queue_policy: QueuePolicy::Append,
            confirmed_long_text: false,
            speaking_pace: None,
        };
        let job = SpeechJob {
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
            speaker_permission_confirmed: false,
        };
        let error = create_voice(State(state), Json(request)).await.unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(error.1.contains("permission"));
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
        let quality = analyze_and_trim_voice(&source, &destination).unwrap();
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
            history_quota_bytes: Some(64 * 1_048_576),
            history_retention_days: Some(90),
            long_text_confirmation_characters: Some(4_000),
            speaking_pace: Some(1.25),
            text_cleaning: Some(say_the_rest_protocol::TextCleaningSettings {
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
        assert!(!settings.text_cleaning.strip_code_blocks);
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
            job: SpeechJob {
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
            job: SpeechJob {
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
        let config = EngineConfig::SherpaOnnxVits(say_the_rest_core::VitsConfig {
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
        let piper = EngineConfig::SherpaOnnxVits(say_the_rest_core::VitsConfig {
            executable: "tts".into(),
            model: assets.join("model"),
            tokens: assets.join("tokens"),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 1,
            speaker_id: 0,
        });
        let pocket = EngineConfig::SherpaOnnxPocket(say_the_rest_core::PocketConfig {
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
                id: voice_id,
                name: "Remembered".into(),
                model_id: "pocket-tts-int8".into(),
                reference_audio_path: directory.path().join("voice.wav").display().to_string(),
                reference_duration_seconds: 4.0,
                created_at: chrono::Utc::now(),
                quality: VoiceQuality::default(),
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
