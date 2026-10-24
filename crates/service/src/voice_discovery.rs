use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sayit_core::{QwenGenerationOptions, QwenSamplingOptions};
use sayit_protocol::{
    VoiceCandidate, VoiceDiscoveryRequest, VoiceDiscoverySnapshot, VoiceDiscoveryState,
    VoiceGenerationTuning,
};

type ApiError = (StatusCode, String);

#[derive(Clone)]
struct CandidateAudio {
    source: PathBuf,
    prepared: Option<PathBuf>,
    quality: Option<VoiceQuality>,
}

pub(super) struct Studio {
    snapshot: VoiceDiscoverySnapshot,
    cancelled: Arc<AtomicBool>,
    directory: Arc<tempfile::TempDir>,
    audio: HashMap<Uuid, CandidateAudio>,
}

pub(super) fn sampling(tuning: VoiceGenerationTuning) -> QwenSamplingOptions {
    QwenSamplingOptions {
        temperature: tuning.temperature,
        top_p: tuning.top_p,
        top_k: tuning.top_k,
        repetition_penalty: tuning.repetition_penalty,
    }
}

fn randomized_tuning(seed: u32) -> VoiceGenerationTuning {
    let mut rng = StdRng::seed_from_u64(u64::from(seed) ^ 0x5354_525f_564f_4943);
    VoiceGenerationTuning {
        temperature: rng.random_range(4..=24) as f32 / 20.0,
        top_p: rng.random_range(10..=20) as f32 / 20.0,
        top_k: rng.random_range(0..=100),
        repetition_penalty: rng.random_range(18..=30) as f32 / 20.0,
    }
}

pub(super) fn cancel_active(state: &ServiceState) {
    if let Some(studio) = state.voice_studio.lock().unwrap().as_ref()
        && studio.snapshot.state == VoiceDiscoveryState::Generating
    {
        studio.cancelled.store(true, Ordering::Relaxed);
    }
}

pub(super) async fn snapshot(
    State(state): State<ServiceState>,
) -> Json<Option<VoiceDiscoverySnapshot>> {
    Json(
        state
            .voice_studio
            .lock()
            .unwrap()
            .as_ref()
            .map(|studio| studio.snapshot.clone()),
    )
}

pub(super) async fn cancel(State(state): State<ServiceState>) -> StatusCode {
    cancel_active(&state);
    StatusCode::NO_CONTENT
}

pub(super) async fn clear(State(state): State<ServiceState>) -> Result<StatusCode, ApiError> {
    let mut slot = state.voice_studio.lock().unwrap();
    if slot
        .as_ref()
        .is_some_and(|studio| studio.snapshot.state == VoiceDiscoveryState::Generating)
    {
        return Err((
            StatusCode::CONFLICT,
            "Cancel generation before clearing candidates".into(),
        ));
    }
    let preview = state.player.snapshot().job_id;
    if slot.as_ref().is_some_and(|studio| {
        studio
            .snapshot
            .candidates
            .iter()
            .any(|candidate| Some(candidate.id) == preview)
    }) {
        state.player.stop().map_err(unprocessable)?;
        state.emit("playback");
    }
    *slot = None;
    drop(slot);
    state.emit("voices");
    Ok(StatusCode::NO_CONTENT)
}

async fn reading_is_busy(state: &ServiceState) -> bool {
    state.inner.read().await.jobs.iter().any(|job| {
        matches!(
            job.state,
            JobState::Queued | JobState::Synthesizing | JobState::Playing | JobState::Paused
        )
    })
}

pub(super) async fn start(
    State(state): State<ServiceState>,
    Json(request): Json<VoiceDiscoveryRequest>,
) -> Result<(StatusCode, Json<VoiceDiscoverySnapshot>), ApiError> {
    // Hold the queue's read lock until the session is published. A concurrent
    // submission must either be seen here or cancel this session when enqueued.
    let inner = state.inner.read().await;
    if inner.jobs.iter().any(|job| {
        matches!(
            job.state,
            JobState::Queued | JobState::Synthesizing | JobState::Playing | JobState::Paused
        )
    }) {
        return Err((
            StatusCode::CONFLICT,
            "Stop the current reading before generating voice candidates".into(),
        ));
    }
    let (model_id, prefer_gpu, cleaning) = (
        inner
            .settings
            .active_model_id
            .clone()
            .ok_or((StatusCode::CONFLICT, "Select a model first".into()))?,
        inner.settings.compute_device != "cpu",
        inner.settings.text_cleaning,
    );
    let mut config =
        EngineConfig::from_path(&state.models.config_path(&model_id)).map_err(unprocessable)?;
    if !matches!(config, EngineConfig::Qwen3TtsOnnx(_)) {
        return Err((
            StatusCode::CONFLICT,
            "Voice discovery requires the native Qwen Base backend".into(),
        ));
    }
    let language = request
        .language
        .as_deref()
        .or(config.language())
        .unwrap_or("auto");
    let language = sayit_core::language::qwen_language(language)
        .map_err(unprocessable)?
        .code
        .to_owned();
    config.set_language(&language).map_err(unprocessable)?;
    sampling(request.tuning).validate().map_err(unprocessable)?;
    let text = clean_text(
        &request.sample_text,
        TextCleaningOptions {
            enabled: cleaning.enabled,
            strip_markdown: cleaning.strip_markdown,
            strip_html: cleaning.strip_html,
            strip_code_blocks: cleaning.strip_code_blocks,
            strip_special_characters: cleaning.strip_special_characters,
            normalize_whitespace: cleaning.normalize_whitespace,
        },
    )
    .text;
    if !(20..=160).contains(&text.chars().count()) || !(1..=8).contains(&request.candidate_count) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Use 20–160 sample characters and 1–8 candidates".into(),
        ));
    }
    let count = usize::from(request.candidate_count);
    let seed = request
        .seed
        .unwrap_or_else(|| Uuid::new_v4().as_u128() as u32);
    let cancelled = Arc::new(AtomicBool::new(false));
    let (directory, initial) = {
        let mut slot = state.voice_studio.lock().unwrap();
        if slot
            .as_ref()
            .is_some_and(|studio| studio.snapshot.state == VoiceDiscoveryState::Generating)
        {
            return Err((
                StatusCode::CONFLICT,
                "Voice generation is already running; cancel it first".into(),
            ));
        }
        let keep = slot
            .as_ref()
            .is_some_and(|studio| studio.snapshot.model_id == model_id);
        if keep
            && slot
                .as_ref()
                .is_some_and(|studio| studio.snapshot.candidates.len() + count > 32)
        {
            return Err((
                StatusCode::CONFLICT,
                "Clear candidates before starting a new collection".into(),
            ));
        }
        let (directory, candidates, audio) = if keep {
            let previous = slot.take().unwrap();
            (
                previous.directory,
                previous.snapshot.candidates,
                previous.audio,
            )
        } else {
            let directory = tempfile::Builder::new()
                .prefix(".discovery-")
                .tempdir()
                .map_err(|error| internal_error(error.into()))?;
            (Arc::new(directory), Vec::new(), HashMap::new())
        };
        let snapshot = VoiceDiscoverySnapshot {
            id: Uuid::new_v4(),
            model_id,
            state: VoiceDiscoveryState::Generating,
            completed: 0,
            total: count,
            candidates,
            error: None,
        };
        *slot = Some(Studio {
            snapshot: snapshot.clone(),
            cancelled: cancelled.clone(),
            directory: directory.clone(),
            audio,
        });
        (directory, snapshot)
    };
    drop(inner);
    let batch = initial.id;
    let background = state.clone();
    tokio::spawn(async move {
        let worker_state = background.clone();
        let worker_cancel = cancelled.clone();
        let result = tokio::task::spawn_blocking(move || {
            run_batch(
                &worker_state,
                batch,
                directory.path(),
                &config,
                prefer_gpu,
                &text,
                &language,
                seed,
                count,
                request.tuning,
                request.randomize_tuning,
                &worker_cancel,
            )
        })
        .await;
        let error = match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(format!("{error:#}")),
            Err(error) => Some(error.to_string()),
        };
        {
            let mut slot = background.voice_studio.lock().unwrap();
            if let Some(studio) = slot.as_mut()
                && studio.snapshot.id == batch
            {
                studio.snapshot.state = if cancelled.load(Ordering::Relaxed) {
                    VoiceDiscoveryState::Cancelled
                } else if error.is_some() {
                    VoiceDiscoveryState::Failed
                } else {
                    VoiceDiscoveryState::Ready
                };
                studio.snapshot.error = if studio.snapshot.state == VoiceDiscoveryState::Failed {
                    error
                } else {
                    None
                };
            }
        }
        background.emit("voices");
    });
    state.emit("voices");
    Ok((StatusCode::ACCEPTED, Json(initial)))
}

#[allow(clippy::too_many_arguments)]
fn run_batch(
    state: &ServiceState,
    batch: Uuid,
    directory: &Path,
    config: &EngineConfig,
    prefer_gpu: bool,
    text: &str,
    language: &str,
    seed: u32,
    count: usize,
    tuning: VoiceGenerationTuning,
    randomize: bool,
    cancelled: &AtomicBool,
) -> Result<()> {
    anyhow::ensure!(
        !cancelled.load(Ordering::Relaxed),
        "voice discovery cancelled"
    );
    let mut resident = state.resident_engine.lock().unwrap();
    *resident = None;
    let engine = ManagedEngine::load(config, &state.inference_runtime, prefer_gpu)?;
    *state.last_inference_device.lock().unwrap() = Some(inference_device(&engine));
    for index in 0..count {
        anyhow::ensure!(
            !cancelled.load(Ordering::Relaxed),
            "voice discovery cancelled"
        );
        let id = Uuid::new_v4();
        // Failed/cancelled attempts remove their audio even when completed
        // candidates keep the collection alive for another batch.
        let candidate_directory = tempfile::Builder::new()
            .prefix("candidate-")
            .tempdir_in(directory)?;
        let source = candidate_directory.path().join("sample.wav");
        let candidate_seed = seed.wrapping_add(index as u32);
        let tuning = if randomize {
            randomized_tuning(candidate_seed)
        } else {
            tuning
        };
        let options = QwenGenerationOptions {
            seed: Some(u64::from(candidate_seed)),
            max_new_tokens: Some(512),
            sampling: sampling(tuning),
        };
        let mut audible = false;
        engine.synthesize_streaming(
            &SynthesisRequest {
                generation: Some(&options),
                text,
                output: &source,
                reference_audio: None,
                reference_text: None,
                speaking_pace: 1.0,
            },
            Some(cancelled),
            &mut |pcm, _| {
                audible |= pcm.iter().any(|sample| sample.abs() > 0.001);
                Ok(())
            },
        )?;
        anyhow::ensure!(audible, "the model produced a silent candidate");
        let duration = wav_duration_seconds(&source)?;
        let prepared = candidate_directory.path().join("reference.wav");
        let (quality, save_error) = match analyze_and_trim_voice(&source, &prepared, 45.0) {
            Ok(quality) => (Some(quality), None),
            Err(error) => (None, Some(error.to_string())),
        };
        let candidate = VoiceCandidate {
            id,
            sample_text: text.into(),
            language: language.into(),
            seed: candidate_seed,
            tuning,
            duration_seconds: duration,
            save_error,
            saved_voice_profile_id: None,
        };
        {
            let mut slot = state.voice_studio.lock().unwrap();
            let studio = slot.as_mut().context("voice session ended")?;
            anyhow::ensure!(
                studio.snapshot.id == batch && !cancelled.load(Ordering::Relaxed),
                "voice discovery cancelled"
            );
            studio.audio.insert(
                id,
                CandidateAudio {
                    source,
                    prepared: quality.as_ref().map(|_| prepared),
                    quality,
                },
            );
            studio.snapshot.candidates.push(candidate);
            studio.snapshot.completed += 1;
            let _ = candidate_directory.keep();
        }
        state.emit("voices");
    }
    Ok(())
}

async fn candidate(
    state: &ServiceState,
    id: Uuid,
) -> Result<
    (
        VoiceCandidate,
        CandidateAudio,
        Arc<tempfile::TempDir>,
        String,
    ),
    ApiError,
> {
    let active = state.inner.read().await.settings.active_model_id.clone();
    let slot = state.voice_studio.lock().unwrap();
    let studio = slot.as_ref().ok_or((
        StatusCode::NOT_FOUND,
        "Voice discovery session not found".into(),
    ))?;
    if active.as_deref() != Some(studio.snapshot.model_id.as_str()) {
        return Err((
            StatusCode::CONFLICT,
            "Select the candidate's model first".into(),
        ));
    }
    let candidate = studio
        .snapshot
        .candidates
        .iter()
        .find(|candidate| candidate.id == id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, "Voice candidate not found".into()))?;
    let audio = studio.audio.get(&id).cloned().ok_or((
        StatusCode::NOT_FOUND,
        "Voice candidate audio not found".into(),
    ))?;
    Ok((
        candidate,
        audio,
        studio.directory.clone(),
        studio.snapshot.model_id.clone(),
    ))
}

pub(super) async fn preview(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
) -> Result<StatusCode, ApiError> {
    if reading_is_busy(&state).await {
        return Err((
            StatusCode::CONFLICT,
            "Stop the current reading before previewing a candidate".into(),
        ));
    }
    let (candidate, audio, _directory, _) = candidate(&state, id).await?;
    let settings = state.inner.read().await.settings.clone();
    state
        .player
        .play(id, audio.source, settings.playback_rate, settings.volume)
        .map_err(unprocessable)?;
    state
        .player
        .set_preview_text(id, candidate.sample_text)
        .map_err(unprocessable)?;
    state.emit("state");
    Ok(StatusCode::ACCEPTED)
}

pub(super) async fn save(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
    Json(request): Json<VoiceRenameRequest>,
) -> Result<(StatusCode, Json<VoiceProfile>), ApiError> {
    let _save_guard = state.voice_studio_save.lock().await;
    let name = request.name.trim();
    if name.is_empty() || name.chars().count() > 80 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Voice name must contain 1 to 80 characters".into(),
        ));
    }
    let (candidate, audio, _directory, model_id) = candidate(&state, id).await?;
    if let Some(saved) = candidate.saved_voice_profile_id
        && let Some(profile) = state
            .inner
            .read()
            .await
            .voices
            .iter()
            .find(|voice| voice.id == saved)
            .cloned()
    {
        return Ok((StatusCode::OK, Json(profile)));
    }
    let source = audio.prepared.as_ref().ok_or((
        StatusCode::UNPROCESSABLE_ENTITY,
        candidate
            .save_error
            .clone()
            .unwrap_or_else(|| "Use a longer sample before saving this voice".into()),
    ))?;
    let voice_id = Uuid::new_v4();
    let destination = state.voices_dir.join(format!("{voice_id}.wav"));
    fs::copy(source, &destination).map_err(|error| internal_error(error.into()))?;
    let quality = audio
        .quality
        .context("candidate has no voice analysis")
        .map_err(internal_error)?;
    let profile = VoiceProfile {
        generated: true,
        language: Some(candidate.language),
        generation_tuning: Some(candidate.tuning),
        id: voice_id,
        name: name.into(),
        model_id,
        reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
        reference_audio_path: destination.display().to_string(),
        reference_text: Some(candidate.sample_text),
        reference_duration_seconds: quality.trimmed_duration_seconds,
        created_at: chrono::Utc::now(),
        quality,
        refinement_steps: 5,
    };
    state.inner.write().await.voices.push(profile.clone());
    state.persist().await.map_err(internal_error)?;
    {
        let mut slot = state.voice_studio.lock().unwrap();
        if let Some(studio) = slot.as_mut()
            && let Some(candidate) = studio
                .snapshot
                .candidates
                .iter_mut()
                .find(|candidate| candidate.id == id)
        {
            candidate.saved_voice_profile_id = Some(voice_id);
        }
    }
    state.emit("voices");
    Ok((StatusCode::CREATED, Json(profile)))
}

pub(super) async fn tune(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<Uuid>,
    Json(tuning): Json<VoiceGenerationTuning>,
) -> Result<StatusCode, ApiError> {
    sampling(tuning).validate().map_err(unprocessable)?;
    let model_id = state
        .inner
        .read()
        .await
        .voices
        .iter()
        .find(|voice| voice.id == id)
        .map(|voice| voice.model_id.clone())
        .ok_or((StatusCode::NOT_FOUND, "Voice profile not found".into()))?;
    let config =
        EngineConfig::from_path(&state.models.config_path(&model_id)).map_err(unprocessable)?;
    if !config.is_qwen() {
        return Err((
            StatusCode::CONFLICT,
            "Generation tuning is available only for Qwen voices".into(),
        ));
    }
    state
        .inner
        .write()
        .await
        .voices
        .iter_mut()
        .find(|voice| voice.id == id)
        .ok_or((StatusCode::NOT_FOUND, "Voice profile not found".into()))?
        .generation_tuning = Some(tuning);
    state.persist().await.map_err(internal_error)?;
    state.emit("voices");
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> VoiceDiscoveryRequest {
        VoiceDiscoveryRequest {
            sample_text: "A clear voice reads these familiar words in a quiet room.".into(),
            language: Some("en".into()),
            candidate_count: 1,
            seed: Some(417),
            randomize_tuning: false,
            tuning: VoiceGenerationTuning::default(),
        }
    }

    #[test]
    fn random_recipes_are_bounded_repeatable_and_vary_between_candidates() {
        let mut recipes = std::collections::HashSet::new();
        for seed in 0..256 {
            let tuning = randomized_tuning(seed);
            sampling(tuning).validate().unwrap();
            let encoded = serde_json::to_string(&tuning).unwrap();
            assert_eq!(
                encoded,
                serde_json::to_string(&randomized_tuning(seed)).unwrap()
            );
            recipes.insert(encoded);
        }
        assert!(recipes.len() > 240);
    }

    fn collection(state: &ServiceState, count: usize, status: VoiceDiscoveryState) -> Vec<Uuid> {
        let directory = Arc::new(tempfile::tempdir().unwrap());
        let source = directory.path().join("reference.wav");
        let mut writer = hound::WavWriter::create(
            &source,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..96_000 {
            writer.write_sample(1000i16).unwrap();
        }
        writer.finalize().unwrap();
        let candidates = (0..count)
            .map(|index| VoiceCandidate {
                id: Uuid::new_v4(),
                sample_text: request().sample_text,
                language: "es".into(),
                seed: index as u32,
                tuning: VoiceGenerationTuning::default(),
                duration_seconds: 4.0,
                save_error: None,
                saved_voice_profile_id: None,
            })
            .collect::<Vec<_>>();
        let ids = candidates.iter().map(|candidate| candidate.id).collect();
        let audio = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.id,
                    CandidateAudio {
                        source: source.clone(),
                        prepared: Some(source.clone()),
                        quality: Some(VoiceQuality {
                            trimmed_duration_seconds: 4.0,
                            ..VoiceQuality::default()
                        }),
                    },
                )
            })
            .collect();
        *state.voice_studio.lock().unwrap() = Some(Studio {
            snapshot: VoiceDiscoverySnapshot {
                id: Uuid::new_v4(),
                model_id: "qwen3-tts-06b-base".into(),
                state: status,
                completed: count,
                total: count,
                candidates,
                error: None,
            },
            cancelled: Arc::new(AtomicBool::new(false)),
            directory,
            audio,
        });
        ids
    }

    #[tokio::test]
    async fn invalid_discovery_requests_leave_no_session_or_history() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::native_voice_test_state(directory.path());
        let mut invalid = vec![
            VoiceDiscoveryRequest {
                sample_text: "too short".into(),
                ..request()
            },
            VoiceDiscoveryRequest {
                sample_text: "字".repeat(161),
                ..request()
            },
            VoiceDiscoveryRequest {
                candidate_count: 0,
                ..request()
            },
            VoiceDiscoveryRequest {
                candidate_count: 9,
                ..request()
            },
            VoiceDiscoveryRequest {
                language: Some("unknown".into()),
                ..request()
            },
        ];
        for tuning in [
            VoiceGenerationTuning {
                temperature: f32::NAN,
                ..Default::default()
            },
            VoiceGenerationTuning {
                temperature: 1.21,
                ..Default::default()
            },
            VoiceGenerationTuning {
                top_p: 0.49,
                ..Default::default()
            },
            VoiceGenerationTuning {
                top_k: 101,
                ..Default::default()
            },
            VoiceGenerationTuning {
                repetition_penalty: 0.89,
                ..Default::default()
            },
        ] {
            invalid.push(VoiceDiscoveryRequest {
                tuning,
                ..request()
            });
        }
        for request in invalid {
            let error = start(State(state.clone()), Json(request))
                .await
                .unwrap_err();
            assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
            assert!(state.voice_studio.lock().unwrap().is_none());
        }
        assert!(state.inner.read().await.jobs.is_empty());
        assert!(state.inner.read().await.history.is_empty());
        assert_eq!(
            start(State(ServiceState::default()), Json(request()))
                .await
                .unwrap_err()
                .0,
            StatusCode::CONFLICT
        );
    }

    #[tokio::test]
    async fn full_collection_and_cancellation_preserve_completed_candidates() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::native_voice_test_state(directory.path());
        let ids = collection(&state, 32, VoiceDiscoveryState::Ready);
        let root = state
            .voice_studio
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .directory
            .path()
            .to_owned();
        assert_eq!(
            start(State(state.clone()), Json(request()))
                .await
                .unwrap_err()
                .0,
            StatusCode::CONFLICT
        );
        let Json(snapshot) = snapshot(State(state.clone())).await;
        assert_eq!(
            snapshot
                .unwrap()
                .candidates
                .iter()
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>(),
            ids
        );
        state
            .voice_studio
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .snapshot
            .state = VoiceDiscoveryState::Generating;
        assert_eq!(
            clear(State(state.clone())).await.unwrap_err().0,
            StatusCode::CONFLICT
        );
        assert_eq!(
            start(State(state.clone()), Json(request()))
                .await
                .unwrap_err()
                .0,
            StatusCode::CONFLICT
        );
        cancel(State(state.clone())).await;
        assert!(
            state
                .voice_studio
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .cancelled
                .load(Ordering::Relaxed)
        );
        assert!(root.join("reference.wav").is_file());
        state
            .voice_studio
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .snapshot
            .state = VoiceDiscoveryState::Cancelled;
        clear(State(state.clone())).await.unwrap();
        assert!(state.voice_studio.lock().unwrap().is_none());
        assert!(!root.exists());
    }

    #[tokio::test]
    async fn concurrent_saves_are_idempotent_and_profile_settings_survive_restart() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::native_voice_test_state(directory.path());
        let id = collection(&state, 1, VoiceDiscoveryState::Ready)[0];
        let request = VoiceRenameRequest {
            name: "Generated narrator".into(),
        };
        let (first, second) = tokio::join!(
            save(State(state.clone()), RoutePath(id), Json(request.clone())),
            save(State(state.clone()), RoutePath(id), Json(request)),
        );
        let (_, Json(profile)) = first.unwrap();
        assert_eq!(second.unwrap().1.0.id, profile.id);
        assert_eq!(state.inner.read().await.voices.len(), 1);
        assert_eq!(fs::read_dir(&state.voices_dir).unwrap().count(), 1);
        assert!(profile.generated);
        let faithful = VoiceGenerationTuning {
            temperature: 0.45,
            top_p: 0.75,
            top_k: 20,
            repetition_penalty: 1.3,
        };
        tune(State(state.clone()), RoutePath(profile.id), Json(faithful))
            .await
            .unwrap();
        assert_eq!(
            tune(
                State(state.clone()),
                RoutePath(profile.id),
                Json(VoiceGenerationTuning {
                    top_k: 101,
                    ..faithful
                })
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        clear(State(state.clone())).await.unwrap();
        assert!(Path::new(&profile.reference_audio_path).is_file());
        let restored = ServiceState::open(directory.path().join("data"), None).unwrap();
        let saved = restored.inner.read().await.voices[0].clone();
        assert!(saved.generated);
        assert_eq!(saved.language.as_deref(), Some("es"));
        assert_eq!(saved.generation_tuning.unwrap().top_k, 20);
        super::super::select_voice(State(restored.clone()), RoutePath(saved.id))
            .await
            .unwrap();
        let config =
            EngineConfig::from_path(&restored.models.config_path(&saved.model_id)).unwrap();
        assert_eq!(config.language(), Some("Spanish"));
        let mut legacy = serde_json::to_value(&*restored.inner.read().await).unwrap();
        for key in ["generated", "language", "generation_tuning"] {
            legacy["voices"][0].as_object_mut().unwrap().remove(key);
        }
        fs::write(&state.store_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        let legacy = ServiceState::open(directory.path().join("data"), None).unwrap();
        let old = legacy.inner.read().await.voices[0].clone();
        assert!(!old.generated);
        assert!(old.language.is_none());
        assert!(old.generation_tuning.is_none());
    }
}
