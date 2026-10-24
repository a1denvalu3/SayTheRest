use super::*;
use sayit_core::QwenGenerationOptions;
use sayit_protocol::{ModelVoiceModeRequest, VoiceGenerationTuning, VoiceSynthesisMode};

pub(super) fn supported(config: &EngineConfig) -> bool {
    matches!(config, EngineConfig::Qwen3TtsOnnx(_))
}

pub(super) fn generation_defaults(mode: VoiceSynthesisMode) -> Option<QwenGenerationOptions> {
    (mode != VoiceSynthesisMode::Standard).then(|| QwenGenerationOptions {
        seed: None,
        max_new_tokens: Some(512),
        sampling: voice_discovery::sampling(VoiceGenerationTuning::default()),
    })
}

fn anchor_text(language: &str) -> &'static str {
    match language {
        "es" => {
            "La luz de la mañana entra por la ventana. Cada voz da a estas palabras su propio ritmo y una expresión diferente."
        }
        "fr" => {
            "La lumière du matin éclaire la pièce. Chaque voix donne à ces mots un rythme personnel et une couleur différente."
        }
        "de" => {
            "Das Morgenlicht fällt durch das Fenster. Jede Stimme gibt diesen vertrauten Worten einen eigenen Klang und Rhythmus."
        }
        "it" => {
            "La luce del mattino entra dalla finestra. Ogni voce dà a queste parole un ritmo personale e un colore diverso."
        }
        "pt" => {
            "A luz da manhã entra pela janela. Cada voz dá a estas palavras o seu próprio ritmo e uma expressão diferente."
        }
        "ru" => {
            "Утренний свет наполняет комнату. Каждый голос придаёт этим словам свой ритм и особое настроение."
        }
        "zh" => {
            "清晨的阳光照进房间。每一种声音都能让这些熟悉的话语拥有不同的节奏，色彩和独特的感情。"
        }
        "ja" => {
            "朝の光が窓から部屋に入ります。それぞれの声が、聞き慣れた言葉に新しい響きとリズムを与えます。"
        }
        "ko" => {
            "아침 햇살이 창문으로 들어옵니다. 각각의 목소리는 익숙한 말에 새로운 느낌과 자신만의 리듬을 더합니다."
        }
        _ => {
            "A quiet morning brings a clear new sentence. Every voice gives familiar words its own color, character, and rhythm."
        }
    }
}

struct Reference {
    _directory: tempfile::TempDir,
    audio: PathBuf,
    transcript: &'static str,
    paragraph: usize,
}

/// One validated synthetic reference per reading, or per logical paragraph.
/// Long paragraphs reuse it across all bounded synthesis blocks. Nothing is
/// added to saved profiles or history, and cancellation drops temporary audio.
pub(super) struct ReadingVoice {
    mode: VoiceSynthesisMode,
    language: String,
    reference: Option<Reference>,
}

impl ReadingVoice {
    pub(super) fn new(mode: VoiceSynthesisMode, config: &EngineConfig) -> Result<Self> {
        anyhow::ensure!(
            mode == VoiceSynthesisMode::Standard || supported(config),
            "random reading modes require the native Qwen Base backend"
        );
        let language = if mode == VoiceSynthesisMode::Standard {
            "auto".into()
        } else {
            sayit_core::language::qwen_language(config.language().unwrap_or("auto"))?
                .code
                .to_owned()
        };
        Ok(Self {
            mode,
            language,
            reference: None,
        })
    }

    pub(super) fn reference(&self) -> Option<(&Path, &str)> {
        self.reference
            .as_ref()
            .map(|reference| (reference.audio.as_path(), reference.transcript))
    }

    pub(super) fn prepare(
        &mut self,
        engine: &ManagedEngine,
        paragraph: usize,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        anyhow::ensure!(
            !cancelled.load(Ordering::Relaxed),
            "speech generation cancelled"
        );
        if self.mode == VoiceSynthesisMode::Standard
            || self.reference.as_ref().is_some_and(|reference| {
                self.mode == VoiceSynthesisMode::AutomaticStable || reference.paragraph == paragraph
            })
        {
            return Ok(());
        }
        let text = anchor_text(&self.language);
        let options =
            generation_defaults(self.mode).context("missing random generation controls")?;
        let mut failure = String::new();
        for _ in 0..3 {
            anyhow::ensure!(
                !cancelled.load(Ordering::Relaxed),
                "speech generation cancelled"
            );
            let directory = tempfile::Builder::new()
                .prefix(".sayit-reading-voice-")
                .tempdir()?;
            let source = directory.path().join("sample.wav");
            let prepared = directory.path().join("reference.wav");
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
                    audible |= pcm
                        .iter()
                        .any(|sample| sample.is_finite() && sample.abs() > 0.001);
                    Ok(())
                },
            )?;
            if !audible {
                failure = "the model produced a silent voice sample".into();
                continue;
            }
            match analyze_and_trim_voice(&source, &prepared, 45.0) {
                Ok(_) => {
                    anyhow::ensure!(
                        !cancelled.load(Ordering::Relaxed),
                        "speech generation cancelled"
                    );
                    self.reference = Some(Reference {
                        _directory: directory,
                        audio: prepared,
                        transcript: text,
                        paragraph,
                    });
                    return Ok(());
                }
                Err(error) => failure = error.to_string(),
            }
        }
        anyhow::bail!(
            "Could not prepare a reusable random voice after three attempts: {failure}. Try a saved voice profile."
        )
    }
}

pub(super) async fn select(
    State(state): State<ServiceState>,
    RoutePath(id): RoutePath<String>,
    Json(request): Json<ModelVoiceModeRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let config = EngineConfig::from_path(&state.models.config_path(&id)).map_err(unprocessable)?;
    if !supported(&config) || request.mode == VoiceSynthesisMode::Standard {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Choose a supported native Qwen random mode, or select a saved voice profile".into(),
        ));
    }
    let mut inner = state.inner.write().await;
    inner
        .settings
        .voice_mode_by_model
        .insert(id.clone(), request.mode);
    inner
        .settings
        .voice_profile_by_model
        .insert(id.clone(), None);
    if inner.settings.active_model_id.as_deref() == Some(id.as_str()) {
        inner.settings.active_voice_profile_id = None;
    }
    voice_discovery::cancel_active(&state);
    drop(inner);
    state.persist().await.map_err(internal_error)?;
    state.emit("settings");
    state.emit("models");
    state.emit("voices");
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn modes_are_captured_and_saved_profile_selection_restores_standard_reading() {
        let directory = tempfile::tempdir().unwrap();
        let state = crate::tests::native_voice_test_state(directory.path());
        let model = "qwen3-tts-06b-base";
        let request: SpeechSubmission=serde_json::from_value(serde_json::json!({"text":"A reading without a recorded reference.","queue_policy":"append"})).unwrap();
        let (_, Json(first)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        assert_eq!(first.voice_mode, VoiceSynthesisMode::AutomaticStable);
        assert!(first.voice_profile_id.is_none());
        select(
            State(state.clone()),
            RoutePath(model.into()),
            Json(ModelVoiceModeRequest {
                mode: VoiceSynthesisMode::RandomPerParagraph,
            }),
        )
        .await
        .unwrap();
        let (_, Json(second)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        assert_eq!(second.voice_mode, VoiceSynthesisMode::RandomPerParagraph);
        assert_eq!(
            state.inner.read().await.jobs[0].voice_mode,
            VoiceSynthesisMode::AutomaticStable
        );
        let id = Uuid::new_v4();
        state.inner.write().await.voices.push(VoiceProfile {
            generated: true,
            language: Some("en".into()),
            generation_tuning: Some(VoiceGenerationTuning::default()),
            id,
            name: "Saved voice".into(),
            model_id: model.into(),
            reference_kind: sayit_protocol::VoiceReferenceKind::Audio,
            reference_audio_path: "reference.wav".into(),
            reference_text: Some("The exact reference words.".into()),
            reference_duration_seconds: 4.0,
            created_at: chrono::Utc::now(),
            quality: VoiceQuality::default(),
            refinement_steps: 5,
        });
        select_voice(State(state.clone()), RoutePath(id))
            .await
            .unwrap();
        let (_, Json(saved)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        assert_eq!(saved.voice_mode, VoiceSynthesisMode::Standard);
        assert_eq!(saved.voice_profile_id, Some(id));
        state.inner.write().await.history.push(HistoryItem {
            job: SpeechJob {
                state: JobState::Completed,
                ..second.clone()
            },
            audio_path: None,
            duration_seconds: None,
            pinned: false,
            spoken_chunks: Vec::new(),
        });
        let (_, Json(regenerated)) = regenerate_history(State(state.clone()), RoutePath(second.id))
            .await
            .unwrap();
        assert_eq!(
            regenerated.voice_mode,
            VoiceSynthesisMode::RandomPerParagraph
        );
        assert!(regenerated.voice_profile_id.is_none());
        assert_eq!(
            state.inner.read().await.settings.active_voice_profile_id,
            Some(id)
        );
        state.persist().await.unwrap();
        let restored = ServiceState::open(directory.path().join("data"), None).unwrap();
        assert_eq!(
            restored
                .inner
                .read()
                .await
                .jobs
                .iter()
                .find(|job| job.id == first.id)
                .unwrap()
                .voice_mode,
            VoiceSynthesisMode::AutomaticStable
        );
    }

    #[tokio::test]
    async fn unsupported_or_conflicting_modes_fail_without_replacing_existing_work() {
        let state = ServiceState::default();
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"An existing reading.","queue_policy":"append"}),
        )
        .unwrap();
        let _ = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        let error = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                voice_mode: Some(VoiceSynthesisMode::RandomPerParagraph),
                queue_policy: QueuePolicy::Replace,
                ..request.clone()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(state.inner.read().await.jobs[0].state, JobState::Queued);
        let directory = tempfile::tempdir().unwrap();
        let native = crate::tests::native_voice_test_state(directory.path());
        let error = submit_job(
            State(native.clone()),
            Json(SpeechSubmission {
                voice_mode: Some(VoiceSynthesisMode::AutomaticStable),
                voice_profile_id: Some(Uuid::new_v4()),
                ..request
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(native.inner.read().await.jobs.is_empty());
    }

    #[test]
    fn every_language_has_a_bounded_nonempty_anchor_and_legacy_jobs_default_to_standard() {
        for language in [
            "auto", "en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
        ] {
            assert!((20..=160).contains(&anchor_text(language).chars().count()));
        }
        let legacy:SpeechJob=serde_json::from_value(serde_json::json!({"id":Uuid::new_v4(),"text":"Old reading.","source":"history","state":"completed","created_at":chrono::Utc::now(),"error":null})).unwrap();
        assert_eq!(legacy.voice_mode, VoiceSynthesisMode::Standard);
    }

    #[test]
    #[ignore = "requires real native Qwen assets/runtime/worker via SAYIT_ONNX_TEST_MODEL, SAYIT_ONNX_TEST_RUNTIME and SAYIT_TEST_WORKER"]
    fn real_random_references_stay_stable_within_readings_and_paragraphs_and_clean_up() {
        let directory = tempfile::tempdir().unwrap();
        let config = EngineConfig::Qwen3TtsOnnx(sayit_core::QwenOnnxConfig {
            model: std::env::var_os("SAYIT_ONNX_TEST_MODEL").unwrap().into(),
            runtime: Some(std::env::var_os("SAYIT_ONNX_TEST_RUNTIME").unwrap().into()),
            worker: std::env::var_os("SAYIT_TEST_WORKER").unwrap().into(),
            device: "cpu".into(),
            precision: "int4".into(),
            num_threads: 4,
            language: "English".into(),
        });
        let engine =
            ManagedEngine::load(&config, &InferenceRuntime::new(directory.path()), false).unwrap();
        let cancelled = AtomicBool::new(false);
        let mut automatic =
            ReadingVoice::new(VoiceSynthesisMode::AutomaticStable, &config).unwrap();
        automatic.prepare(&engine, 0, &cancelled).unwrap();
        let first = automatic.reference().unwrap().0.to_owned();
        automatic.prepare(&engine, 99, &cancelled).unwrap();
        assert_eq!(automatic.reference().unwrap().0, first);
        let output = directory.path().join("automatic.wav");
        let render = |voice: &ReadingVoice, output: &Path| {
            let (audio, transcript) = voice.reference().unwrap();
            let mut options = generation_defaults(VoiceSynthesisMode::AutomaticStable).unwrap();
            options.seed = Some(417);
            let mut streamed = 0;
            let mut audible = false;
            engine
                .synthesize_streaming(
                    &SynthesisRequest {
                        generation: Some(&options),
                        text: "The morning is quiet and the words are clear.",
                        output,
                        reference_audio: Some(audio),
                        reference_text: Some(transcript),
                        speaking_pace: 1.0,
                    },
                    Some(&cancelled),
                    &mut |pcm, rate| {
                        assert!(!output.exists());
                        assert_eq!(rate, 24000);
                        assert!(pcm.iter().all(|sample| sample.is_finite()));
                        streamed += pcm.len();
                        audible |= pcm.iter().any(|sample| sample.abs() > 0.001);
                        Ok(())
                    },
                )
                .unwrap();
            assert!(audible && streamed > 12000);
            assert_eq!(
                hound::WavReader::open(output).unwrap().len() as usize,
                streamed
            );
        };
        render(&automatic, &output);
        drop(automatic);
        assert!(!first.exists());
        let mut paragraph =
            ReadingVoice::new(VoiceSynthesisMode::RandomPerParagraph, &config).unwrap();
        paragraph.prepare(&engine, 0, &cancelled).unwrap();
        let second = paragraph.reference().unwrap().0.to_owned();
        paragraph.prepare(&engine, 0, &cancelled).unwrap();
        assert_eq!(paragraph.reference().unwrap().0, second);
        let second_audio = fs::read(&second).unwrap();
        let left = directory.path().join("paragraph-one.wav");
        render(&paragraph, &left);
        paragraph.prepare(&engine, 1, &cancelled).unwrap();
        let third = paragraph.reference().unwrap().0.to_owned();
        assert_ne!(second_audio, fs::read(&third).unwrap());
        assert!(!second.exists());
        let right = directory.path().join("paragraph-two.wav");
        render(&paragraph, &right);
        assert_ne!(fs::read(left).unwrap(), fs::read(right).unwrap());
        cancelled.store(true, Ordering::Relaxed);
        assert!(paragraph.prepare(&engine, 2, &cancelled).is_err());
        drop(paragraph);
        assert!(!third.exists());
    }
}
