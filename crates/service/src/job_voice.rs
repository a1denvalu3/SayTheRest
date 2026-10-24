use super::*;

pub(super) fn submission_configuration(
    models: &ModelManager,
    path: &Path,
    model: Option<&str>,
    submission: &SpeechSubmission,
) -> Result<EngineConfig> {
    match submission.preset_voice_id.as_deref() {
        Some(preset) => {
            anyhow::ensure!(
                submission.voice_profile_id.is_none()
                    && submission
                        .voice_mode
                        .is_none_or(|mode| mode == VoiceSynthesisMode::Standard),
                "A preset voice cannot be combined with an explicit profile or random reading mode"
            );
            models.preset_voice_config(model.context("A preset voice requires a model")?, preset)
        }
        None => EngineConfig::from_path(path),
    }
}

pub(super) fn configuration(
    models: &ModelManager,
    path: &Path,
    job: &SpeechJob,
) -> Result<EngineConfig> {
    let mut config = match job.preset_voice_id.as_deref() {
        Some(preset) => models.preset_voice_config(
            job.model_id
                .as_deref()
                .context("The queued preset has no model ID")?,
            preset,
        )?,
        None => EngineConfig::from_path(path)?,
    };
    if let Some(language) = &job.language {
        config.set_language(language)?;
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "kokoro-int8-multi-lang-v1-1";

    fn state(root: &Path) -> ServiceState {
        let directory = root.join("model");
        fs::create_dir_all(directory.join("espeak-ng-data")).unwrap();
        for name in [
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "lexicon-us-en.txt",
            "lexicon-gb-en.txt",
            "lexicon-zh.txt",
        ] {
            fs::write(directory.join(name), b"UI metadata fixture").unwrap();
        }
        let config = EngineConfig::SherpaOnnxKokoro(sayit_core::KokoroConfig {
            executable: "unused-test-runtime".into(),
            model: directory.join("model.int8.onnx"),
            voices: directory.join("voices.bin"),
            tokens: directory.join("tokens.txt"),
            data_dir: directory.join("espeak-ng-data"),
            lexicons: vec![
                directory.join("lexicon-us-en.txt"),
                directory.join("lexicon-zh.txt"),
            ],
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 0,
        });
        let path = root.join("config.json");
        fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        ServiceState::open(root.join("data"), Some(path)).unwrap()
    }

    #[tokio::test]
    async fn queued_and_regenerated_presets_survive_selection_changes_and_restart() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"A queued reading.","queue_policy":"append"}),
        )
        .unwrap();
        let (_, Json(first)) = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        assert_eq!(first.preset_voice_id.as_deref(), Some("af_maple"));
        state.models.select_preset_voice(MODEL, "bf_vale").unwrap();
        let (_, Json(second)) = submit_job(State(state.clone()), Json(request))
            .await
            .unwrap();
        assert_eq!(second.preset_voice_id.as_deref(), Some("bf_vale"));
        let path = state.models.config_path(MODEL);
        let EngineConfig::SherpaOnnxKokoro(captured) =
            configuration(&state.models, &path, &first).unwrap()
        else {
            panic!("Kokoro configuration")
        };
        assert_eq!(captured.speaker_id, 0);
        assert!(captured.lexicons[0].ends_with("lexicon-us-en.txt"));
        {
            let mut inner = state.inner.write().await;
            inner.jobs[0].state = JobState::Completed;
            let completed = inner.jobs[0].clone();
            inner.history.push(HistoryItem {
                job: completed,
                audio_path: None,
                duration_seconds: None,
                pinned: false,
                spoken_chunks: Vec::new(),
            });
        }
        let (_, Json(regenerated)) = regenerate_history(State(state.clone()), RoutePath(first.id))
            .await
            .unwrap();
        assert_eq!(regenerated.preset_voice_id.as_deref(), Some("af_maple"));
        let EngineConfig::SherpaOnnxKokoro(selected) = EngineConfig::from_path(&path).unwrap()
        else {
            panic!("Kokoro selection")
        };
        assert_eq!(selected.speaker_id, 2);
        assert!(selected.lexicons[0].ends_with("lexicon-gb-en.txt"));
        state.persist().await.unwrap();
        let restored = ServiceState::open(directory.path().join("data"), None).unwrap();
        let persisted = restored
            .inner
            .read()
            .await
            .jobs
            .iter()
            .find(|job| job.id == regenerated.id)
            .unwrap()
            .clone();
        assert_eq!(persisted.preset_voice_id.as_deref(), Some("af_maple"));
        let EngineConfig::SherpaOnnxKokoro(captured) =
            configuration(&restored.models, &path, &persisted).unwrap()
        else {
            panic!("restored preset")
        };
        assert_eq!(captured.speaker_id, 0);
        let mut legacy = serde_json::to_value(&first).unwrap();
        legacy.as_object_mut().unwrap().remove("preset_voice_id");
        let legacy: SpeechJob = serde_json::from_value(legacy).unwrap();
        assert!(legacy.preset_voice_id.is_none());
        let EngineConfig::SherpaOnnxKokoro(current) =
            configuration(&state.models, &path, &legacy).unwrap()
        else {
            panic!("legacy configuration")
        };
        assert_eq!(current.speaker_id, 2);
    }

    #[tokio::test]
    async fn overrides_preserve_defaults_and_reject_conflicts_before_queue_side_effects() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        let request: SpeechSubmission = serde_json::from_value(
            serde_json::json!({"text":"An existing reading.","queue_policy":"append"}),
        )
        .unwrap();
        let _ = submit_job(State(state.clone()), Json(request.clone()))
            .await
            .unwrap();
        let stale = Uuid::new_v4();
        state.inner.write().await.settings.active_voice_profile_id = Some(stale);
        let (_, Json(overridden)) = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                preset_voice_id: Some("af_sol".into()),
                ..request.clone()
            }),
        )
        .await
        .unwrap();
        assert_eq!(overridden.preset_voice_id.as_deref(), Some("af_sol"));
        assert!(overridden.voice_profile_id.is_none());
        assert_eq!(
            state.inner.read().await.settings.active_voice_profile_id,
            Some(stale)
        );
        let configured = EngineConfig::from_path(&state.models.config_path(MODEL)).unwrap();
        assert_eq!(
            state
                .models
                .configured_preset_voice(MODEL, &configured)
                .as_deref(),
            Some("af_maple")
        );
        for policy in [QueuePolicy::Replace, QueuePolicy::Interrupt] {
            for (preset, profile, mode) in [
                ("unknown-preset", None, None),
                ("af_maple", Some(stale), None),
                ("af_maple", None, Some(VoiceSynthesisMode::AutomaticStable)),
            ] {
                let error = submit_job(
                    State(state.clone()),
                    Json(SpeechSubmission {
                        preset_voice_id: Some(preset.into()),
                        voice_profile_id: profile,
                        voice_mode: mode,
                        queue_policy: policy.clone(),
                        ..request.clone()
                    }),
                )
                .await
                .unwrap_err();
                assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
                let inner = state.inner.read().await;
                assert_eq!(inner.jobs.len(), 2);
                assert!(inner.jobs.iter().all(|job| job.state == JobState::Queued));
            }
        }
    }
}
