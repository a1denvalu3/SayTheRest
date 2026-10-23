use super::*;
use sayit_protocol::{VoiceImportRequest, VoiceReferenceKind};

pub(super) async fn import_voice(
    State(state): State<ServiceState>,
    Json(request): Json<VoiceImportRequest>,
) -> Result<(StatusCode, Json<VoiceProfile>), (StatusCode, String)> {
    if request.kind != VoiceReferenceKind::PhononEmbedding {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "Use voice cloning to import an audio reference".into(),
        ));
    }
    let model_id = state
        .inner
        .read()
        .await
        .settings
        .active_model_id
        .clone()
        .ok_or((StatusCode::CONFLICT, "select a Phonon model first".into()))?;
    let config_path = state.models.config_path(&model_id);
    if !matches!(
        EngineConfig::from_path(&config_path).map_err(unprocessable)?,
        EngineConfig::Phonon(_)
    ) {
        return Err((
            StatusCode::CONFLICT,
            "this voice file requires a Phonon model".into(),
        ));
    }
    let name = request.name.trim();
    if name.is_empty() || name.chars().count() > 80 || name.chars().any(char::is_control) {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "voice name must contain 1 to 80 characters".into(),
        ));
    }
    let source = PathBuf::from(request.path);
    if source
        .extension()
        .and_then(|value| value.to_str())
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("safetensors"))
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "choose an exported Phonon .safetensors voice file".into(),
        ));
    }
    let bytes = tokio::task::spawn_blocking(move || sayit_core::phonon::read_voice(&source))
        .await
        .map_err(|error| internal_error(error.into()))?
        .map_err(unprocessable)?;
    let id = Uuid::new_v4();
    let destination = state.voices_dir.join(format!("{id}.safetensors"));
    fs::write(&destination, &bytes).map_err(|error| internal_error(error.into()))?;
    let profile = VoiceProfile {
        id,
        name: name.into(),
        model_id,
        reference_kind: VoiceReferenceKind::PhononEmbedding,
        reference_audio_path: destination.display().to_string(),
        reference_text: None,
        reference_duration_seconds: 0.0,
        created_at: chrono::Utc::now(),
        quality: VoiceQuality::default(),
        refinement_steps: 5,
    };
    state.inner.write().await.voices.push(profile.clone());
    state.persist().await.map_err(internal_error)?;
    state.emit("voices");
    Ok((StatusCode::CREATED, Json(profile)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exported_voice() -> Vec<u8> {
        let header=serde_json::json!({"speaker_wavs":{"dtype":"F32","shape":[1,512,1],"data_offsets":[0,2048]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        for _ in 0..512 {
            bytes.extend(0.25_f32.to_le_bytes());
        }
        bytes
    }

    fn setup(root: &Path) -> (ServiceState, PathBuf, PathBuf) {
        let model = root.join("assets");
        fs::create_dir(&model).unwrap();
        for name in ["config.json", "tokenizer.json", "model.q8.gguf"] {
            fs::write(model.join(name), b"fixture").unwrap();
        }
        fs::write(model.join("default-voice.safetensors"), exported_voice()).unwrap();
        let config = root.join("phonon.json");
        fs::write(
            &config,
            serde_json::to_vec(&EngineConfig::Phonon(sayit_core::PhononConfig {
                worker: "worker".into(),
                model,
                voice: "default".into(),
            }))
            .unwrap(),
        )
        .unwrap();
        let source = root.join("export.safetensors");
        fs::write(&source, exported_voice()).unwrap();
        (
            ServiceState::open(root.join("state"), Some(config.clone())).unwrap(),
            config,
            source,
        )
    }

    #[tokio::test]
    async fn imported_voice_is_copied_scoped_persisted_and_removed_locally() {
        let root = tempfile::tempdir().unwrap();
        let (state, config, source) = setup(root.path());
        let (_, Json(profile)) = import_voice(
            State(state.clone()),
            Json(VoiceImportRequest {
                name: "Narrator".into(),
                path: source.display().to_string(),
                kind: VoiceReferenceKind::PhononEmbedding,
            }),
        )
        .await
        .unwrap();
        assert_eq!(profile.reference_kind, VoiceReferenceKind::PhononEmbedding);
        assert_eq!(profile.model_id, "phonon-31m-en");
        assert_eq!(
            fs::read(&profile.reference_audio_path).unwrap(),
            exported_voice()
        );
        fs::remove_file(&source).unwrap();
        select_voice(State(state.clone()), RoutePath(profile.id))
            .await
            .unwrap();
        let (_, Json(job)) = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                language: None,
                text: "Imported voice preview".into(),
                source: SpeechSource::Desktop,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: true,
                voice_profile_id: None,
                speaking_pace: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(job.voice_profile_id, Some(profile.id));
        assert_eq!(job.speaking_pace, Some(1.0));
        assert!(
            tune_voice(
                State(state.clone()),
                RoutePath(profile.id),
                Json(VoiceTuningRequest {
                    refinement_steps: 5
                })
            )
            .await
            .is_err()
        );
        assert!(
            delete_voice(State(state.clone()), RoutePath(profile.id))
                .await
                .is_err()
        );
        let restored = ServiceState::open(root.path().join("state"), Some(config)).unwrap();
        assert_eq!(
            restored.inner.read().await.voices[0].reference_kind,
            VoiceReferenceKind::PhononEmbedding
        );
        select_default_voice(State(restored.clone())).await.unwrap();
        delete_voice(State(restored.clone()), RoutePath(profile.id))
            .await
            .unwrap();
        assert!(!Path::new(&profile.reference_audio_path).exists());
        assert!(restored.inner.read().await.voices.is_empty());
        restored.process_job(job.clone()).await;
        assert_eq!(
            restored
                .inner
                .read()
                .await
                .jobs
                .iter()
                .find(|entry| entry.id == job.id)
                .unwrap()
                .state,
            JobState::Failed
        );
    }

    #[tokio::test]
    async fn invalid_import_and_native_pace_leave_no_profile_or_job() {
        let root = tempfile::tempdir().unwrap();
        let (state, _, source) = setup(root.path());
        fs::write(&source, b"corrupt voice file").unwrap();
        let error = import_voice(
            State(state.clone()),
            Json(VoiceImportRequest {
                name: "Broken".into(),
                path: source.display().to_string(),
                kind: VoiceReferenceKind::PhononEmbedding,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(error.0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(state.inner.read().await.voices.is_empty());
        assert_eq!(fs::read_dir(&state.voices_dir).unwrap().count(), 0);
        let result = submit_job(
            State(state.clone()),
            Json(SpeechSubmission {
                language: None,
                text: "Please read".into(),
                source: SpeechSource::Desktop,
                queue_policy: QueuePolicy::Append,
                confirmed_long_text: true,
                voice_profile_id: None,
                speaking_pace: Some(1.25),
            }),
        )
        .await;
        assert_eq!(result.unwrap_err().0, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(state.inner.read().await.jobs.is_empty());
    }
}
