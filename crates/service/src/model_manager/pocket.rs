use super::*;
use sayit_core::pocket as assets;

const ID: &str = "pocket-tts-int8";

pub(super) fn presets(config: Option<&EngineConfig>) -> Vec<ModelPresetVoice> {
    let selected = match config {
        Some(EngineConfig::SherpaOnnxPocket(config)) => config.preset_voice.as_deref(),
        _ => None,
    };
    assets::voice_pack()
        .voices
        .iter()
        .map(|voice| ModelPresetVoice {
            id: voice.id.clone(),
            name: voice.name.clone(),
            language: voice.language.clone(),
            selected: selected == Some(voice.id.as_str()),
            license: Some(voice.license.clone()),
            license_url: Some(voice.license_url.clone()),
            attribution: Some(voice.attribution.clone()),
        })
        .collect()
}

impl ModelManager {
    /// Voice-pack updates reuse weights only after verifying the exact files
    /// extracted from the pinned archive; damaged installations use a new download.
    pub(super) fn install_pocket_from_verified_files(
        &self,
        cancelled: &AtomicBool,
    ) -> Result<bool> {
        let pack = assets::voice_pack();
        let destination = self.root.join(ID);
        for file in &pack.files {
            anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
            let path = destination.join(&file.path);
            if !path.is_file()
                || fs::metadata(&path)?.len() != file.size
                || sha256_file(&path)? != file.sha256
            {
                return Ok(false);
            }
        }
        let temporary = tempfile::tempdir_in(&self.root)?;
        let staging = temporary.path();
        let total = pack.files.iter().map(|file| file.size).sum::<u64>()
            + pack.voices.iter().map(|voice| voice.size).sum::<u64>();
        let mut completed = 0;
        for file in &pack.files {
            anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
            fs::copy(destination.join(&file.path), staging.join(&file.path))?;
            completed += file.size;
            self.set_progress(ID, DownloadState::Installing, completed, total, None);
        }
        let mut config = config_for(ID, &destination, &self.executable)?;
        self.prepare_pocket_presets(
            staging,
            &destination,
            &mut config,
            cancelled,
            &mut completed,
            total,
        )?;
        fs::write(
            staging.join("config.json"),
            serde_json::to_vec_pretty(&config)?,
        )?;
        fs::write(
            staging.join("catalog-install.json"),
            serde_json::to_vec_pretty(&CatalogInstallManifest {
                id: ID.into(),
                sha256: assets::manifest_fingerprint(),
            })?,
        )?;
        anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
        replace_model_directory(staging, &destination)?;
        EngineConfig::from_path(&self.config_path(ID))?;
        self.set_progress(ID, DownloadState::Installed, total, total, None);
        Ok(true)
    }

    pub(super) fn pocket_preset_config(&self, id: &str) -> Result<EngineConfig> {
        let voice = assets::voice_pack()
            .voices
            .iter()
            .find(|voice| voice.id == id)
            .context("unknown Pocket preset")?;
        let EngineConfig::SherpaOnnxPocket(mut config) =
            EngineConfig::from_path(&self.config_path(ID))?
        else {
            bail!("Pocket is not installed");
        };
        let path = self.root.join(ID).join(&voice.path);
        anyhow::ensure!(path.is_file(), "update Pocket to install its preset voices");
        config.reference_audio = Some(path);
        config.preset_voice = Some(id.into());
        let config = EngineConfig::SherpaOnnxPocket(config);
        config.validate()?;
        Ok(config)
    }

    pub(super) fn pocket_update_available(&self) -> bool {
        if !self.config_path(ID).is_file() {
            return false;
        }
        let fingerprint = fs::read(self.root.join(ID).join("catalog-install.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CatalogInstallManifest>(&bytes).ok());
        if !fingerprint
            .is_some_and(|value| value.id == ID && value.sha256 == assets::manifest_fingerprint())
        {
            return true;
        }
        if EngineConfig::from_path(&self.config_path(ID)).is_err() {
            return true;
        }
        assets::voice_pack().voices.iter().any(|voice| {
            fs::metadata(self.root.join(ID).join(&voice.path))
                .map_or(true, |meta| !meta.is_file() || meta.len() != voice.size)
        })
    }

    pub(super) fn prepare_pocket_presets(
        &self,
        staging: &Path,
        destination: &Path,
        config: &mut EngineConfig,
        cancelled: &AtomicBool,
        completed: &mut u64,
        total: u64,
    ) -> Result<()> {
        let base = url::Url::parse("https://huggingface.co/")?;
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        let pack = assets::voice_pack();
        let mut notice = String::from(
            "# Pocket preset voice recordings\n\nSources selected by Kyutai for Pocket TTS. No user recordings are included.\n\n",
        );
        fs::create_dir_all(staging.join("voices"))?;
        for voice in &pack.voices {
            anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
            let target = staging.join(safe_repository_path(&voice.path)?);
            let existing = destination.join(&voice.path);
            if existing.is_file()
                && fs::metadata(&existing)?.len() == voice.size
                && sha256_file(&existing)? == voice.sha256
            {
                fs::copy(&existing, &target)?;
                *completed += voice.size;
                self.set_progress(ID, DownloadState::Downloading, *completed, total, None);
            } else {
                let source = HuggingFaceSibling {
                    rfilename: voice.source_path.clone(),
                    size: Some(voice.size),
                    lfs: Some(HuggingFaceLfs {
                        size: Some(voice.size),
                        sha256: Some(voice.sha256.clone()),
                    }),
                };
                let mut progress = |count| {
                    *completed += count;
                    self.set_progress(ID, DownloadState::Downloading, *completed, total, None);
                };
                download_hf_file(
                    &client,
                    &pack.repository,
                    &pack.revision,
                    &source,
                    None,
                    Some(&target),
                    &base,
                    Some(cancelled),
                    Some(&mut progress),
                )?;
            }
            let reader = hound::WavReader::open(&target)?;
            anyhow::ensure!(
                reader.spec().channels == 1 && reader.duration() > 0,
                "Pocket preset must contain mono audio"
            );
            notice.push_str(&format!("## {}\n\n{}\n\nLicense: {} — {}\n\nSource: https://huggingface.co/{}/blob/{}/{}\n\nSHA-256: {}\n\n",voice.name,voice.attribution,voice.license,voice.license_url,pack.repository,pack.revision,voice.source_path,voice.sha256));
        }
        fs::write(staging.join("voices/NOTICE.md"), notice)?;
        // The source archive's test clips are unrelated to the preset catalog.
        let examples = staging.join("test_wavs");
        if examples.is_dir() {
            fs::remove_dir_all(examples)?;
        }
        let current = fs::read(self.config_path(ID))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<EngineConfig>(&bytes).ok());
        let selected = match current {
            Some(EngineConfig::SherpaOnnxPocket(config)) => config.preset_voice,
            _ => None,
        };
        let selected = pack
            .voices
            .iter()
            .find(|voice| Some(voice.id.as_str()) == selected.as_deref())
            .unwrap_or(&pack.voices[0]);
        let EngineConfig::SherpaOnnxPocket(config) = config else {
            bail!("invalid Pocket installation configuration");
        };
        config.reference_audio = Some(destination.join(&selected.path));
        config.preset_voice = Some(selected.id.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_pocket_install_stays_cloneable_and_offers_preset_upgrade() {
        let storage = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let root = storage.path().join(ID);
        fs::create_dir_all(&root).unwrap();
        for file in &assets::voice_pack().files {
            fs::write(root.join(&file.path), []).unwrap();
        }
        let config = config_for(ID, &root, Path::new("unused-runtime")).unwrap();
        let mut legacy = serde_json::to_value(config).unwrap();
        let object = legacy.as_object_mut().unwrap();
        object.remove("reference_audio");
        object.remove("preset_voice");
        fs::write(
            manager.config_path(ID),
            serde_json::to_vec(&legacy).unwrap(),
        )
        .unwrap();
        let config = EngineConfig::from_path(&manager.config_path(ID)).unwrap();
        assert!(config.requires_reference());
        assert!(config.supports_voice_cloning());
        assert!(manager.pocket_update_available());
        let descriptor = manager
            .descriptors(None, &HashMap::new())
            .into_iter()
            .find(|model| model.id == ID)
            .unwrap();
        assert!(descriptor.installed && descriptor.requires_voice_profile);
        assert_eq!(descriptor.preset_voices.len(), 8);
        assert!(manager.pocket_preset_config("alba").is_err());
        assert!(manager.pocket_preset_config("../invalid").is_err());

        fs::create_dir(root.join("voices")).unwrap();
        fs::write(root.join("voices/alba.wav"), []).unwrap();
        let config = manager.pocket_preset_config("alba").unwrap();
        assert!(!config.requires_reference());
        assert!(
            config.supports_voice_cloning(),
            "presets must not disable optional cloning"
        );
    }

    #[test]
    fn cached_pocket_update_rejects_damaged_weights_and_honors_cancellation() {
        let storage = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let root = storage.path().join(ID);
        fs::create_dir_all(&root).unwrap();
        let config_path = manager.config_path(ID);
        fs::write(&config_path, b"active configuration").unwrap();
        let file = &assets::voice_pack().files[0];
        let path = root.join(&file.path);
        // Correct length alone must never permit a damaged weight to be reused.
        fs::File::create(&path).unwrap().set_len(file.size).unwrap();
        assert!(
            !manager
                .install_pocket_from_verified_files(&AtomicBool::new(false))
                .unwrap()
        );
        let error = manager
            .install_pocket_from_verified_files(&AtomicBool::new(true))
            .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(fs::read(config_path).unwrap(), b"active configuration");
        assert_eq!(fs::metadata(path).unwrap().len(), file.size);
    }
}
