use super::*;
use sayit_core::{
    PhononConfig,
    phonon::{self, ModelAssets, VoicePack},
};

const ID: &str = "phonon-31m-en";

pub(super) fn presets(config: Option<&EngineConfig>) -> Vec<ModelPresetVoice> {
    let selected = match config {
        Some(EngineConfig::Phonon(config)) => Some(config.voice.as_str()),
        _ => None,
    };
    std::iter::once(("default", "en"))
        .chain(
            phonon::voice_pack()
                .voices
                .iter()
                .map(|voice| (voice.id.as_str(), voice.language.as_str())),
        )
        .map(|(id, language)| ModelPresetVoice {
            id: id.into(),
            name: if id == "default" {
                "Default".into()
            } else {
                id.into()
            },
            language: language.into(),
            selected: selected == Some(id),
        })
        .collect()
}

impl ModelManager {
    pub(super) fn phonon_preset_config(&self, voice: &str) -> Result<EngineConfig> {
        anyhow::ensure!(
            voice == "default"
                || phonon::voice_pack()
                    .voices
                    .iter()
                    .any(|item| item.id == voice),
            "unknown Phonon preset voice"
        );
        let EngineConfig::Phonon(mut config) = EngineConfig::from_path(&self.config_path(ID))?
        else {
            bail!("Phonon is not installed");
        };
        config.voice = voice.into();
        let config = EngineConfig::Phonon(config);
        config.validate()?;
        Ok(config)
    }

    pub(super) fn phonon_update_available(&self) -> bool {
        if !self.config_path(ID).is_file() {
            return false;
        }
        if EngineConfig::from_path(&self.config_path(ID)).is_err() {
            return true;
        }
        let installed = fs::read(self.root.join(ID).join("catalog-install.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CatalogInstallManifest>(&bytes).ok());
        if !installed.is_some_and(|manifest| {
            manifest.id == ID && manifest.sha256 == phonon::manifest_fingerprint()
        }) {
            return true;
        }
        phonon::model_assets()
            .files
            .iter()
            .chain(phonon::voice_pack().voices.iter().map(|voice| &voice.file))
            .any(|asset| {
                fs::metadata(self.root.join(ID).join(&asset.path))
                    .map_or(true, |meta| !meta.is_file() || meta.len() != asset.size)
            })
    }

    pub(super) fn install_phonon(&self, cancelled: &AtomicBool) -> Result<()> {
        anyhow::ensure!(
            sayit_core::native_worker_path().is_file(),
            "Native inference worker is missing; install the complete release or build the workspace"
        );
        self.install_phonon_files(
            phonon::model_assets(),
            phonon::voice_pack(),
            &url::Url::parse("https://huggingface.co/")?,
            cancelled,
        )
    }

    fn install_phonon_files(
        &self,
        model: &ModelAssets,
        voices: &VoicePack,
        base: &url::Url,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let destination = self.root.join(ID);
        fs::create_dir_all(&self.root)?;
        let temporary = tempfile::tempdir_in(&self.root)?;
        let staging = temporary.path();
        let total = model.files.iter().map(|file| file.size).sum::<u64>()
            + voices
                .voices
                .iter()
                .map(|voice| voice.file.size)
                .sum::<u64>();
        let mut completed = 0;
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        let assets = model
            .files
            .iter()
            .map(|file| {
                (
                    file,
                    model.repository.as_str(),
                    model.revision.as_str(),
                    file.path.as_str(),
                )
            })
            .chain(voices.voices.iter().map(|voice| {
                (
                    &voice.file,
                    voices.repository.as_str(),
                    voices.revision.as_str(),
                    voice.source_path.as_str(),
                )
            }));
        for (asset, repository, revision, source_path) in assets {
            anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
            let relative = safe_repository_path(&asset.path)?;
            let target = staging.join(&relative);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let existing = destination.join(&relative);
            // Reuse verified assets during voice-pack updates. A failed or
            // cancelled update cannot modify the active installation.
            if asset_matches(&existing, asset)? {
                fs::copy(&existing, &target)?;
                completed += asset.size;
                self.set_progress(ID, DownloadState::Downloading, completed, total, None);
                continue;
            }
            let source = HuggingFaceSibling {
                rfilename: source_path.into(),
                size: Some(asset.size),
                lfs: Some(HuggingFaceLfs {
                    sha256: Some(asset.sha256.clone()),
                    size: Some(asset.size),
                }),
            };
            let mut progress = |count| {
                completed += count;
                self.set_progress(ID, DownloadState::Downloading, completed, total, None);
            };
            download_hf_file(
                &client,
                repository,
                revision,
                &source,
                None,
                Some(&target),
                base,
                Some(cancelled),
                Some(&mut progress),
            )?;
        }
        self.set_progress(ID, DownloadState::Verifying, completed, total, None);
        for voice in &voices.voices {
            phonon::read_voice(&staging.join(&voice.file.path))?;
        }
        phonon::read_voice(&staging.join("default-voice.safetensors"))?;
        let selected = match EngineConfig::from_path(&self.config_path(ID)) {
            Ok(EngineConfig::Phonon(config))
                if config.voice == "default"
                    || voices.voices.iter().any(|voice| voice.id == config.voice) =>
            {
                config.voice
            }
            _ => "default".into(),
        };
        let staged = EngineConfig::Phonon(PhononConfig {
            worker: sayit_core::native_worker_path(),
            model: staging.into(),
            voice: selected.clone(),
        });
        staged.validate()?;
        let config = EngineConfig::Phonon(PhononConfig {
            worker: sayit_core::native_worker_path(),
            model: destination.clone(),
            voice: selected,
        });
        fs::write(
            staging.join(SAYIT_CONFIG),
            serde_json::to_vec_pretty(&config)?,
        )?;
        fs::write(
            staging.join("catalog-install.json"),
            serde_json::to_vec_pretty(&CatalogInstallManifest {
                id: ID.into(),
                sha256: phonon::manifest_fingerprint(),
            })?,
        )?;
        fs::write(
            staging.join("LICENSE.txt"),
            include_bytes!("../../../core/data/Apache-2.0.txt"),
        )?;
        fs::write(
            staging.join("NOTICE.txt"),
            include_bytes!("../../../core/data/PHONON-NOTICE.txt"),
        )?;
        anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "download cancelled");
        self.set_progress(ID, DownloadState::Installing, completed, total, None);
        replace_model_directory(staging, &destination)?;
        EngineConfig::from_path(&self.config_path(ID))?;
        self.set_progress(ID, DownloadState::Installed, total, total, None);
        Ok(())
    }
}

fn asset_matches(path: &Path, asset: &phonon::Asset) -> Result<bool> {
    let Ok(metadata) = fs::metadata(path) else {
        return Ok(false);
    };
    if !metadata.is_file() || metadata.len() != asset.size {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()) == asset.sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice_bytes() -> Vec<u8> {
        let header = serde_json::json!({"speaker_wavs":{"dtype":"F32","shape":[1,512,1],"data_offsets":[0,2048]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        for _ in 0..512 {
            bytes.extend_from_slice(&0.5_f32.to_le_bytes());
        }
        bytes
    }

    fn asset(path: &str, bytes: &[u8]) -> phonon::Asset {
        phonon::Asset {
            path: path.into(),
            size: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    }

    fn fixture(root: &Path) -> (ModelManager, ModelAssets, VoicePack) {
        let model = root.join(ID);
        fs::create_dir_all(model.join("voices")).unwrap();
        let voice = voice_bytes();
        let contents: Vec<(&str, &[u8])> = vec![
            ("config.json", b"{}"),
            ("tokenizer.json", b"{}"),
            ("model.q8.gguf", b"fixture"),
            ("default-voice.safetensors", &voice),
        ];
        for (path, bytes) in &contents {
            fs::write(model.join(path), bytes).unwrap();
        }
        fs::write(model.join("voices/Freya.safetensors"), &voice).unwrap();
        let config = EngineConfig::Phonon(PhononConfig {
            worker: "worker".into(),
            model: model.clone(),
            voice: "Freya".into(),
        });
        fs::write(
            model.join(SAYIT_CONFIG),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let manager = ModelManager::new(root.into(), None).unwrap();
        let assets = ModelAssets {
            id: ID.into(),
            repository: "gradium/phonon-31m-en".into(),
            revision: "fixture".into(),
            files: contents
                .into_iter()
                .map(|(path, bytes)| asset(path, bytes))
                .collect(),
        };
        let pack = VoicePack {
            repository: "gradium/phonon-voices".into(),
            revision: "fixture".into(),
            voices: vec![phonon::VoiceAsset {
                id: "Freya".into(),
                language: "en".into(),
                source_path: "voices/source.safetensors".into(),
                file: asset("voices/Freya.safetensors", &voice),
            }],
        };
        (manager, assets, pack)
    }

    #[test]
    fn cached_update_preserves_selection_and_separate_custom_profiles() {
        let root = tempfile::tempdir().unwrap();
        let models = root.path().join("models");
        let custom = root.path().join("custom.safetensors");
        fs::write(&custom, voice_bytes()).unwrap();
        let (manager, assets, pack) = fixture(&models);
        manager
            .install_phonon_files(
                &assets,
                &pack,
                &url::Url::parse("http://127.0.0.1:9/").unwrap(),
                &AtomicBool::new(false),
            )
            .unwrap();
        let EngineConfig::Phonon(config) =
            EngineConfig::from_path(&manager.config_path(ID)).unwrap()
        else {
            panic!()
        };
        assert_eq!(config.voice, "Freya");
        assert_eq!(config.model, models.join(ID));
        assert_eq!(fs::read(&custom).unwrap(), voice_bytes());
        let voices = presets(Some(&EngineConfig::Phonon(config)));
        assert_eq!(voices.len(), 38);
        assert_eq!(
            voices
                .iter()
                .filter(|voice| voice.selected)
                .map(|voice| voice.id.as_str())
                .collect::<Vec<_>>(),
            ["Freya"]
        );
    }

    #[test]
    fn cancelled_and_failed_updates_preserve_the_active_installation() {
        use std::io::{BufRead, BufReader, Write};
        let root = tempfile::tempdir().unwrap();
        let (manager, assets, mut pack) = fixture(root.path());
        let original = fs::read(manager.config_path(ID)).unwrap();
        let base = url::Url::parse("http://127.0.0.1:9/").unwrap();
        assert!(
            manager
                .install_phonon_files(&assets, &pack, &base, &AtomicBool::new(true))
                .is_err()
        );
        assert_eq!(fs::read(manager.config_path(ID)).unwrap(), original);
        pack.voices.push(phonon::VoiceAsset {
            id: "New voice".into(),
            language: "en".into(),
            source_path: "voices/new.safetensors".into(),
            file: asset("voices/New voice.safetensors", &voice_bytes()),
        });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = url::Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = BufReader::new(&mut stream);
            loop {
                let mut line = String::new();
                assert!(request.read_line(&mut line).unwrap() > 0);
                if line == "\r\n" {
                    break;
                }
            }
            drop(request);
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        assert!(
            manager
                .install_phonon_files(&assets, &pack, &base, &AtomicBool::new(false))
                .is_err()
        );
        server.join().unwrap();
        assert_eq!(fs::read(manager.config_path(ID)).unwrap(), original);
        assert!(
            !root
                .path()
                .join(ID)
                .join("voices/New voice.safetensors")
                .exists()
        );
        assert!(!fs::read_dir(root.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp")
        }));
    }
}
