use crate::qwen_runtime::QwenRuntimeManager;
use anyhow::{Context, Result, bail};
use bzip2::read::BzDecoder;
use sayit_core::{
    EngineConfig, KittenConfig, KokoroConfig, PocketConfig, Qwen3TtsConfig, Qwen3TtsMode,
    SherpaOnnxEngine, VitsConfig,
};
use sayit_protocol::{
    DownloadSnapshot, DownloadState, ModelBenchmark, ModelCapabilities, ModelDescriptor,
    ModelPresetVoice,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone)]
pub struct ModelManager {
    root: PathBuf,
    executable: PathBuf,
    progress: Arc<Mutex<HashMap<String, DownloadSnapshot>>>,
    cancellations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    community: Arc<Mutex<Vec<CommunityEntry>>>,
    qwen_runtime: QwenRuntimeManager,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CommunityEntry {
    id: String,
    name: String,
    size: u64,
    license: String,
    license_url: String,
    voice_cloning: bool,
}

#[derive(Deserialize, Serialize)]
struct CatalogInstallManifest {
    id: String,
    sha256: String,
}

#[derive(Deserialize)]
struct HuggingFaceMetadata {
    sha: String,
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<HuggingFaceLfs>,
}

#[derive(Deserialize)]
struct HuggingFaceLfs {
    sha256: Option<String>,
    size: Option<u64>,
}

#[derive(Clone, Copy)]
struct CatalogEntry {
    id: &'static str,
    name: &'static str,
    archive_root: &'static str,
    url: &'static str,
    sha256: &'static str,
    size: u64,
    languages: &'static [&'static str],
    license: &'static str,
    license_url: &'static str,
    quality_note: &'static str,
    speed_note: &'static str,
    capabilities: ModelCapabilitiesStatic,
    preset_voices: &'static [&'static str],
    hf_repository: Option<&'static str>,
    hf_revision: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct ModelCapabilitiesStatic {
    preset_voices: bool,
    voice_cloning: bool,
    streaming: bool,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "piper-en-us-lessac-medium",
        name: "Piper Lessac Medium",
        archive_root: "vits-piper-en_US-lessac-medium",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
        sha256: "9e3febfacf0abf4270172d2958bcec246032b7e88efc2720840cc80c93de334e",
        size: 67_230_653,
        languages: &["en-US"],
        license: "MIT",
        license_url: "https://github.com/rhasspy/piper",
        quality_note: "Clear single-speaker English; dependable for long-form reading.",
        speed_note: "Lowest latency and memory use; optimized for CPU.",
        capabilities: ModelCapabilitiesStatic {
            preset_voices: true,
            voice_cloning: false,
            streaming: false,
        },
        preset_voices: &["lessac"],
        hf_repository: None,
        hf_revision: None,
    },
    CatalogEntry {
        id: "kokoro-int8-multi-lang-v1-1",
        name: "Kokoro 82M INT8",
        archive_root: "kokoro-int8-multi-lang-v1_1",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2",
        sha256: "a1e94694776049035c4f2c6529f003aaece993c76aae9a78995831c3c4dcafc6",
        size: 147_031_220,
        languages: &["en-US", "en-GB", "zh-CN"],
        license: "Apache-2.0",
        license_url: "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/blob/main/LICENSE",
        quality_note: "Natural multilingual speech with 53 named built-in voices.",
        speed_note: "INT8 CPU model; benchmark locally to compare it with Piper.",
        capabilities: ModelCapabilitiesStatic {
            preset_voices: true,
            voice_cloning: false,
            streaming: false,
        },
        preset_voices: KOKORO_VOICES,
        hf_repository: None,
        hf_revision: None,
    },
    CatalogEntry {
        id: "pocket-tts-int8",
        name: "PocketTTS INT8",
        archive_root: "sherpa-onnx-pocket-tts-int8-2026-01-26",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
        sha256: "2f3b88823cbbb9bf0b2477ec8ae7b3fec417b3a87b6bb5f256dba66f2ad967cb",
        size: 98_336_520,
        languages: &["en"],
        license: "MIT",
        license_url: "https://github.com/kyutai-labs/pocket-tts",
        quality_note: "Expressive zero-shot English voice cloning from a short reference.",
        speed_note: "Heavier than Piper; streaming-capable and still designed for local CPU use.",
        capabilities: ModelCapabilitiesStatic {
            preset_voices: false,
            voice_cloning: true,
            streaming: true,
        },
        preset_voices: &[],
        hf_repository: None,
        hf_revision: None,
    },
    CatalogEntry {
        id: "kitten-mini-en-v0-8",
        name: "Kitten Mini 0.8",
        archive_root: "kitten-mini-en-v0_8",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-mini-en-v0_8.tar.bz2",
        sha256: "518f9b130320f690d5b5476df77bde4215fca67773cda16710318e5081234b9d",
        size: 67_547_594,
        languages: &["en"],
        license: "Apache-2.0",
        license_url: "https://huggingface.co/KittenML/kitten-tts-mini-0.8",
        quality_note: "Eight expressive English voices; experimental and less natural than larger models.",
        speed_note: "Small 74M-parameter CPU model; benchmark locally before selecting it.",
        capabilities: ModelCapabilitiesStatic {
            preset_voices: true,
            voice_cloning: false,
            streaming: true,
        },
        preset_voices: KITTEN_VOICES,
        hf_repository: None,
        hf_revision: None,
    },
    CatalogEntry {
        id: "kitten-nano-en-v0-8-int8",
        name: "Kitten Nano 0.8 INT8",
        archive_root: "kitten-nano-en-v0_8-int8",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_8-int8.tar.bz2",
        sha256: "6fa5be852612ce761094ba74ee6123b4fc4acfefa79bf64dc63acae4a83af2fd",
        size: 31_220_690,
        languages: &["en"],
        license: "Apache-2.0",
        license_url: "https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32",
        quality_note: "Extremely compact English speech with eight voices; quality is intentionally modest.",
        speed_note: "INT8 CPU model optimized for the smallest download and memory footprint.",
        capabilities: ModelCapabilitiesStatic {
            preset_voices: true,
            voice_cloning: false,
            streaming: true,
        },
        preset_voices: KITTEN_VOICES,
        hf_repository: None,
        hf_revision: None,
    },
    CatalogEntry {
        id: "qwen3-tts-06b-base",
        name: "Qwen3 TTS 0.6B Base",
        archive_root: "",
        url: "",
        sha256: "5d83992436eae1d760afd27aff78a71d676296fc",
        size: 2_516_106_051,
        languages: &[
            "auto", "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it",
        ],
        license: "Apache-2.0",
        license_url: "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        quality_note: "Expressive multilingual zero-shot voice cloning and random voice sampling.",
        speed_note: "Large generative model; installs a managed CPU inference runtime and should be benchmarked locally.",
        capabilities: ModelCapabilitiesStatic {
            preset_voices: false,
            voice_cloning: true,
            streaming: true,
        },
        preset_voices: &[],
        hf_repository: Some("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        hf_revision: Some("5d83992436eae1d760afd27aff78a71d676296fc"),
    },
];

const KITTEN_VOICES: &[&str] = &[
    "bella", "jasper", "luna", "bruno", "rosie", "hugo", "kiki", "leo",
];

// sherpa-onnx writes the Kitten embeddings male-first, while Kitten's friendly
// voice list is female-first. Keep the friendlier display order above and map
// each name to its actual position in voices.bin here.
const KITTEN_SPEAKER_IDS: &[u32] = &[1, 0, 3, 2, 5, 4, 7, 6];

fn preset_voice_speaker_id(entry: &CatalogEntry, voice_id: &str) -> Option<u32> {
    let position = entry
        .preset_voices
        .iter()
        .position(|candidate| *candidate == voice_id)?;
    if entry.preset_voices == KITTEN_VOICES {
        KITTEN_SPEAKER_IDS.get(position).copied()
    } else {
        Some(position as u32)
    }
}

fn packaged_qwen_manifest_dir() -> PathBuf {
    let packaged = std::env::current_exe().ok().and_then(|path| {
        path.parent()
            .map(|parent| parent.join("runtime/generative"))
    });
    if let Some(path) = packaged
        && path.join("qwen_worker.py").is_file()
    {
        return path;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime")
}

const KOKORO_VOICES: &[&str] = &[
    "af_maple", "af_sol", "bf_vale", "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006",
    "zf_007", "zf_008", "zf_017", "zf_018", "zf_019", "zf_021", "zf_022", "zf_023", "zf_024",
    "zf_026", "zf_027", "zf_028", "zf_032", "zf_036", "zf_038", "zf_039", "zf_040", "zf_042",
    "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049", "zf_051", "zf_059", "zf_060",
    "zf_067", "zf_070", "zf_071", "zf_072", "zf_073", "zf_074", "zf_075", "zf_076", "zf_077",
    "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086", "zf_087", "zf_088", "zf_090",
    "zf_092", "zf_093", "zf_094", "zf_099", "zm_009", "zm_010", "zm_011", "zm_012", "zm_013",
    "zm_014", "zm_015", "zm_016", "zm_020", "zm_025", "zm_029", "zm_030", "zm_031", "zm_033",
    "zm_034", "zm_035", "zm_037", "zm_041", "zm_045", "zm_050", "zm_052", "zm_053", "zm_054",
    "zm_055", "zm_056", "zm_057", "zm_058", "zm_061", "zm_062", "zm_063", "zm_064", "zm_065",
    "zm_066", "zm_068", "zm_069", "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", "zm_095",
    "zm_096", "zm_097", "zm_098", "zm_100",
];

impl ModelManager {
    pub fn new(root: PathBuf, initial_config: Option<&Path>) -> Result<Self> {
        fs::create_dir_all(&root)?;
        let executable = initial_config
            .and_then(|path| EngineConfig::from_path(path).ok())
            .map(|config| SherpaOnnxEngine::from_config(config).executable())
            .unwrap_or_else(|| {
                let name = if cfg!(windows) {
                    "sherpa-onnx-offline-tts.exe"
                } else {
                    "sherpa-onnx-offline-tts"
                };
                std::env::current_exe()
                    .ok()
                    .and_then(|path| path.parent().map(Path::to_owned))
                    .map(|dir| dir.join("runtime/bin").join(name))
                    .unwrap_or_else(|| name.into())
            });
        let community_path = root.join("community-models.json");
        let community = if community_path.is_file() {
            serde_json::from_slice(&fs::read(&community_path)?)
                .context("invalid community model catalog")?
        } else {
            Vec::new()
        };
        let qwen_runtime = QwenRuntimeManager::new(
            root.parent().unwrap_or(&root).join("runtimes/qwen"),
            packaged_qwen_manifest_dir(),
        );
        let manager = Self {
            root,
            executable,
            progress: Default::default(),
            cancellations: Default::default(),
            community: Arc::new(Mutex::new(community)),
            qwen_runtime,
        };
        manager.rebind_installed_runtime_paths()?;
        Ok(manager)
    }

    fn rebind_installed_runtime_paths(&self) -> Result<()> {
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let path = entry.path().join("config.json");
            if !path.is_file() {
                continue;
            }
            let Ok(mut config) = EngineConfig::from_path(&path) else {
                continue;
            };
            let changed = match &mut config {
                EngineConfig::SherpaOnnxVits(config) => {
                    replace_path(&mut config.executable, &self.executable)
                }
                EngineConfig::SherpaOnnxKokoro(config) => {
                    replace_path(&mut config.executable, &self.executable)
                }
                EngineConfig::SherpaOnnxKitten(config) => {
                    replace_path(&mut config.executable, &self.executable)
                }
                EngineConfig::SherpaOnnxPocket(config) => {
                    replace_path(&mut config.executable, &self.executable)
                }
                EngineConfig::Qwen3Tts(config) => {
                    replace_path(&mut config.python, &self.qwen_runtime.python())
                        | replace_path(&mut config.worker, &self.qwen_runtime.worker())
                }
            };
            if !changed {
                continue;
            }
            let temporary = path.with_extension("json.tmp");
            fs::write(&temporary, serde_json::to_vec_pretty(&config)?)?;
            fs::rename(temporary, path)?;
        }
        Ok(())
    }

    pub fn descriptors(
        &self,
        selected: Option<&str>,
        benchmarks: &HashMap<String, ModelBenchmark>,
    ) -> Vec<ModelDescriptor> {
        let recommended = benchmarks
            .iter()
            .filter(|(id, _)| self.config_path(id).is_file())
            .min_by(|left, right| left.1.mean_rtf.total_cmp(&right.1.mean_rtf))
            .map(|(id, _)| id.as_str());
        let mut descriptors = CATALOG
            .iter()
            .map(|entry| {
                let installed = self.config_path(entry.id).is_file();
                let selected_speaker = EngineConfig::from_path(&self.config_path(entry.id))
                    .ok()
                    .and_then(|config| match config {
                        EngineConfig::SherpaOnnxVits(config) => Some(config.speaker_id),
                        EngineConfig::SherpaOnnxKokoro(config) => Some(config.speaker_id),
                        EngineConfig::SherpaOnnxKitten(config) => Some(config.speaker_id),
                        EngineConfig::SherpaOnnxPocket(_) => None,
                        EngineConfig::Qwen3Tts(_) => None,
                    });
                ModelDescriptor {
                    id: entry.id.into(),
                    name: entry.name.into(),
                    languages: entry
                        .languages
                        .iter()
                        .map(|value| (*value).into())
                        .collect(),
                    size_bytes: entry.size,
                    license: entry.license.into(),
                    license_url: entry.license_url.into(),
                    installed,
                    update_available: self.catalog_update_available(entry),
                    resident: false,
                    selected: installed && selected == Some(entry.id),
                    capabilities: ModelCapabilities {
                        preset_voices: entry.capabilities.preset_voices,
                        voice_cloning: entry.capabilities.voice_cloning,
                        voice_description: false,
                        streaming: entry.capabilities.streaming,
                        long_form: true,
                    },
                    download: self.progress.lock().unwrap().get(entry.id).cloned(),
                    quality_note: entry.quality_note.into(),
                    speed_note: entry.speed_note.into(),
                    benchmark: benchmarks.get(entry.id).cloned(),
                    recommended: recommended == Some(entry.id),
                    supports_native_speaking_pace: !entry.capabilities.voice_cloning,
                    preset_voices: entry
                        .preset_voices
                        .iter()
                        .map(|id| ModelPresetVoice {
                            id: (*id).into(),
                            name: preset_voice_name(id),
                            language: preset_voice_language(id).into(),
                            selected: selected_speaker == preset_voice_speaker_id(entry, id),
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        descriptors.extend(
            self.community
                .lock()
                .unwrap()
                .iter()
                .map(|entry| ModelDescriptor {
                    id: entry.id.clone(),
                    name: entry.name.clone(),
                    languages: Vec::new(),
                    size_bytes: entry.size,
                    license: entry.license.clone(),
                    license_url: entry.license_url.clone(),
                    installed: self.config_path(&entry.id).is_file(),
                    update_available: false,
                    resident: false,
                    selected: self.config_path(&entry.id).is_file()
                        && selected == Some(entry.id.as_str()),
                    capabilities: ModelCapabilities {
                        preset_voices: !entry.voice_cloning,
                        voice_cloning: entry.voice_cloning,
                        voice_description: false,
                        streaming: entry.voice_cloning,
                        long_form: true,
                    },
                    download: None,
                    quality_note: "Community model — not tested by sayIt.".into(),
                    speed_note: "Run the local benchmark before relying on this model.".into(),
                    benchmark: benchmarks.get(&entry.id).cloned(),
                    recommended: recommended == Some(entry.id.as_str()),
                    supports_native_speaking_pace: !entry.voice_cloning,
                    preset_voices: Vec::new(),
                }),
        );
        descriptors
    }

    pub fn config_path(&self, id: &str) -> PathBuf {
        self.root.join(id).join("config.json")
    }

    pub fn select_preset_voice(&self, id: &str, voice_id: &str) -> Result<()> {
        let path = self.config_path(id);
        let config = self.preset_voice_config(id, voice_id)?;
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&config)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    pub fn preset_voice_config(&self, id: &str, voice_id: &str) -> Result<EngineConfig> {
        let entry = CATALOG
            .iter()
            .find(|entry| entry.id == id)
            .context("unknown curated model")?;
        let speaker_id = preset_voice_speaker_id(entry, voice_id)
            .context("unknown preset voice for this model")?;
        let mut config =
            EngineConfig::from_path(&self.config_path(id)).context("model is not installed")?;
        match &mut config {
            EngineConfig::SherpaOnnxVits(config) => config.speaker_id = speaker_id,
            EngineConfig::SherpaOnnxKokoro(config) => {
                config.speaker_id = speaker_id;
                let root = config
                    .model
                    .parent()
                    .context("Kokoro model path has no parent directory")?;
                let english = if voice_id.starts_with("bf_") || voice_id.starts_with("bm_") {
                    "lexicon-gb-en.txt"
                } else {
                    "lexicon-us-en.txt"
                };
                config.lexicons = vec![root.join(english), root.join("lexicon-zh.txt")];
            }
            EngineConfig::SherpaOnnxKitten(config) => config.speaker_id = speaker_id,
            EngineConfig::SherpaOnnxPocket(_) => bail!("model does not provide preset voices"),
            EngineConfig::Qwen3Tts(config) => config.speaker = Some(voice_id.into()),
        }
        Ok(config)
    }

    fn catalog_update_available(&self, entry: &CatalogEntry) -> bool {
        if !self.config_path(entry.id).is_file() {
            return false;
        }
        let manifest = self.root.join(entry.id).join("catalog-install.json");
        fs::read(manifest)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CatalogInstallManifest>(&bytes).ok())
            .is_none_or(|installed| installed.id != entry.id || installed.sha256 != entry.sha256)
    }

    pub fn start_install(&self, id: &str) -> Result<()> {
        let entry = *CATALOG
            .iter()
            .find(|entry| entry.id == id)
            .context("unknown model")?;
        if self.progress.lock().unwrap().get(id).is_some_and(|item| {
            matches!(
                item.state,
                DownloadState::Downloading | DownloadState::Verifying | DownloadState::Installing
            )
        }) {
            bail!("model installation is already running");
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        self.cancellations
            .lock()
            .unwrap()
            .insert(id.into(), cancelled.clone());
        self.set_progress(id, DownloadState::Downloading, 0, entry.size, None);
        let manager = self.clone();
        std::thread::Builder::new()
            .name(format!("model-install-{id}"))
            .spawn(move || {
                if let Err(error) = manager.install(entry, &cancelled) {
                    let state = if cancelled.load(Ordering::Relaxed) {
                        DownloadState::Cancelled
                    } else {
                        DownloadState::Failed
                    };
                    manager.set_progress(entry.id, state, 0, entry.size, Some(error.to_string()));
                }
                manager.cancellations.lock().unwrap().remove(entry.id);
            })?;
        Ok(())
    }

    pub fn cancel(&self, id: &str) -> Result<()> {
        let flag = self
            .cancellations
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .context("model is not downloading")?;
        flag.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn remove(&self, id: &str) -> Result<()> {
        let curated = CATALOG.iter().any(|entry| entry.id == id);
        let community = self
            .community
            .lock()
            .unwrap()
            .iter()
            .any(|entry| entry.id == id);
        anyhow::ensure!(curated || community, "unknown model");
        let target = self.root.join(id);
        if target.is_dir() {
            fs::remove_dir_all(target)?;
        }
        self.progress.lock().unwrap().remove(id);
        if community {
            self.community
                .lock()
                .unwrap()
                .retain(|entry| entry.id != id);
            self.persist_community()?;
        }
        Ok(())
    }

    pub fn import_local(
        &self,
        source: &Path,
        display_name: &str,
        license: &str,
        license_url: &str,
    ) -> Result<String> {
        let source = source
            .canonicalize()
            .with_context(|| format!("cannot open model source {}", source.display()))?;
        if source.is_dir() {
            return self.import_local_directory(&source, display_name, license, license_url);
        }
        anyhow::ensure!(
            source.is_file(),
            "model import source is not a file or directory"
        );
        let filename = source
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        anyhow::ensure!(
            filename.ends_with(".tar.bz2") || filename.ends_with(".tbz2"),
            "local model archives must use .tar.bz2 or .tbz2"
        );
        let fingerprint = sha256_file(&source)?;
        let extraction = self
            .root
            .join(format!(".archive-{}.importing", &fingerprint[..12]));
        if extraction.exists() {
            fs::remove_dir_all(&extraction)?;
        }
        fs::create_dir_all(&extraction)?;
        let result = (|| {
            extract_model_archive(&source, &extraction)?;
            let model_root = find_archive_model_root(&extraction)?;
            self.import_local_directory(&model_root, display_name, license, license_url)
        })();
        let cleanup = fs::remove_dir_all(&extraction);
        if result.is_ok() {
            cleanup.context("cannot remove temporary model extraction")?;
        }
        result
    }

    fn import_local_directory(
        &self,
        source: &Path,
        display_name: &str,
        license: &str,
        license_url: &str,
    ) -> Result<String> {
        validate_community_metadata(display_name, license, license_url)?;
        let name = display_name.trim();
        let source_config_path = source.join("config.json");
        let source_config: EngineConfig = serde_json::from_slice(
            &fs::read(&source_config_path)
                .context("community directory must contain a readable config.json")?,
        )
        .context("community directory must contain a supported config.json")?;
        relocate_config(source_config.clone(), source, source, &self.executable)
            .and_then(|config| config.validate())
            .context("community directory must contain a valid config.json")?;
        let (fingerprint, size) = directory_fingerprint(&source)?;
        let id = format!("community-local-{}", &fingerprint[..12]);
        let staging = self.root.join(format!(".{id}.importing"));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        copy_model_tree(&source, &staging)?;
        let config = relocate_config(source_config.clone(), &source, &staging, &self.executable)?;
        fs::write(
            staging.join("config.json"),
            serde_json::to_vec_pretty(&config)?,
        )?;
        EngineConfig::from_path(&staging.join("config.json"))?;
        let destination = self.root.join(&id);
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(&staging, &destination)?;
        let config = relocate_config(source_config, &source, &destination, &self.executable)?;
        fs::write(
            destination.join("config.json"),
            serde_json::to_vec_pretty(&config)?,
        )?;
        let voice_cloning = supports_voice_cloning(&config);
        let mut community = self.community.lock().unwrap();
        community.retain(|entry| entry.id != id);
        community.push(CommunityEntry {
            id: id.clone(),
            name: name.to_owned(),
            size,
            license: license.trim().to_owned(),
            license_url: license_url.to_owned(),
            voice_cloning,
        });
        drop(community);
        self.persist_community()?;
        Ok(id)
    }

    pub fn import_hugging_face(
        &self,
        repository: &str,
        revision: Option<&str>,
        display_name: &str,
        license: &str,
        license_url: &str,
        access_token: Option<&str>,
    ) -> Result<String> {
        self.import_hugging_face_from_base(
            repository,
            revision,
            display_name,
            license,
            license_url,
            access_token,
            &url::Url::parse("https://huggingface.co/")?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn import_hugging_face_from_base(
        &self,
        repository: &str,
        revision: Option<&str>,
        display_name: &str,
        license: &str,
        license_url: &str,
        access_token: Option<&str>,
        base_url: &url::Url,
    ) -> Result<String> {
        validate_community_metadata(display_name, license, license_url)?;
        let repository = repository.trim();
        let parts = repository.split('/').collect::<Vec<_>>();
        anyhow::ensure!(
            parts.len() == 2 && parts.iter().all(|part| valid_hf_component(part)),
            "repository must be a namespace/name Hugging Face ID"
        );
        let revision = revision
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("main");
        anyhow::ensure!(
            valid_hf_revision(revision),
            "revision contains unsupported characters"
        );
        let client = reqwest::blocking::Client::builder()
            .user_agent("sayit/0.1")
            .build()?;
        let mut metadata_url = base_url.join("api/models/")?;
        metadata_url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid Hugging Face API URL"))?
            .pop_if_empty()
            .extend([parts[0], parts[1], "revision", revision]);
        metadata_url.query_pairs_mut().append_pair("blobs", "true");
        let metadata: HuggingFaceMetadata = hf_request(&client, metadata_url, access_token)
            .send()?
            .error_for_status()
            .context("Hugging Face repository or revision is unavailable")?
            .json()?;
        anyhow::ensure!(
            metadata.sha.len() >= 12
                && metadata
                    .sha
                    .chars()
                    .all(|character| character.is_ascii_hexdigit()),
            "Hugging Face returned an invalid immutable revision"
        );
        let config_sibling = metadata
            .siblings
            .iter()
            .find(|file| file.rfilename == "config.json")
            .context("repository does not contain config.json")?;
        let config_bytes = download_hf_file(
            &client,
            repository,
            &metadata.sha,
            config_sibling,
            access_token,
            None,
            base_url,
            None,
            None,
        )?;
        let source_config: EngineConfig = serde_json::from_slice(&config_bytes).context(
            "repository config.json is not a supported sherpa/Piper or PocketTTS config",
        )?;
        let requirements = config_requirements(&source_config)?;
        let mut selected = vec![config_sibling];
        for (path, directory) in requirements {
            let matches = metadata.siblings.iter().filter(|file| {
                file.rfilename == path
                    || (directory && file.rfilename.starts_with(&format!("{path}/")))
            });
            let before = selected.len();
            for file in matches {
                if !selected
                    .iter()
                    .any(|existing| existing.rfilename == file.rfilename)
                {
                    selected.push(file);
                }
            }
            anyhow::ensure!(
                selected.len() > before,
                "repository is missing required model path {path}"
            );
        }
        let total = selected.iter().try_fold(0u64, |total, file| {
            total
                .checked_add(
                    file.lfs
                        .as_ref()
                        .and_then(|lfs| lfs.size)
                        .or(file.size)
                        .unwrap_or(0),
                )
                .context("model size overflow")
        })?;
        anyhow::ensure!(
            total <= 20 * 1024 * 1024 * 1024,
            "community model exceeds the 20 GB import limit"
        );
        let safe_repository = repository.to_ascii_lowercase().replace(['/', '_'], "-");
        let id = format!("community-{safe_repository}-{}", &metadata.sha[..8]);
        let staging = self.root.join(format!(".{id}.importing"));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;
        for file in selected {
            let relative = safe_repository_path(&file.rfilename)?;
            let destination = staging.join(relative);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            if file.rfilename == "config.json" {
                fs::write(destination, &config_bytes)?;
            } else {
                download_hf_file(
                    &client,
                    repository,
                    &metadata.sha,
                    file,
                    access_token,
                    Some(&destination),
                    base_url,
                    None,
                    None,
                )?;
            }
        }
        let relocated =
            relocate_config(source_config.clone(), &staging, &staging, &self.executable)?;
        fs::write(
            staging.join("config.json"),
            serde_json::to_vec_pretty(&relocated)?,
        )?;
        EngineConfig::from_path(&staging.join("config.json"))?;
        let destination = self.root.join(&id);
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(&staging, &destination)?;
        let relocated =
            relocate_config(source_config, &destination, &destination, &self.executable)?;
        fs::write(
            destination.join("config.json"),
            serde_json::to_vec_pretty(&relocated)?,
        )?;
        let voice_cloning = supports_voice_cloning(&relocated);
        let (_, actual_size) = directory_fingerprint(&destination)?;
        let mut community = self.community.lock().unwrap();
        community.retain(|entry| entry.id != id);
        community.push(CommunityEntry {
            id: id.clone(),
            name: display_name.trim().to_owned(),
            size: actual_size,
            license: license.trim().to_owned(),
            license_url: license_url.to_owned(),
            voice_cloning,
        });
        drop(community);
        self.persist_community()?;
        Ok(id)
    }

    fn persist_community(&self) -> Result<()> {
        let path = self.root.join("community-models.json");
        let temporary = self.root.join("community-models.json.tmp");
        fs::write(
            &temporary,
            serde_json::to_vec_pretty(&*self.community.lock().unwrap())?,
        )?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    fn install(&self, entry: CatalogEntry, cancelled: &AtomicBool) -> Result<()> {
        if entry.hf_repository.is_some() {
            self.install_hugging_face_catalog(entry, cancelled)
        } else {
            self.install_from(entry, entry.url, entry.sha256, cancelled)
        }
    }

    fn install_hugging_face_catalog(
        &self,
        entry: CatalogEntry,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let repository = entry
            .hf_repository
            .context("missing Hugging Face repository")?;
        let revision = entry
            .hf_revision
            .context("missing immutable Hugging Face revision")?;
        self.set_progress(entry.id, DownloadState::Installing, 0, entry.size, None);
        if !self.qwen_runtime.is_installed() {
            self.qwen_runtime.install()?;
        }
        if cancelled.load(Ordering::Relaxed) {
            bail!("download cancelled");
        }

        let base_url = url::Url::parse("https://huggingface.co/")?;
        let client = reqwest::blocking::Client::builder()
            .user_agent("sayit/0.1")
            .build()?;
        let mut metadata_url = base_url.join("api/models/")?;
        metadata_url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid Hugging Face API URL"))?
            .pop_if_empty()
            .extend(repository.split('/'))
            .extend(["revision", revision]);
        metadata_url.query_pairs_mut().append_pair("blobs", "true");
        let metadata: HuggingFaceMetadata = client
            .get(metadata_url)
            .send()?
            .error_for_status()?
            .json()?;
        anyhow::ensure!(
            metadata.sha == revision,
            "curated model revision changed: expected {revision}, received {}",
            metadata.sha
        );
        let selected = metadata
            .siblings
            .iter()
            .filter(|file| !matches!(file.rfilename.as_str(), ".gitattributes" | "README.md"))
            .collect::<Vec<_>>();
        let total = selected.iter().try_fold(0u64, |total, file| {
            total
                .checked_add(
                    file.lfs
                        .as_ref()
                        .and_then(|lfs| lfs.size)
                        .or(file.size)
                        .unwrap_or(0),
                )
                .context("model size overflow")
        })?;
        anyhow::ensure!(total <= entry.size, "curated model exceeds its pinned size");

        let staging = self.root.join(format!(".{}.installing", entry.id));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;
        let result = (|| {
            let mut downloaded = 0u64;
            for file in selected {
                let destination = staging.join(safe_repository_path(&file.rfilename)?);
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut progress = |count| {
                    downloaded += count;
                    self.set_progress(
                        entry.id,
                        DownloadState::Downloading,
                        downloaded,
                        total,
                        None,
                    );
                };
                download_hf_file(
                    &client,
                    repository,
                    revision,
                    file,
                    None,
                    Some(&destination),
                    &base_url,
                    Some(cancelled),
                    Some(&mut progress),
                )?;
            }
            self.set_progress(entry.id, DownloadState::Verifying, downloaded, total, None);
            let destination = self.root.join(entry.id);
            let config = EngineConfig::Qwen3Tts(Qwen3TtsConfig {
                python: self.qwen_runtime.python(),
                worker: self.qwen_runtime.worker(),
                model: destination.clone(),
                mode: Qwen3TtsMode::VoiceClone,
                device: "cpu".into(),
                dtype: "float32".into(),
                language: "Auto".into(),
                speaker: None,
                voice_description: None,
            });
            fs::write(
                staging.join("config.json"),
                serde_json::to_vec_pretty(&config)?,
            )?;
            fs::write(
                staging.join("catalog-install.json"),
                serde_json::to_vec_pretty(&CatalogInstallManifest {
                    id: entry.id.into(),
                    sha256: revision.into(),
                })?,
            )?;
            replace_model_directory(&staging, &destination)?;
            EngineConfig::from_path(&destination.join("config.json"))?;
            self.set_progress(entry.id, DownloadState::Installed, total, total, None);
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(staging);
        }
        result
    }

    fn install_from(
        &self,
        entry: CatalogEntry,
        url: &str,
        expected_sha256: &str,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        let downloads = self.root.join(".downloads");
        fs::create_dir_all(&downloads)?;
        let archive_path = downloads.join(format!("{}.tar.bz2", entry.id));
        let staging = downloads.join(format!("{}-staging", entry.id));
        let result = (|| {
            let mut response = reqwest::blocking::get(url)?.error_for_status()?;
            let total = response.content_length().unwrap_or(entry.size);
            let mut output = File::create(&archive_path)?;
            let mut downloaded = 0u64;
            let mut buffer = [0u8; 128 * 1024];
            loop {
                if cancelled.load(Ordering::Relaxed) {
                    bail!("download cancelled");
                }
                let count = response.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                output.write_all(&buffer[..count])?;
                downloaded += count as u64;
                self.set_progress(
                    entry.id,
                    DownloadState::Downloading,
                    downloaded,
                    total,
                    None,
                );
            }
            output.sync_all()?;
            drop(output);
            self.set_progress(entry.id, DownloadState::Verifying, downloaded, total, None);
            let digest = sha256_file(&archive_path)?;
            if digest != expected_sha256 {
                bail!(
                    "download checksum mismatch: expected {}, received {digest}",
                    expected_sha256
                );
            }
            if cancelled.load(Ordering::Relaxed) {
                bail!("download cancelled");
            }
            self.set_progress(entry.id, DownloadState::Installing, downloaded, total, None);
            if staging.is_dir() {
                fs::remove_dir_all(&staging)?;
            }
            fs::create_dir_all(&staging)?;
            tar::Archive::new(BzDecoder::new(File::open(&archive_path)?)).unpack(&staging)?;
            let unpacked = staging.join(entry.archive_root);
            validate_model_files(entry.id, &unpacked)?;
            let destination = self.root.join(entry.id);
            let config = config_for(entry.id, &destination, &self.executable)?;
            fs::write(
                unpacked.join("config.json"),
                serde_json::to_vec_pretty(&config)?,
            )?;
            fs::write(
                unpacked.join("catalog-install.json"),
                serde_json::to_vec_pretty(&CatalogInstallManifest {
                    id: entry.id.into(),
                    sha256: expected_sha256.into(),
                })?,
            )?;
            replace_model_directory(&unpacked, &destination)?;
            self.set_progress(entry.id, DownloadState::Installed, total, total, None);
            Ok(())
        })();
        let _ = fs::remove_file(&archive_path);
        let _ = fs::remove_dir_all(&staging);
        result
    }

    fn set_progress(
        &self,
        id: &str,
        state: DownloadState,
        downloaded_bytes: u64,
        total_bytes: u64,
        error: Option<String>,
    ) {
        self.progress.lock().unwrap().insert(
            id.into(),
            DownloadSnapshot {
                state,
                downloaded_bytes,
                total_bytes,
                error,
            },
        );
    }
}

fn replace_path(current: &mut PathBuf, expected: &Path) -> bool {
    if current == expected {
        false
    } else {
        *current = expected.to_owned();
        true
    }
}

fn preset_voice_name(id: &str) -> String {
    id.split('_')
        .last()
        .unwrap_or(id)
        .split('-')
        .map(|word| {
            let mut characters = word.chars();
            characters
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + characters.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn preset_voice_language(id: &str) -> &'static str {
    match id.get(..2) {
        Some("af" | "am") => "en-US",
        Some("bf" | "bm") => "en-GB",
        Some("zf" | "zm") => "zh-CN",
        Some("ef" | "em" | "ff" | "hf" | "hm" | "if" | "im" | "jf" | "jm" | "pf" | "pm") => "en",
        _ => "en-US",
    }
}

fn replace_model_directory(staged: &Path, destination: &Path) -> Result<()> {
    let backup = destination.with_extension("previous");
    if backup.exists() {
        fs::remove_dir_all(&backup)?;
    }
    let had_previous = destination.is_dir();
    if had_previous {
        fs::rename(destination, &backup).context("cannot preserve the installed model")?;
    }
    if let Err(error) = fs::rename(staged, destination) {
        if had_previous {
            let _ = fs::rename(&backup, destination);
        }
        return Err(error).context("cannot activate the downloaded model");
    }
    if had_previous {
        let _ = fs::remove_dir_all(backup);
    }
    Ok(())
}

fn extract_model_archive(archive_path: &Path, destination: &Path) -> Result<()> {
    const MAX_FILES: usize = 20_000;
    const MAX_BYTES: u64 = 20 * 1024 * 1024 * 1024;
    let mut archive = tar::Archive::new(BzDecoder::new(File::open(archive_path)?));
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let kind = entry.header().entry_type();
        anyhow::ensure!(
            kind.is_file() || kind.is_dir(),
            "model archives may contain only regular files and directories"
        );
        let path = entry.path()?;
        anyhow::ensure!(
            !path.is_absolute()
                && path.components().all(|component| matches!(
                    component,
                    std::path::Component::Normal(_) | std::path::Component::CurDir
                )),
            "model archive contains an unsafe path"
        );
        if kind.is_file() {
            files = files
                .checked_add(1)
                .context("archive file count overflow")?;
            bytes = bytes
                .checked_add(entry.header().size()?)
                .context("archive size overflow")?;
            anyhow::ensure!(files <= MAX_FILES, "model archive contains too many files");
            anyhow::ensure!(bytes <= MAX_BYTES, "model archive exceeds the 20 GB limit");
        }
        anyhow::ensure!(
            entry.unpack_in(destination)?,
            "model archive entry escapes the extraction directory"
        );
    }
    Ok(())
}

fn find_archive_model_root(extraction: &Path) -> Result<PathBuf> {
    fn visit(directory: &Path, configs: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = fs::symlink_metadata(entry.path())?;
            anyhow::ensure!(
                !metadata.file_type().is_symlink(),
                "model archives may not contain symbolic links"
            );
            if metadata.is_dir() {
                visit(&entry.path(), configs)?;
            } else if metadata.is_file() && entry.file_name() == "config.json" {
                configs.push(entry.path());
            }
        }
        Ok(())
    }

    let mut configs = Vec::new();
    visit(extraction, &mut configs)?;
    anyhow::ensure!(
        configs.len() == 1,
        "model archive must contain exactly one config.json"
    );
    Ok(configs.pop().unwrap().parent().unwrap().to_owned())
}

fn directory_fingerprint(root: &Path) -> Result<(String, u64)> {
    let mut files = Vec::new();
    collect_model_files(root, root, &mut files)?;
    anyhow::ensure!(
        files.len() <= 20_000,
        "model directory contains too many files"
    );
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    let mut size = 0u64;
    for (relative, path, bytes) in files {
        size = size.checked_add(bytes).context("model size overflow")?;
        anyhow::ensure!(
            size <= 20 * 1024 * 1024 * 1024,
            "model directory exceeds the 20 GB import limit"
        );
        digest.update(relative.as_bytes());
        digest.update(bytes.to_le_bytes());
        let mut file = File::open(path)?;
        let mut buffer = [0u8; 128 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok((format!("{:x}", digest.finalize()), size))
}

fn collect_model_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf, u64)>,
) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "model directories may not contain symbolic links"
        );
        if metadata.is_dir() {
            collect_model_files(root, &entry.path(), files)?;
        } else if metadata.is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            files.push((relative, entry.path(), metadata.len()));
        }
    }
    Ok(())
}

fn copy_model_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "model directories may not contain symbolic links"
        );
        let target = destination.join(entry.file_name());
        if metadata.is_dir() {
            copy_model_tree(&entry.path(), &target)?;
        } else if metadata.is_file() {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn relocate_config(
    config: EngineConfig,
    source: &Path,
    destination: &Path,
    executable: &Path,
) -> Result<EngineConfig> {
    // Windows canonicalization commonly adds the verbatim `\\?\` prefix. Compare
    // canonical paths on both sides so an in-tree model file is not mistaken for
    // an archive escape solely because the source retained its display form.
    let source = source.canonicalize()?;
    let relocate = |path: PathBuf| -> Result<PathBuf> {
        let path = if path.is_absolute() {
            path
        } else {
            source.join(path)
        };
        let path = path.canonicalize()?;
        anyhow::ensure!(
            path.starts_with(&source),
            "model config references a file outside the imported directory"
        );
        Ok(destination.join(path.strip_prefix(&source)?))
    };
    match config {
        EngineConfig::SherpaOnnxVits(config) => Ok(EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: executable.into(),
            model: relocate(config.model)?,
            tokens: relocate(config.tokens)?,
            data_dir: config.data_dir.map(&relocate).transpose()?,
            lexicon: config.lexicon.map(&relocate).transpose()?,
            provider: config.provider,
            num_threads: config.num_threads,
            speaker_id: config.speaker_id,
        })),
        EngineConfig::SherpaOnnxPocket(config) => {
            Ok(EngineConfig::SherpaOnnxPocket(PocketConfig {
                executable: executable.into(),
                lm_flow: relocate(config.lm_flow)?,
                lm_main: relocate(config.lm_main)?,
                encoder: relocate(config.encoder)?,
                decoder: relocate(config.decoder)?,
                text_conditioner: relocate(config.text_conditioner)?,
                vocab_json: relocate(config.vocab_json)?,
                token_scores_json: relocate(config.token_scores_json)?,
                provider: config.provider,
                num_threads: config.num_threads,
                num_steps: config.num_steps,
            }))
        }
        EngineConfig::SherpaOnnxKokoro(config) => {
            Ok(EngineConfig::SherpaOnnxKokoro(KokoroConfig {
                executable: executable.into(),
                model: relocate(config.model)?,
                voices: relocate(config.voices)?,
                tokens: relocate(config.tokens)?,
                data_dir: relocate(config.data_dir)?,
                lexicons: config
                    .lexicons
                    .into_iter()
                    .map(&relocate)
                    .collect::<Result<Vec<_>>>()?,
                provider: config.provider,
                num_threads: config.num_threads,
                speaker_id: config.speaker_id,
            }))
        }
        EngineConfig::SherpaOnnxKitten(config) => {
            Ok(EngineConfig::SherpaOnnxKitten(KittenConfig {
                executable: executable.into(),
                model: relocate(config.model)?,
                voices: relocate(config.voices)?,
                tokens: relocate(config.tokens)?,
                data_dir: relocate(config.data_dir)?,
                provider: config.provider,
                num_threads: config.num_threads,
                speaker_id: config.speaker_id,
            }))
        }
        EngineConfig::Qwen3Tts(mut config) => {
            config.model = relocate(config.model)?;
            Ok(EngineConfig::Qwen3Tts(config))
        }
    }
}

fn valid_hf_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
}

fn validate_community_metadata(display_name: &str, license: &str, license_url: &str) -> Result<()> {
    let name = display_name.trim();
    anyhow::ensure!(
        !name.is_empty() && name.chars().count() <= 100,
        "display name must contain 1 to 100 characters"
    );
    anyhow::ensure!(
        !license.trim().is_empty(),
        "a source license identifier is required"
    );
    anyhow::ensure!(
        license_url.starts_with("https://"),
        "license URL must use HTTPS"
    );
    Ok(())
}

fn supports_voice_cloning(config: &EngineConfig) -> bool {
    matches!(config, EngineConfig::SherpaOnnxPocket(_))
        || matches!(
            config,
            EngineConfig::Qwen3Tts(qwen)
                if matches!(qwen.mode, sayit_core::Qwen3TtsMode::VoiceClone)
        )
}

fn valid_hf_revision(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && !value.contains("..")
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/')
        })
}

fn safe_repository_path(value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    anyhow::ensure!(!path.is_absolute(), "repository file path must be relative");
    anyhow::ensure!(
        path.components()
            .all(|component| matches!(component, std::path::Component::Normal(_))),
        "repository contains an unsafe file path"
    );
    Ok(path.into())
}

fn config_requirements(config: &EngineConfig) -> Result<Vec<(String, bool)>> {
    let path = |path: &Path, directory: bool| -> Result<(String, bool)> {
        let path = safe_repository_path(&path.to_string_lossy())?;
        Ok((path.to_string_lossy().replace('\\', "/"), directory))
    };
    match config {
        EngineConfig::SherpaOnnxVits(config) => {
            let mut required = vec![path(&config.model, false)?, path(&config.tokens, false)?];
            if let Some(data_dir) = &config.data_dir {
                required.push(path(data_dir, true)?);
            }
            if let Some(lexicon) = &config.lexicon {
                required.push(path(lexicon, false)?);
            }
            Ok(required)
        }
        EngineConfig::SherpaOnnxPocket(config) => Ok(vec![
            path(&config.lm_flow, false)?,
            path(&config.lm_main, false)?,
            path(&config.encoder, false)?,
            path(&config.decoder, false)?,
            path(&config.text_conditioner, false)?,
            path(&config.vocab_json, false)?,
            path(&config.token_scores_json, false)?,
        ]),
        EngineConfig::SherpaOnnxKokoro(config) => {
            let mut required = vec![
                path(&config.model, false)?,
                path(&config.voices, false)?,
                path(&config.tokens, false)?,
                path(&config.data_dir, true)?,
            ];
            for lexicon in &config.lexicons {
                required.push(path(lexicon, false)?);
            }
            Ok(required)
        }
        EngineConfig::SherpaOnnxKitten(config) => Ok(vec![
            path(&config.model, false)?,
            path(&config.voices, false)?,
            path(&config.tokens, false)?,
            path(&config.data_dir, true)?,
        ]),
        EngineConfig::Qwen3Tts(config) => Ok(vec![path(&config.model, true)?]),
    }
}

fn hf_request(
    client: &reqwest::blocking::Client,
    url: url::Url,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    let request = client.get(url);
    match token.map(str::trim).filter(|token| !token.is_empty()) {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

fn download_hf_file(
    client: &reqwest::blocking::Client,
    repository: &str,
    revision: &str,
    file: &HuggingFaceSibling,
    token: Option<&str>,
    destination: Option<&Path>,
    base_url: &url::Url,
    cancelled: Option<&AtomicBool>,
    mut on_chunk: Option<&mut dyn FnMut(u64)>,
) -> Result<Vec<u8>> {
    let mut url = base_url.clone();
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| anyhow::anyhow!("invalid Hugging Face download URL"))?;
    segments.extend(repository.split('/'));
    segments.extend(["resolve", revision]);
    segments.extend(file.rfilename.split('/'));
    drop(segments);
    let mut response = hf_request(client, url, token)
        .send()?
        .error_for_status()
        .with_context(|| format!("failed to download {}", file.rfilename))?;
    let expected_size = file.lfs.as_ref().and_then(|lfs| lfs.size).or(file.size);
    let expected_sha = file.lfs.as_ref().and_then(|lfs| lfs.sha256.as_deref());
    let mut digest = Sha256::new();
    let mut bytes = Vec::new();
    let mut output = destination.map(File::create).transpose()?;
    let mut downloaded = 0u64;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        if cancelled.is_some_and(|cancelled| cancelled.load(Ordering::Relaxed)) {
            bail!("download cancelled");
        }
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        downloaded += count as u64;
        if let Some(callback) = &mut on_chunk {
            callback(count as u64);
        }
        anyhow::ensure!(
            downloaded <= 20 * 1024 * 1024 * 1024,
            "download exceeds the 20 GB file limit"
        );
        digest.update(&buffer[..count]);
        if let Some(output) = &mut output {
            output.write_all(&buffer[..count])?;
        } else {
            anyhow::ensure!(
                downloaded <= 10 * 1024 * 1024,
                "config.json exceeds the 10 MB limit"
            );
            bytes.extend_from_slice(&buffer[..count]);
        }
    }
    if let Some(expected) = expected_size {
        anyhow::ensure!(
            downloaded == expected,
            "download size mismatch for {}",
            file.rfilename
        );
    }
    if let Some(expected) = expected_sha {
        let received = format!("{:x}", digest.finalize());
        anyhow::ensure!(
            received.eq_ignore_ascii_case(expected),
            "checksum mismatch for {}",
            file.rfilename
        );
    }
    if let Some(output) = output {
        output.sync_all()?;
    }
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_model_files(id: &str, root: &Path) -> Result<()> {
    let required: &[&str] = match id {
        "piper-en-us-lessac-medium" => {
            &["en_US-lessac-medium.onnx", "tokens.txt", "espeak-ng-data"]
        }
        "pocket-tts-int8" => &[
            "lm_flow.int8.onnx",
            "lm_main.int8.onnx",
            "encoder.onnx",
            "decoder.int8.onnx",
            "text_conditioner.onnx",
            "vocab.json",
            "token_scores.json",
        ],
        "kokoro-int8-multi-lang-v1-1" => &[
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data",
            "lexicon-us-en.txt",
            "lexicon-gb-en.txt",
            "lexicon-zh.txt",
        ],
        "kitten-mini-en-v0-8" => &["model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data"],
        "kitten-nano-en-v0-8-int8" => &[
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data",
        ],
        _ => bail!("unknown model"),
    };
    for name in required {
        if !root.join(name).exists() {
            bail!("archive is missing required file {name}");
        }
    }
    Ok(())
}

fn config_for(id: &str, root: &Path, executable: &Path) -> Result<EngineConfig> {
    match id {
        "piper-en-us-lessac-medium" => Ok(EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: executable.into(),
            model: root.join("en_US-lessac-medium.onnx"),
            tokens: root.join("tokens.txt"),
            data_dir: Some(root.join("espeak-ng-data")),
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 0,
        })),
        "pocket-tts-int8" => Ok(EngineConfig::SherpaOnnxPocket(PocketConfig {
            executable: executable.into(),
            lm_flow: root.join("lm_flow.int8.onnx"),
            lm_main: root.join("lm_main.int8.onnx"),
            encoder: root.join("encoder.onnx"),
            decoder: root.join("decoder.int8.onnx"),
            text_conditioner: root.join("text_conditioner.onnx"),
            vocab_json: root.join("vocab.json"),
            token_scores_json: root.join("token_scores.json"),
            provider: "cpu".into(),
            num_threads: 4,
            num_steps: 5,
        })),
        "kokoro-int8-multi-lang-v1-1" => Ok(EngineConfig::SherpaOnnxKokoro(KokoroConfig {
            executable: executable.into(),
            model: root.join("model.int8.onnx"),
            voices: root.join("voices.bin"),
            tokens: root.join("tokens.txt"),
            data_dir: root.join("espeak-ng-data"),
            lexicons: vec![root.join("lexicon-us-en.txt"), root.join("lexicon-zh.txt")],
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 3,
        })),
        "kitten-mini-en-v0-8" => Ok(EngineConfig::SherpaOnnxKitten(KittenConfig {
            executable: executable.into(),
            model: root.join("model.onnx"),
            voices: root.join("voices.bin"),
            tokens: root.join("tokens.txt"),
            data_dir: root.join("espeak-ng-data"),
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 1,
        })),
        "kitten-nano-en-v0-8-int8" => Ok(EngineConfig::SherpaOnnxKitten(KittenConfig {
            executable: executable.into(),
            model: root.join("model.int8.onnx"),
            voices: root.join("voices.bin"),
            tokens: root.join("tokens.txt"),
            data_dir: root.join("espeak-ng-data"),
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 1,
        })),
        _ => bail!("unknown model"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_preset_and_cloning_models() {
        let manager = ModelManager::new(tempfile::tempdir().unwrap().path().into(), None).unwrap();
        let models = manager.descriptors(None, &HashMap::new());
        assert!(models.iter().any(|model| model.capabilities.preset_voices));
        assert!(models.iter().any(|model| model.capabilities.voice_cloning));
        assert!(models.iter().all(|model| {
            !model.name.is_empty()
                && !model.languages.is_empty()
                && model.size_bytes > 0
                && !model.quality_note.is_empty()
                && !model.speed_note.is_empty()
                && !model.license.is_empty()
                && model.license_url.starts_with("https://")
        }));
        let kokoro = models
            .iter()
            .find(|model| model.id == "kokoro-int8-multi-lang-v1-1")
            .unwrap();
        assert_eq!(kokoro.preset_voices.len(), 103);
        assert_eq!(kokoro.preset_voices[0].id, "af_maple");
        assert_eq!(kokoro.preset_voices[2].id, "bf_vale");
        assert_eq!(kokoro.preset_voices[3].language, "zh-CN");
        assert_eq!(kokoro.preset_voices[11].id, "zf_017");
        assert!(
            !kokoro
                .preset_voices
                .iter()
                .any(|voice| voice.id == "am_adam")
        );
        for id in ["kitten-mini-en-v0-8", "kitten-nano-en-v0-8-int8"] {
            let kitten = models.iter().find(|model| model.id == id).unwrap();
            assert_eq!(kitten.preset_voices.len(), 8);
            assert_eq!(kitten.preset_voices[1].id, "jasper");
            assert_eq!(kitten.preset_voices[1].language, "en-US");
            assert!(kitten.capabilities.streaming);
        }
        let uninstalled = manager.descriptors(Some("piper-en-us-lessac-medium"), &HashMap::new());
        let piper = uninstalled
            .iter()
            .find(|model| model.id == "piper-en-us-lessac-medium")
            .unwrap();
        assert!(!piper.installed);
        assert!(!piper.selected);
    }

    #[test]
    #[ignore = "downloads the pinned Qwen runtime and 0.6B Base weights, then performs CPU inference"]
    fn qwen_base_installs_and_synthesizes_offline() {
        let acceptance_root =
            PathBuf::from(std::env::var_os("SAYIT_QWEN_ACCEPTANCE_ROOT").unwrap());
        fs::create_dir_all(&acceptance_root).unwrap();
        let manager = ModelManager::new(acceptance_root.join("models"), None).unwrap();
        let entry = *CATALOG
            .iter()
            .find(|entry| entry.id == "qwen3-tts-06b-base")
            .unwrap();
        manager.install(entry, &AtomicBool::new(false)).unwrap();
        let config = EngineConfig::from_path(&manager.config_path(entry.id)).unwrap();
        let EngineConfig::Qwen3Tts(config) = config else {
            panic!("Qwen install wrote the wrong engine config")
        };
        let reference = PathBuf::from(std::env::var_os("SAYIT_QWEN_REFERENCE_AUDIO").unwrap());
        let output = acceptance_root.join("qwen-acceptance.wav");
        let mut engine = sayit_core::ResidentQwenEngine::load(&config).unwrap();
        engine
            .synthesize(&sayit_core::SynthesisRequest {
                text: "sayIt now runs Qwen voice cloning locally and offline.",
                output: &output,
                reference_audio: Some(&reference),
                speaking_pace: 1.0,
            })
            .unwrap();
        assert!(sayit_core::wav_duration_seconds(&output).unwrap() > 1.0);
    }

    #[test]
    fn catalog_install_manifest_drives_update_availability() {
        let storage = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let entry = &CATALOG[0];
        let model_dir = storage.path().join(entry.id);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), b"{}").unwrap();

        let models = manager.descriptors(None, &HashMap::new());
        assert!(
            models
                .iter()
                .find(|model| model.id == entry.id)
                .unwrap()
                .update_available
        );

        fs::write(
            model_dir.join("catalog-install.json"),
            serde_json::to_vec(&CatalogInstallManifest {
                id: entry.id.into(),
                sha256: entry.sha256.into(),
            })
            .unwrap(),
        )
        .unwrap();
        let models = manager.descriptors(None, &HashMap::new());
        assert!(
            !models
                .iter()
                .find(|model| model.id == entry.id)
                .unwrap()
                .update_available
        );
    }

    #[test]
    fn startup_rebinds_installed_models_to_the_current_packaged_runtime() {
        let storage = tempfile::tempdir().unwrap();
        let model_dir = storage.path().join("piper-en-us-lessac-medium");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.onnx"), b"model").unwrap();
        fs::write(model_dir.join("tokens.txt"), b"tokens").unwrap();
        let config = EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: "/tmp/.mount_old/usr/bin/runtime/bin/sherpa-onnx-offline-tts".into(),
            model: model_dir.join("model.onnx"),
            tokens: model_dir.join("tokens.txt"),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 2,
            speaker_id: 0,
        });
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let rebound = EngineConfig::from_path(&model_dir.join("config.json")).unwrap();
        let EngineConfig::SherpaOnnxVits(rebound) = rebound else {
            panic!("expected VITS config")
        };
        assert_eq!(rebound.executable, manager.executable);
    }

    #[test]
    fn selecting_a_kokoro_voice_persists_speaker_and_matching_lexicon() {
        let storage = tempfile::tempdir().unwrap();
        let model_dir = storage.path().join("kokoro-int8-multi-lang-v1-1");
        fs::create_dir_all(model_dir.join("espeak-ng-data")).unwrap();
        for name in [
            "model.int8.onnx",
            "voices.bin",
            "tokens.txt",
            "lexicon-us-en.txt",
            "lexicon-gb-en.txt",
            "lexicon-zh.txt",
        ] {
            fs::write(model_dir.join(name), b"test").unwrap();
        }
        let config = config_for(
            "kokoro-int8-multi-lang-v1-1",
            &model_dir,
            Path::new("stale-runtime"),
        )
        .unwrap();
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        manager
            .select_preset_voice("kokoro-int8-multi-lang-v1-1", "bf_vale")
            .unwrap();
        let selected = EngineConfig::from_path(&model_dir.join("config.json")).unwrap();
        let EngineConfig::SherpaOnnxKokoro(selected) = selected else {
            panic!("expected Kokoro config")
        };
        assert_eq!(selected.speaker_id, 2);
        assert!(selected.lexicons[0].ends_with("lexicon-gb-en.txt"));
        let descriptor = manager
            .descriptors(Some("kokoro-int8-multi-lang-v1-1"), &HashMap::new())
            .into_iter()
            .find(|model| model.id == "kokoro-int8-multi-lang-v1-1")
            .unwrap();
        assert_eq!(
            descriptor
                .preset_voices
                .iter()
                .find(|voice| voice.selected)
                .unwrap()
                .id,
            "bf_vale"
        );
    }

    #[test]
    fn kitten_friendly_names_map_to_sherpa_speaker_ids() {
        let storage = tempfile::tempdir().unwrap();
        let model_dir = storage.path().join("kitten-mini-en-v0-8");
        fs::create_dir_all(model_dir.join("espeak-ng-data")).unwrap();
        for name in ["model.onnx", "voices.bin", "tokens.txt"] {
            fs::write(model_dir.join(name), b"test").unwrap();
        }
        let config = config_for(
            "kitten-mini-en-v0-8",
            &model_dir,
            Path::new("stale-runtime"),
        )
        .unwrap();
        fs::write(
            model_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();

        let bella = manager
            .preset_voice_config("kitten-mini-en-v0-8", "bella")
            .unwrap();
        let EngineConfig::SherpaOnnxKitten(bella) = bella else {
            panic!("expected Kitten config")
        };
        assert_eq!(bella.speaker_id, 1);

        manager
            .select_preset_voice("kitten-mini-en-v0-8", "jasper")
            .unwrap();
        let descriptor = manager
            .descriptors(Some("kitten-mini-en-v0-8"), &HashMap::new())
            .into_iter()
            .find(|model| model.id == "kitten-mini-en-v0-8")
            .unwrap();
        assert_eq!(
            descriptor
                .preset_voices
                .iter()
                .find(|voice| voice.selected)
                .unwrap()
                .id,
            "jasper"
        );
    }

    #[test]
    fn rejects_unknown_model_before_network_access() {
        let manager = ModelManager::new(tempfile::tempdir().unwrap().path().into(), None).unwrap();
        assert!(manager.start_install("not-a-model").is_err());
    }

    #[test]
    fn imports_and_persists_self_contained_community_model() {
        let storage = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("model.onnx"), b"model").unwrap();
        fs::write(source.path().join("tokens.txt"), b"tokens").unwrap();
        let config = EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: "ignored-during-import".into(),
            model: source.path().join("model.onnx"),
            tokens: source.path().join("tokens.txt"),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 2,
            speaker_id: 0,
        });
        fs::write(
            source.path().join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let id = manager
            .import_local(
                source.path(),
                "Local voice",
                "MIT",
                "https://example.com/license",
            )
            .unwrap();
        assert!(id.starts_with("community-local-"));
        let installed = EngineConfig::from_path(&manager.config_path(&id)).unwrap();
        assert!(matches!(installed, EngineConfig::SherpaOnnxVits(_)));
        let reopened = ModelManager::new(storage.path().into(), None).unwrap();
        assert!(
            reopened
                .descriptors(None, &HashMap::new())
                .iter()
                .any(|model| model.id == id && model.installed)
        );
    }

    #[test]
    fn imports_a_self_contained_tar_bz2_model_archive() {
        let storage = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("model.onnx"), b"model").unwrap();
        fs::write(source.path().join("tokens.txt"), b"tokens").unwrap();
        let config = EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: "ignored-during-import".into(),
            model: "model.onnx".into(),
            tokens: "tokens.txt".into(),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 2,
            speaker_id: 0,
        });
        fs::write(
            source.path().join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let archive_path = storage.path().join("voice.tar.bz2");
        let encoder = bzip2::write::BzEncoder::new(
            File::create(&archive_path).unwrap(),
            bzip2::Compression::best(),
        );
        let mut archive = tar::Builder::new(encoder);
        archive.append_dir_all("voice", source.path()).unwrap();
        archive.into_inner().unwrap().finish().unwrap();

        let manager = ModelManager::new(storage.path().join("models"), None).unwrap();
        let id = manager
            .import_local(
                &archive_path,
                "Archived voice",
                "MIT",
                "https://example.com/license",
            )
            .unwrap();
        assert!(id.starts_with("community-local-"));
        assert!(manager.config_path(&id).is_file());
        assert!(
            fs::read_dir(storage.path().join("models"))
                .unwrap()
                .all(|entry| !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".archive-"))
        );
    }

    #[test]
    fn hugging_face_import_validation_rejects_unsafe_identifiers_and_paths() {
        assert!(valid_hf_component("safe-model_1"));
        assert!(!valid_hf_component("../model"));
        assert!(valid_hf_revision("refs/pr/12"));
        assert!(!valid_hf_revision("../../secret"));
        assert!(safe_repository_path("weights/model.onnx").is_ok());
        assert!(safe_repository_path("../model.onnx").is_err());
        let unsafe_config = EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: "ignored".into(),
            model: "/outside/model.onnx".into(),
            tokens: "tokens.txt".into(),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 1,
            speaker_id: 0,
        });
        assert!(config_requirements(&unsafe_config).is_err());
    }

    #[test]
    fn curated_install_verifies_download_installs_and_removes_atomically() {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;

        let entry = CATALOG[0];
        let encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::best());
        let mut builder = tar::Builder::new(encoder);
        for (path, contents) in [
            (
                format!("{}/en_US-lessac-medium.onnx", entry.archive_root),
                b"model".as_slice(),
            ),
            (
                format!("{}/tokens.txt", entry.archive_root),
                b"tokens".as_slice(),
            ),
            (
                format!("{}/espeak-ng-data/placeholder", entry.archive_root),
                b"data".as_slice(),
            ),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, path, contents).unwrap();
        }
        let archive = builder.into_inner().unwrap().finish().unwrap();
        let digest = format!("{:x}", Sha256::digest(&archive));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server_archive = archive.clone();
        let server = std::thread::spawn(move || {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    server_archive.len()
                )
                .unwrap();
                stream.write_all(&server_archive).unwrap();
            }
        });
        let storage = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let url = format!("http://{address}/model.tar.bz2");

        manager
            .install_from(entry, &url, &digest, &AtomicBool::new(false))
            .unwrap();
        let descriptor = manager
            .descriptors(None, &HashMap::new())
            .into_iter()
            .find(|model| model.id == entry.id)
            .unwrap();
        assert!(descriptor.installed);
        assert!(matches!(
            descriptor.download.unwrap().state,
            DownloadState::Installed
        ));
        assert!(EngineConfig::from_path(&manager.config_path(entry.id)).is_ok());
        let stale = manager.root.join(entry.id).join("stale-version");
        fs::write(&stale, b"old").unwrap();
        manager
            .install_from(entry, &url, &digest, &AtomicBool::new(false))
            .unwrap();
        assert!(!stale.exists());
        manager.remove(entry.id).unwrap();
        assert!(!manager.config_path(entry.id).exists());

        let checksum_error = manager
            .install_from(entry, &url, &"0".repeat(64), &AtomicBool::new(false))
            .unwrap_err();
        assert!(checksum_error.to_string().contains("checksum mismatch"));
        let cancellation = manager
            .install_from(entry, &url, &digest, &AtomicBool::new(true))
            .unwrap_err();
        assert!(cancellation.to_string().contains("cancelled"));
        assert!(
            !manager
                .root
                .join(".downloads")
                .join(format!("{}.tar.bz2", entry.id))
                .exists()
        );
        assert!(
            !manager
                .root
                .join(".downloads")
                .join(format!("{}-staging", entry.id))
                .exists()
        );
        server.join().unwrap();
    }

    #[test]
    fn imports_hugging_face_snapshot_at_immutable_revision() {
        use std::io::{BufRead, BufReader};
        use std::net::TcpListener;

        let model = b"mock-onnx".to_vec();
        let tokens = b"mock-tokens".to_vec();
        let source_config = EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: "ignored".into(),
            model: "model.onnx".into(),
            tokens: "tokens.txt".into(),
            data_dir: None,
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 1,
            speaker_id: 0,
        });
        let config = serde_json::to_vec(&source_config).unwrap();
        let revision = "a".repeat(40);
        let metadata = serde_json::to_vec(&serde_json::json!({
            "sha": revision,
            "siblings": [
                {"rfilename": "config.json", "size": config.len()},
                {"rfilename": "model.onnx", "lfs": {"size": model.len(), "sha256": format!("{:x}", Sha256::digest(&model))}},
                {"rfilename": "tokens.txt", "lfs": {"size": tokens.len(), "sha256": format!("{:x}", Sha256::digest(&tokens))}}
            ]
        })).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                }
                let path = request_line.split_whitespace().nth(1).unwrap();
                let (status, content_type, body) = if path.starts_with("/api/models/") {
                    ("200 OK", "application/json", metadata.as_slice())
                } else if path.ends_with("/config.json") {
                    ("200 OK", "application/json", config.as_slice())
                } else if path.ends_with("/model.onnx") {
                    ("200 OK", "application/octet-stream", model.as_slice())
                } else if path.ends_with("/tokens.txt") {
                    ("200 OK", "text/plain", tokens.as_slice())
                } else {
                    ("404 Not Found", "text/plain", b"missing".as_slice())
                };
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(body).unwrap();
            }
        });
        let storage = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(storage.path().into(), None).unwrap();
        let id = manager
            .import_hugging_face_from_base(
                "acme/voice",
                None,
                "Mock voice",
                "MIT",
                "https://example.com/license",
                None,
                &url::Url::parse(&format!("http://{address}/")).unwrap(),
            )
            .unwrap();
        server.join().unwrap();
        assert_eq!(id, "community-acme-voice-aaaaaaaa");
        let config = EngineConfig::from_path(&manager.config_path(&id)).unwrap();
        assert!(matches!(config, EngineConfig::SherpaOnnxVits(_)));
    }
}
