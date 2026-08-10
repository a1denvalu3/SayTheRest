use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "engine", rename_all = "kebab-case")]
pub enum EngineConfig {
    SherpaOnnxVits(VitsConfig),
    SherpaOnnxKokoro(KokoroConfig),
    SherpaOnnxKitten(KittenConfig),
    SherpaOnnxPocket(PocketConfig),
    Qwen3Tts(Qwen3TtsConfig),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Qwen3TtsMode {
    CustomVoice,
    VoiceDesign,
    VoiceClone,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen3TtsConfig {
    pub python: PathBuf,
    pub worker: PathBuf,
    pub model: PathBuf,
    pub mode: Qwen3TtsMode,
    #[serde(default = "default_qwen_device")]
    pub device: String,
    #[serde(default = "default_qwen_dtype")]
    pub dtype: String,
    #[serde(default = "default_qwen_language")]
    pub language: String,
    pub speaker: Option<String>,
    pub voice_description: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VitsConfig {
    #[serde(default = "default_executable")]
    pub executable: PathBuf,
    pub model: PathBuf,
    pub tokens: PathBuf,
    pub data_dir: Option<PathBuf>,
    pub lexicon: Option<PathBuf>,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_threads")]
    pub num_threads: usize,
    #[serde(default)]
    pub speaker_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PocketConfig {
    #[serde(default = "default_executable")]
    pub executable: PathBuf,
    pub lm_flow: PathBuf,
    pub lm_main: PathBuf,
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub text_conditioner: PathBuf,
    pub vocab_json: PathBuf,
    pub token_scores_json: PathBuf,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_threads")]
    pub num_threads: usize,
    #[serde(default = "default_steps")]
    pub num_steps: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KokoroConfig {
    #[serde(default = "default_executable")]
    pub executable: PathBuf,
    pub model: PathBuf,
    pub voices: PathBuf,
    pub tokens: PathBuf,
    pub data_dir: PathBuf,
    pub lexicons: Vec<PathBuf>,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_threads")]
    pub num_threads: usize,
    #[serde(default)]
    pub speaker_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KittenConfig {
    #[serde(default = "default_executable")]
    pub executable: PathBuf,
    pub model: PathBuf,
    pub voices: PathBuf,
    pub tokens: PathBuf,
    pub data_dir: PathBuf,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_threads")]
    pub num_threads: usize,
    #[serde(default)]
    pub speaker_id: u32,
}

fn default_steps() -> usize {
    5
}
fn default_qwen_device() -> String {
    "cpu".into()
}
fn default_qwen_dtype() -> String {
    "float32".into()
}
fn default_qwen_language() -> String {
    "Auto".into()
}

fn default_executable() -> PathBuf {
    "sherpa-onnx-offline-tts".into()
}
fn default_provider() -> String {
    "cpu".into()
}
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

impl EngineConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config: Self = serde_json::from_str(&raw)
            .with_context(|| format!("invalid config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::SherpaOnnxVits(vits) => {
                if vits.num_threads == 0 {
                    bail!("num_threads must be at least 1");
                }
                for (name, path) in [("model", &vits.model), ("tokens", &vits.tokens)] {
                    if !path.is_file() {
                        bail!("{name} file does not exist: {}", path.display());
                    }
                }
                for (name, path) in [("data_dir", &vits.data_dir), ("lexicon", &vits.lexicon)] {
                    if let Some(path) = path
                        && !path.exists()
                    {
                        bail!("{name} does not exist: {}", path.display());
                    }
                }
            }
            Self::SherpaOnnxPocket(pocket) => {
                if pocket.num_threads == 0 || pocket.num_steps == 0 {
                    bail!("num_threads and num_steps must be at least 1");
                }
                for (name, path) in [
                    ("lm_flow", &pocket.lm_flow),
                    ("lm_main", &pocket.lm_main),
                    ("encoder", &pocket.encoder),
                    ("decoder", &pocket.decoder),
                    ("text_conditioner", &pocket.text_conditioner),
                    ("vocab_json", &pocket.vocab_json),
                    ("token_scores_json", &pocket.token_scores_json),
                ] {
                    if !path.is_file() {
                        bail!("{name} file does not exist: {}", path.display());
                    }
                }
            }
            Self::SherpaOnnxKokoro(kokoro) => {
                if kokoro.num_threads == 0 {
                    bail!("num_threads must be at least 1");
                }
                for (name, path) in [
                    ("model", &kokoro.model),
                    ("voices", &kokoro.voices),
                    ("tokens", &kokoro.tokens),
                ] {
                    if !path.is_file() {
                        bail!("{name} file does not exist: {}", path.display());
                    }
                }
                if !kokoro.data_dir.is_dir() {
                    bail!("data_dir does not exist: {}", kokoro.data_dir.display());
                }
                if kokoro.lexicons.is_empty() || kokoro.lexicons.iter().any(|path| !path.is_file())
                {
                    bail!("Kokoro requires one or more existing lexicon files");
                }
            }
            Self::SherpaOnnxKitten(kitten) => {
                if kitten.num_threads == 0 {
                    bail!("num_threads must be at least 1");
                }
                for (name, path) in [
                    ("model", &kitten.model),
                    ("voices", &kitten.voices),
                    ("tokens", &kitten.tokens),
                ] {
                    if !path.is_file() {
                        bail!("{name} file does not exist: {}", path.display());
                    }
                }
                if !kitten.data_dir.is_dir() {
                    bail!("data_dir does not exist: {}", kitten.data_dir.display());
                }
            }
            Self::Qwen3Tts(qwen) => {
                for (name, path) in [("python", &qwen.python), ("worker", &qwen.worker)] {
                    if !path.is_file() {
                        bail!("{name} file does not exist: {}", path.display());
                    }
                }
                if !qwen.model.is_dir() {
                    bail!("model directory does not exist: {}", qwen.model.display());
                }
                if !matches!(qwen.dtype.as_str(), "float32" | "float16" | "bfloat16") {
                    bail!("Qwen dtype must be float32, float16, or bfloat16");
                }
                if matches!(qwen.mode, Qwen3TtsMode::CustomVoice)
                    && qwen.speaker.as_deref().is_none_or(str::is_empty)
                {
                    bail!("Qwen custom-voice models require a speaker");
                }
                if matches!(qwen.mode, Qwen3TtsMode::VoiceDesign)
                    && qwen.voice_description.as_deref().is_none_or(str::is_empty)
                {
                    bail!("Qwen voice-design models require a voice description");
                }
            }
        }
        Ok(())
    }
}
