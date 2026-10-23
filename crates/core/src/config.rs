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
    Qwen3TtsOnnx(QwenOnnxConfig),
    Phonon(PhononConfig),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PhononConfig {
    #[serde(default = "native_worker_path")]
    pub worker: PathBuf,
    pub model: PathBuf,
    #[serde(default = "default_phonon_voice")]
    pub voice: String,
}

fn default_phonon_voice() -> String {
    "default".into()
}

impl PhononConfig {
    pub fn voice_path(&self) -> Result<PathBuf> {
        anyhow::ensure!(
            !self.voice.is_empty()
                && self.voice.len() <= 128
                && self.voice != "."
                && self.voice != ".."
                && !self
                    .voice
                    .chars()
                    .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':')),
            "invalid Phonon voice name"
        );
        Ok(if self.voice == "default" {
            self.model.join("default-voice.safetensors")
        } else {
            self.model
                .join("voices")
                .join(format!("{}.safetensors", self.voice))
        })
    }
}

#[cfg(test)]
mod phonon_tests {
    use super::*;

    #[test]
    fn selected_voice_stays_inside_the_model_and_is_required() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = PhononConfig {
            worker: native_worker_path(),
            model: directory.path().into(),
            voice: "default".into(),
        };
        for voice in [
            "",
            ".",
            "..",
            "../escape",
            "..\\escape",
            "C:voice",
            "voice\nname",
        ] {
            config.voice = voice.into();
            assert!(config.voice_path().is_err(), "must reject {voice:?}");
        }
        config.voice = "Old Mortlake".into();
        assert_eq!(
            config.voice_path().unwrap(),
            directory.path().join("voices/Old Mortlake.safetensors")
        );
        for file in ["config.json", "tokenizer.json", "model.q8.gguf"] {
            fs::write(directory.path().join(file), b"fixture").unwrap();
        }
        assert!(
            EngineConfig::Phonon(config.clone()).validate().is_err(),
            "a missing selected voice cannot fall back silently"
        );
        fs::create_dir(directory.path().join("voices")).unwrap();
        fs::write(config.voice_path().unwrap(), b"fixture").unwrap();
        EngineConfig::Phonon(config).validate().unwrap();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QwenOnnxConfig {
    #[serde(default = "native_worker_path")]
    pub worker: PathBuf,
    pub model: PathBuf,
    pub runtime: Option<PathBuf>,
    #[serde(default = "default_qwen_device")]
    pub device: String,
    #[serde(default = "default_onnx_precision")]
    pub precision: String,
    #[serde(default = "default_threads")]
    pub num_threads: usize,
    #[serde(default = "default_qwen_language")]
    pub language: String,
}

fn default_onnx_precision() -> String {
    "int4".into()
}
pub fn native_worker_path() -> PathBuf {
    let name = if cfg!(windows) {
        "sayit-inference-worker.exe"
    } else {
        "sayit-inference-worker"
    };
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join(name)))
        .unwrap_or_else(|| name.into())
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
    "auto".into()
}
fn default_qwen_dtype() -> String {
    "auto".into()
}
fn default_qwen_language() -> String {
    "Auto".into()
}

fn default_executable() -> PathBuf {
    "sherpa-onnx-offline-tts".into()
}
fn default_provider() -> String {
    "auto".into()
}
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

impl EngineConfig {
    pub fn provider(&self) -> &str {
        match self {
            Self::SherpaOnnxVits(c) => &c.provider,
            Self::SherpaOnnxPocket(c) => &c.provider,
            Self::SherpaOnnxKokoro(c) => &c.provider,
            Self::SherpaOnnxKitten(c) => &c.provider,
            Self::Qwen3Tts(c) => &c.device,
            Self::Qwen3TtsOnnx(c) => &c.device,
            Self::Phonon(_) => "cpu",
        }
    }

    pub fn set_sherpa_provider(&mut self, provider: &str, executable: Option<&Path>) {
        let (target, binary) = match self {
            Self::SherpaOnnxVits(c) => (&mut c.provider, &mut c.executable),
            Self::SherpaOnnxPocket(c) => (&mut c.provider, &mut c.executable),
            Self::SherpaOnnxKokoro(c) => (&mut c.provider, &mut c.executable),
            Self::SherpaOnnxKitten(c) => (&mut c.provider, &mut c.executable),
            Self::Qwen3Tts(_) | Self::Qwen3TtsOnnx(_) | Self::Phonon(_) => return,
        };
        *target = provider.into();
        if let Some(path) = executable {
            *binary = path.into();
        }
    }

    pub fn cpu(&self) -> Self {
        let mut config = self.clone();
        if let Self::Qwen3Tts(qwen) = &mut config {
            qwen.device = "cpu".into();
            qwen.dtype = "float32".into();
        } else if let Self::Qwen3TtsOnnx(qwen) = &mut config {
            qwen.device = "cpu".into();
        } else {
            config.set_sherpa_provider("cpu", None);
        }
        config
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config: Self = serde_json::from_str(&raw)
            .with_context(|| format!("invalid config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn is_qwen(&self) -> bool {
        matches!(self, Self::Qwen3Tts(_) | Self::Qwen3TtsOnnx(_))
    }

    pub fn requires_reference(&self) -> bool {
        matches!(self, Self::SherpaOnnxPocket(_) | Self::Qwen3TtsOnnx(_))
            || matches!(self, Self::Qwen3Tts(qwen) if matches!(qwen.mode, Qwen3TtsMode::VoiceClone))
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Phonon(phonon) => {
                for name in ["config.json", "tokenizer.json", "model.q8.gguf"] {
                    anyhow::ensure!(
                        phonon.model.join(name).is_file(),
                        "Phonon is missing {name}"
                    );
                }
                anyhow::ensure!(
                    phonon.voice_path()?.is_file(),
                    "Phonon voice is not installed"
                );
            }
            Self::Qwen3TtsOnnx(qwen) => {
                anyhow::ensure!(
                    matches!(qwen.precision.as_str(), "int4" | "fp32"),
                    "native Qwen precision must be int4 or fp32"
                );
                anyhow::ensure!(
                    matches!(qwen.device.as_str(), "auto" | "cpu" | "cuda" | "cuda:0"),
                    "invalid native Qwen device"
                );
                anyhow::ensure!(
                    (1..=64).contains(&qwen.num_threads),
                    "native Qwen threads must be between 1 and 64"
                );
                for name in [
                    "model.json",
                    "vocab.json",
                    "merges.txt",
                    "tokenizer_config.json",
                ] {
                    anyhow::ensure!(
                        qwen.model.join(name).is_file(),
                        "native Qwen is missing {name}"
                    );
                }
                for name in [
                    "talker_cache",
                    "code_predictor",
                    "text_embed",
                    "codec_embed",
                    "residual_embed",
                    "speaker_encoder",
                    "tok_encoder",
                    "tok_decoder",
                ] {
                    anyhow::ensure!(
                        qwen.model
                            .join(&qwen.precision)
                            .join(format!("{name}.onnx"))
                            .is_file(),
                        "native Qwen is missing {name}"
                    );
                }
            }
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
                if !matches!(
                    qwen.dtype.as_str(),
                    "auto" | "float32" | "float16" | "bfloat16"
                ) {
                    bail!("Qwen dtype must be auto, float32, float16, or bfloat16");
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
