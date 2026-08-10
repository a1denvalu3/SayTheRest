use crate::EngineConfig;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug)]
pub struct SynthesisRequest<'a> {
    pub text: &'a str,
    pub output: &'a Path,
    pub reference_audio: Option<&'a Path>,
    pub speaking_pace: f64,
}

pub trait TtsEngine {
    fn synthesize(&self, request: &SynthesisRequest<'_>) -> Result<()>;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxEngine {
    config: EngineConfig,
}

impl SherpaOnnxEngine {
    pub fn from_config(config: EngineConfig) -> Self {
        Self { config }
    }

    fn arg(name: &str, path: &Path) -> String {
        format!("--{name}={}", path.display())
    }

    pub fn command_args(&self, request: &SynthesisRequest<'_>) -> Result<Vec<String>> {
        if matches!(self.config, EngineConfig::Qwen3Tts(_)) {
            bail!("Qwen models require the persistent generative worker");
        }
        anyhow::ensure!(
            request.speaking_pace.is_finite() && (0.5..=2.0).contains(&request.speaking_pace),
            "speaking pace must be between 0.5 and 2.0"
        );
        let mut args = match &self.config {
            EngineConfig::SherpaOnnxVits(config) => {
                let mut args = vec![
                    Self::arg("vits-model", &config.model),
                    Self::arg("vits-tokens", &config.tokens),
                    format!("--sid={}", config.speaker_id),
                    format!("--vits-length-scale={}", 1.0 / request.speaking_pace),
                ];
                if let Some(path) = &config.data_dir {
                    args.push(Self::arg("vits-data-dir", path));
                }
                if let Some(path) = &config.lexicon {
                    args.push(Self::arg("vits-lexicon", path));
                }
                args
            }
            EngineConfig::SherpaOnnxPocket(config) => {
                let reference = request
                    .reference_audio
                    .context("the selected PocketTTS model requires a voice profile")?;
                vec![
                    Self::arg("pocket-lm-flow", &config.lm_flow),
                    Self::arg("pocket-lm-main", &config.lm_main),
                    Self::arg("pocket-encoder", &config.encoder),
                    Self::arg("pocket-decoder", &config.decoder),
                    Self::arg("pocket-text-conditioner", &config.text_conditioner),
                    Self::arg("pocket-vocab-json", &config.vocab_json),
                    Self::arg("pocket-token-scores-json", &config.token_scores_json),
                    Self::arg("reference-audio", reference),
                    format!("--num-steps={}", config.num_steps),
                ]
            }
            EngineConfig::SherpaOnnxKokoro(config) => vec![
                Self::arg("kokoro-model", &config.model),
                Self::arg("kokoro-voices", &config.voices),
                Self::arg("kokoro-tokens", &config.tokens),
                Self::arg("kokoro-data-dir", &config.data_dir),
                format!(
                    "--kokoro-lexicon={}",
                    config
                        .lexicons
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                format!("--sid={}", config.speaker_id),
                format!("--kokoro-length-scale={}", 1.0 / request.speaking_pace),
            ],
            EngineConfig::SherpaOnnxKitten(config) => vec![
                Self::arg("kitten-model", &config.model),
                Self::arg("kitten-voices", &config.voices),
                Self::arg("kitten-tokens", &config.tokens),
                Self::arg("kitten-data-dir", &config.data_dir),
                format!("--sid={}", config.speaker_id),
                format!("--kitten-length-scale={}", 1.0 / request.speaking_pace),
            ],
            EngineConfig::Qwen3Tts(_) => unreachable!(),
        };
        let (provider, threads) = match &self.config {
            EngineConfig::SherpaOnnxVits(config) => (&config.provider, config.num_threads),
            EngineConfig::SherpaOnnxPocket(config) => (&config.provider, config.num_threads),
            EngineConfig::SherpaOnnxKokoro(config) => (&config.provider, config.num_threads),
            EngineConfig::SherpaOnnxKitten(config) => (&config.provider, config.num_threads),
            EngineConfig::Qwen3Tts(_) => unreachable!(),
        };
        args.push(Self::arg("output-filename", request.output));
        args.push(format!("--provider={provider}"));
        args.push(format!("--num-threads={threads}"));
        args.push(request.text.to_owned());
        Ok(args)
    }

    pub fn executable(&self) -> PathBuf {
        match &self.config {
            EngineConfig::SherpaOnnxVits(config) => config.executable.clone(),
            EngineConfig::SherpaOnnxPocket(config) => config.executable.clone(),
            EngineConfig::SherpaOnnxKokoro(config) => config.executable.clone(),
            EngineConfig::SherpaOnnxKitten(config) => config.executable.clone(),
            EngineConfig::Qwen3Tts(config) => config.python.clone(),
        }
    }
}

impl TtsEngine for SherpaOnnxEngine {
    fn synthesize(&self, request: &SynthesisRequest<'_>) -> Result<()> {
        if request.text.trim().is_empty() {
            bail!("text must not be empty");
        }
        if let Some(parent) = request.output.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let executable = self.executable();
        let status = Command::new(&executable)
            .args(self.command_args(request)?)
            .status()
            .with_context(|| format!("failed to start {}", executable.display()))?;
        if !status.success() {
            bail!("sherpa-onnx exited with {status}");
        }
        if !request.output.is_file() {
            bail!(
                "engine succeeded but did not create {}",
                request.output.display()
            );
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        match self.config {
            EngineConfig::SherpaOnnxVits(_) => "sherpa-onnx-vits/process-cold",
            EngineConfig::SherpaOnnxPocket(_) => "sherpa-onnx-pocket/process-cold",
            EngineConfig::SherpaOnnxKokoro(_) => "sherpa-onnx-kokoro/process-cold",
            EngineConfig::SherpaOnnxKitten(_) => "sherpa-onnx-kitten/process-cold",
            EngineConfig::Qwen3Tts(_) => "qwen3-tts/persistent-worker",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KittenConfig, KokoroConfig, PocketConfig, VitsConfig};

    #[test]
    fn builds_vits_arguments_without_shell_interpolation() {
        let engine = SherpaOnnxEngine::from_config(EngineConfig::SherpaOnnxVits(VitsConfig {
            executable: "sherpa-onnx-offline-tts".into(),
            model: "model.onnx".into(),
            tokens: "tokens.txt".into(),
            data_dir: Some("espeak data".into()),
            lexicon: None,
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 2,
        }));
        let request = SynthesisRequest {
            text: "hello; still text",
            output: Path::new("out file.wav"),
            reference_audio: None,
            speaking_pace: 1.25,
        };
        let args = engine.command_args(&request).unwrap();
        assert!(args.contains(&"--vits-data-dir=espeak data".to_owned()));
        assert!(args.contains(&"--vits-length-scale=0.8".to_owned()));
        assert_eq!(args.last().unwrap(), "hello; still text");
    }

    #[test]
    fn pocket_requires_reference_audio() {
        let config = PocketConfig {
            executable: "tts".into(),
            lm_flow: "flow".into(),
            lm_main: "main".into(),
            encoder: "encoder".into(),
            decoder: "decoder".into(),
            text_conditioner: "text".into(),
            vocab_json: "vocab".into(),
            token_scores_json: "scores".into(),
            provider: "cpu".into(),
            num_threads: 4,
            num_steps: 5,
        };
        let engine = SherpaOnnxEngine::from_config(EngineConfig::SherpaOnnxPocket(config));
        let request = SynthesisRequest {
            text: "hello",
            output: Path::new("out.wav"),
            reference_audio: None,
            speaking_pace: 1.0,
        };
        assert!(
            engine
                .command_args(&request)
                .unwrap_err()
                .to_string()
                .contains("voice profile")
        );
    }

    #[test]
    fn builds_kokoro_multivoice_arguments() {
        let engine = SherpaOnnxEngine::from_config(EngineConfig::SherpaOnnxKokoro(KokoroConfig {
            executable: "tts".into(),
            model: "model.int8.onnx".into(),
            voices: "voices.bin".into(),
            tokens: "tokens.txt".into(),
            data_dir: "espeak-ng-data".into(),
            lexicons: vec!["lexicon-us-en.txt".into(), "lexicon-zh.txt".into()],
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 3,
        }));
        let args = engine
            .command_args(&SynthesisRequest {
                text: "hello",
                output: Path::new("out.wav"),
                reference_audio: None,
                speaking_pace: 1.25,
            })
            .unwrap();
        assert!(args.contains(&"--sid=3".into()));
        assert!(args.contains(&"--kokoro-length-scale=0.8".into()));
        assert!(args.contains(&"--kokoro-lexicon=lexicon-us-en.txt,lexicon-zh.txt".into()));
    }

    #[test]
    fn builds_kitten_multivoice_arguments() {
        let engine = SherpaOnnxEngine::from_config(EngineConfig::SherpaOnnxKitten(KittenConfig {
            executable: "tts".into(),
            model: "model.int8.onnx".into(),
            voices: "voices.bin".into(),
            tokens: "tokens.txt".into(),
            data_dir: "espeak-ng-data".into(),
            provider: "cpu".into(),
            num_threads: 4,
            speaker_id: 1,
        }));
        let args = engine
            .command_args(&SynthesisRequest {
                text: "hello",
                output: Path::new("out.wav"),
                reference_audio: None,
                speaking_pace: 1.25,
            })
            .unwrap();
        assert!(args.contains(&"--kitten-model=model.int8.onnx".into()));
        assert!(args.contains(&"--sid=1".into()));
        assert!(args.contains(&"--kitten-length-scale=0.8".into()));
    }
}
