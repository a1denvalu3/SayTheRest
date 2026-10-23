use crate::EngineConfig;
use anyhow::{Context, Result, bail};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct QwenSamplingOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for QwenSamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
        }
    }
}

impl QwenSamplingOptions {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.temperature.is_finite() && (0.2..=1.2).contains(&self.temperature),
            "temperature must be between 0.2 and 1.2"
        );
        anyhow::ensure!(
            self.top_p.is_finite() && (0.5..=1.0).contains(&self.top_p),
            "top_p must be between 0.5 and 1"
        );
        anyhow::ensure!(self.top_k <= 100, "top_k must be between 0 and 100");
        anyhow::ensure!(
            self.repetition_penalty.is_finite() && (0.9..=1.5).contains(&self.repetition_penalty),
            "repetition_penalty must be between 0.9 and 1.5"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct QwenGenerationOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<usize>,
    #[serde(flatten)]
    pub sampling: QwenSamplingOptions,
}

impl QwenGenerationOptions {
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        anyhow::ensure!(
            self.max_new_tokens
                .is_none_or(|value| (1..=2048).contains(&value)),
            "generation limit must be between 1 and 2048"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SynthesisRequest<'a> {
    pub generation: Option<&'a QwenGenerationOptions>,
    pub text: &'a str,
    pub output: &'a Path,
    pub reference_audio: Option<&'a Path>,
    pub reference_text: Option<&'a str>,
    pub speaking_pace: f64,
}

/// Mono PCM delivered during synthesis, in source sample order. Returning an
/// error cancels generation. The sink runs on the inference thread.
pub type AudioSink<'a> = dyn FnMut(&[f32], u32) -> Result<()> + 'a;

pub trait TtsEngine {
    fn synthesize(&self, request: &SynthesisRequest<'_>) -> Result<()>;
    fn name(&self) -> &'static str;
    fn startup_ms(&self) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug)]
pub struct SherpaOnnxEngine {
    config: EngineConfig,
    environment: Vec<(OsString, OsString)>,
}

impl SherpaOnnxEngine {
    pub fn from_config(config: EngineConfig) -> Self {
        Self {
            config,
            environment: Vec::new(),
        }
    }

    pub fn with_environment(mut self, environment: Vec<(OsString, OsString)>) -> Self {
        self.environment = environment;
        self
    }

    fn arg(name: &str, path: &Path) -> String {
        format!("--{name}={}", path.display())
    }

    pub fn command_args(&self, request: &SynthesisRequest<'_>) -> Result<Vec<String>> {
        anyhow::ensure!(
            request.generation.is_none(),
            "generation tuning requires the Qwen speech worker"
        );
        if self.config.is_qwen() || matches!(self.config, EngineConfig::Phonon(_)) {
            bail!("This model requires the persistent speech worker");
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
                    .or(config.reference_audio.as_deref())
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
            EngineConfig::Qwen3Tts(_) | EngineConfig::Qwen3TtsOnnx(_) | EngineConfig::Phonon(_) => {
                unreachable!()
            }
        };
        let (provider, threads) = match &self.config {
            EngineConfig::SherpaOnnxVits(config) => (&config.provider, config.num_threads),
            EngineConfig::SherpaOnnxPocket(config) => (&config.provider, config.num_threads),
            EngineConfig::SherpaOnnxKokoro(config) => (&config.provider, config.num_threads),
            EngineConfig::SherpaOnnxKitten(config) => (&config.provider, config.num_threads),
            EngineConfig::Qwen3Tts(_) | EngineConfig::Qwen3TtsOnnx(_) | EngineConfig::Phonon(_) => {
                unreachable!()
            }
        };
        args.push(Self::arg("output-filename", request.output));
        args.push(format!("--provider={provider}"));
        args.push(format!("--num-threads={threads}"));
        args.push("--tts-silence-scale=1.0".into());
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
            EngineConfig::Qwen3TtsOnnx(config) => config.worker.clone(),
            EngineConfig::Phonon(config) => config.worker.clone(),
        }
    }

    pub(crate) fn synthesize_streaming(
        &self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        sink: &mut AudioSink<'_>,
    ) -> Result<()> {
        self.synthesize_streaming_with_worker(
            request,
            cancelled,
            sink,
            &crate::native_worker_path(),
        )
    }

    pub(crate) fn synthesize_streaming_with_worker(
        &self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        sink: &mut AudioSink<'_>,
        executable: &Path,
    ) -> Result<()> {
        // Isolate CUDA provider faults from the service, while retaining native
        // callbacks. The worker exits after this request; CPU engines stay resident.
        let parent = request
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let pending = tempfile::NamedTempFile::new_in(parent)?.into_temp_path();
        let bounded = SynthesisRequest {
            output: pending.as_ref(),
            ..request.clone()
        };
        let (mut worker, fallback) = crate::ResidentQwenEngine::load_sherpa_with_worker(
            &self.config,
            &self.environment,
            executable,
        )?;
        let result = worker.synthesize_streaming(&bounded, cancelled, sink);
        drop(worker); // Drain every provider log before reporting success.
        anyhow::ensure!(
            self.config.provider() != "cuda"
                || !fallback.load(std::sync::atomic::Ordering::Relaxed),
            "CUDA execution provider fell back to CPU during inference"
        );
        result?;
        anyhow::ensure!(
            crate::wav_duration_seconds(pending.as_ref())? > 0.0,
            "inference generated empty audio"
        );
        pending
            .persist(request.output)
            .map_err(|error| error.error)?;
        Ok(())
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
        let result = Command::new(&executable)
            .args(self.command_args(request)?)
            .envs(self.environment.iter().cloned())
            .output()
            .with_context(|| format!("failed to start {}", executable.display()))?;
        let stderr = String::from_utf8_lossy(&result.stderr);
        eprint!("{stderr}");
        if !result.status.success() {
            bail!(
                "sherpa-onnx exited with {}: {}",
                result.status,
                stderr.trim()
            );
        }
        if self.config.provider() == "cuda" && stderr.to_lowercase().contains("fallback to cpu") {
            bail!(
                "CUDA execution provider fell back to CPU: {}",
                stderr.trim()
            );
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
            EngineConfig::Phonon(_) => "phonon/persistent-worker",
            EngineConfig::SherpaOnnxVits(_) => "sherpa-onnx-vits/process-cold",
            EngineConfig::SherpaOnnxPocket(_) => "sherpa-onnx-pocket/process-cold",
            EngineConfig::SherpaOnnxKokoro(_) => "sherpa-onnx-kokoro/process-cold",
            EngineConfig::SherpaOnnxKitten(_) => "sherpa-onnx-kitten/process-cold",
            EngineConfig::Qwen3Tts(_) | EngineConfig::Qwen3TtsOnnx(_) => {
                "qwen3-tts/persistent-worker"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn successful_cuda_process_that_reports_cpu_fallback_is_not_advertised_as_gpu() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("fake-tts");
        std::fs::write(&executable, "#!/bin/sh\necho 'Available providers: CPUExecutionProvider. Fallback to cpu!' >&2\nexit 0\n").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let config: EngineConfig = serde_json::from_value(serde_json::json!({
            "engine": "sherpa-onnx-vits", "executable": executable, "model": "model", "tokens": "tokens", "provider": "cuda"
        })).unwrap();
        let output = directory.path().join("speech.wav");
        let error = SherpaOnnxEngine::from_config(config)
            .synthesize(&SynthesisRequest {
                generation: None,
                text: "Hello",
                output: &output,
                reference_audio: None,
                reference_text: None,
                speaking_pace: 1.0,
            })
            .unwrap_err();
        assert!(error.to_string().contains("fell back to CPU"));
    }
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
            generation: None,
            text: "hello; still text",
            output: Path::new("out file.wav"),
            reference_audio: None,
            reference_text: None,
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
            reference_audio: None,
            preset_voice: None,
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
            generation: None,
            text: "hello",
            output: Path::new("out.wav"),
            reference_audio: None,
            reference_text: None,
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
                generation: None,
                text: "hello",
                output: Path::new("out.wav"),
                reference_audio: None,
                reference_text: None,
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
                generation: None,
                text: "hello",
                output: Path::new("out.wav"),
                reference_audio: None,
                reference_text: None,
                speaking_pace: 1.25,
            })
            .unwrap();
        assert!(args.contains(&"--kitten-model=model.int8.onnx".into()));
        assert!(args.contains(&"--sid=1".into()));
        assert!(args.contains(&"--kitten-length-scale=0.8".into()));
    }
}
