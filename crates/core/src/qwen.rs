use crate::{AudioSink, Qwen3TtsConfig, Qwen3TtsMode, QwenOnnxConfig, SynthesisRequest};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

pub struct ResidentSpeechWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    language: String,
    speaker: Option<String>,
    voice_description: Option<String>,
    device: String,
    dtype: String,
    backend: String,
    quantization: Option<String>,
    stderr_reader: Option<std::thread::JoinHandle<()>>,
}

/// Compatibility name for callers using the Qwen-specific constructors.
pub type ResidentQwenEngine = ResidentSpeechWorker;

#[derive(Deserialize)]
struct ReadyResponse {
    ready: bool,
    device: Option<String>,
    dtype: Option<String>,
    backend: Option<String>,
    quantization: Option<String>,
}

#[derive(Deserialize)]
struct SynthesisResponse {
    ok: Option<bool>,
    error: Option<String>,
    audio: Option<WorkerAudio>,
}

#[derive(Deserialize)]
struct WorkerAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

#[derive(Serialize)]
struct WorkerRequest<'a> {
    text: &'a str,
    output: String,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_description: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_audio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_text: Option<&'a str>,
    stream: bool,
    speaking_pace: f64,
}

impl ResidentSpeechWorker {
    pub fn load_phonon(config: &crate::PhononConfig) -> Result<Self> {
        let configuration = tempfile::NamedTempFile::new()?.into_temp_path();
        std::fs::write(
            &configuration,
            serde_json::to_vec(&crate::EngineConfig::Phonon(config.clone()))?,
        )?;
        let child = Command::new(&config.worker)
            .arg("--phonon-config")
            .arg(&configuration)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("failed to start native Phonon worker")?;
        Self::from_child(child, "en", Some(config.voice.clone()), None)
    }
    pub(crate) fn load_sherpa_with_worker(
        config: &crate::EngineConfig,
        environment: &[(std::ffi::OsString, std::ffi::OsString)],
        executable: &std::path::Path,
    ) -> Result<(Self, Arc<AtomicBool>)> {
        let configuration = tempfile::NamedTempFile::new()?.into_temp_path();
        std::fs::write(&configuration, serde_json::to_vec(config)?)?;
        let mut child = Command::new(executable)
            .arg("--sherpa-config")
            .arg(&configuration)
            .envs(environment.iter().cloned())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to start isolated speech worker")?;
        let stderr = child
            .stderr
            .take()
            .context("speech worker has no error stream")?;
        let fallback = Arc::new(AtomicBool::new(false));
        let detected = fallback.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else {
                    break;
                };
                if line == "SAYIT_SHERPA_READY" {
                    let _ = ready_tx.send(());
                    continue;
                }
                let lower = line.to_lowercase();
                if lower.contains("fallback to cpu") || lower.contains("falling back to cpu") {
                    detected.store(true, Ordering::Relaxed);
                }
                eprintln!("{line}");
            }
        });
        let mut engine = Self::from_child(child, "auto", None, None)?;
        engine.stderr_reader = Some(reader);
        // The marker follows all initialization logs on the same pipe. Waiting
        // for it prevents an unnoticed provider fallback before the first PCM.
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .context("speech worker initialization log did not complete")?;
        anyhow::ensure!(
            config.provider() != "cuda" || !fallback.load(Ordering::Relaxed),
            "CUDA execution provider fell back to CPU during initialization"
        );
        Ok((engine, fallback))
    }
    pub fn load(config: &Qwen3TtsConfig) -> Result<Self> {
        let mode = match config.mode {
            Qwen3TtsMode::CustomVoice => "custom-voice",
            Qwen3TtsMode::VoiceDesign => "voice-design",
            Qwen3TtsMode::VoiceClone => "voice-clone",
        };
        let child = Command::new(&config.python)
            .arg(&config.worker)
            .arg("--model")
            .arg(&config.model)
            .arg("--mode")
            .arg(mode)
            .arg("--device")
            .arg(&config.device)
            .arg("--dtype")
            .arg(&config.dtype)
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to start {}", config.python.display()))?;
        Self::from_child(
            child,
            &config.language,
            config.speaker.clone(),
            config.voice_description.clone(),
        )
    }

    pub fn load_native(config: &QwenOnnxConfig) -> Result<Self> {
        let runtime = config
            .runtime
            .as_ref()
            .context("native ONNX runtime was not prepared")?;
        let mut command = Command::new(&config.worker);
        command
            .arg("--model")
            .arg(&config.model)
            .arg("--runtime")
            .arg(runtime)
            .arg("--precision")
            .arg(&config.precision)
            .arg("--device")
            .arg(&config.device)
            .arg("--threads")
            .arg(config.num_threads.to_string());
        let key = if cfg!(windows) {
            "PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        let mut directories = vec![runtime.parent().context("invalid runtime path")?.to_owned()];
        if let Some(existing) = std::env::var_os(key) {
            directories.extend(std::env::split_paths(&existing));
        }
        command.env(key, std::env::join_paths(directories)?);
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| {
                format!(
                    "cannot start {}; build/install sayit-inference-worker",
                    config.worker.display()
                )
            })?;
        Self::from_child(child, &config.language, None, None)
    }

    fn from_child(
        mut child: Child,
        language: &str,
        speaker: Option<String>,
        voice_description: Option<String>,
    ) -> Result<Self> {
        let stdin = child.stdin.take().context("speech worker has no stdin")?;
        let stdout = BufReader::new(child.stdout.take().context("speech worker has no stdout")?);
        // Own the child before reading its startup response so every failure
        // below runs Drop and cannot leave an orphan allocating model memory.
        let mut engine = Self {
            child,
            stdin,
            stdout,
            language: language.into(),
            speaker,
            voice_description,
            device: "cpu".into(),
            dtype: "float32".into(),
            backend: "qwen3-tts-pytorch".into(),
            quantization: None,
            stderr_reader: None,
        };
        let mut line = String::new();
        anyhow::ensure!(
            engine.stdout.read_line(&mut line)? > 0,
            "speech worker exited before loading the model"
        );
        let ready: ReadyResponse = serde_json::from_str(&line)
            .context("speech worker returned an invalid ready response")?;
        anyhow::ensure!(ready.ready, "speech worker did not become ready");
        if let Some(device) = ready.device {
            engine.device = device;
        }
        if let Some(dtype) = ready.dtype {
            engine.dtype = dtype;
        }
        if let Some(backend) = ready.backend {
            engine.backend = backend;
        }
        engine.quantization = ready.quantization;
        Ok(engine)
    }

    pub fn device(&self) -> &str {
        &self.device
    }
    pub fn dtype(&self) -> &str {
        &self.dtype
    }
    pub fn backend(&self) -> &str {
        &self.backend
    }
    pub fn quantization(&self) -> Option<&str> {
        self.quantization.as_deref()
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<()> {
        self.synthesize_cancellable(request, None)
    }

    pub fn synthesize_cancellable(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        self.run_synthesis(request, cancelled, None)
    }

    pub fn synthesize_streaming(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
        sink: &mut AudioSink<'_>,
    ) -> Result<()> {
        self.run_synthesis(request, cancelled, Some(sink))
    }

    fn run_synthesis(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
        mut sink: Option<&mut AudioSink<'_>>,
    ) -> Result<()> {
        if request.text.trim().is_empty() {
            bail!("text must not be empty");
        }
        if let Some(parent) = request.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = WorkerRequest {
            text: request.text,
            output: request.output.to_string_lossy().into_owned(),
            language: &self.language,
            speaker: self.speaker.as_deref(),
            voice_description: self.voice_description.as_deref(),
            reference_audio: request
                .reference_audio
                .map(|path| path.to_string_lossy().into_owned()),
            reference_text: request.reference_text,
            stream: sink.is_some(),
            speaking_pace: request.speaking_pace,
        };
        serde_json::to_writer(&mut self.stdin, &payload)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        std::thread::scope(|scope| -> Result<()> {
            let (tx, rx) = mpsc::sync_channel(1);
            let stdout = &mut self.stdout;
            let reader = scope.spawn(move || {
                loop {
                    let mut line = String::new();
                    let result = stdout.read_line(&mut line).map(|count| (count, line));
                    let terminal = match &result {
                        Ok((count, line)) => {
                            *count == 0
                                || serde_json::from_str::<serde_json::Value>(line)
                                    .map_or(true, |value| value.get("ok").is_some())
                        }
                        Err(_) => true,
                    };
                    if tx.send(result).is_err() || terminal {
                        break;
                    }
                }
            });
            let result = (|| -> Result<()> {
                loop {
                    anyhow::ensure!(
                        cancelled.is_none_or(|flag| !flag.load(Ordering::Relaxed)),
                        "speech generation cancelled"
                    );
                    let (count, line) = match rx.recv_timeout(std::time::Duration::from_millis(25))
                    {
                        Ok(result) => result?,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(_) => bail!("speech worker exited during synthesis"),
                    };
                    anyhow::ensure!(count > 0, "speech worker exited during synthesis");
                    let response: SynthesisResponse = serde_json::from_str(&line)
                        .context("speech worker returned an invalid response")?;
                    if let Some(audio) = response.audio {
                        anyhow::ensure!(
                            (8_000..=192_000).contains(&audio.sample_rate)
                                && audio.samples.len() <= 65_536
                                && audio.samples.iter().all(|s| s.is_finite()),
                            "invalid streaming audio"
                        );
                        if let Some(sink) = sink.as_deref_mut() {
                            sink(&audio.samples, audio.sample_rate)?;
                        }
                    } else if response.ok == Some(true) {
                        return Ok(());
                    } else {
                        bail!(
                            "speech synthesis failed: {}",
                            response
                                .error
                                .as_deref()
                                .unwrap_or("invalid worker response")
                        );
                    }
                }
            })();
            if result.is_err() {
                let _ = self.child.kill();
            }
            drop(rx);
            reader
                .join()
                .map_err(|_| anyhow::anyhow!("speech response reader failed"))?;
            result
        })?;
        anyhow::ensure!(
            request.output.is_file(),
            "speech worker did not create output audio"
        );
        Ok(())
    }
}

impl Drop for ResidentSpeechWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.stderr_reader.take() {
            let _ = reader.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[cfg(target_os = "linux")]
    #[test]
    fn cancelling_inference_terminates_the_worker_promptly() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("worker.py");
        std::fs::write(&worker,"import sys,json,time\nprint(json.dumps({'ready':True}),flush=True)\nfor line in sys.stdin: time.sleep(60)\n").unwrap();
        let config = Qwen3TtsConfig {
            python: "/usr/bin/python3".into(),
            worker,
            model: directory.path().into(),
            mode: Qwen3TtsMode::VoiceClone,
            device: "cpu".into(),
            dtype: "float32".into(),
            language: "Auto".into(),
            speaker: None,
            voice_description: None,
        };
        let mut engine = ResidentQwenEngine::load(&config).unwrap();
        let cancelled = AtomicBool::new(false);
        let output = directory.path().join("out.wav");
        let start = std::time::Instant::now();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(100));
                cancelled.store(true, Ordering::Relaxed);
            });
            let error = engine
                .synthesize_cancellable(
                    &SynthesisRequest {
                        text: "hello",
                        output: &output,
                        reference_audio: None,
                        reference_text: None,
                        speaking_pace: 1.0,
                    },
                    Some(&cancelled),
                )
                .unwrap_err();
            assert!(error.to_string().contains("cancelled"));
        });
        assert!(start.elapsed() < std::time::Duration::from_secs(3));
        assert!(engine.child.wait().unwrap().code().is_none());
        assert!(!output.exists());
    }

    #[test]
    fn worker_request_never_serializes_absent_reference_audio() {
        let request = WorkerRequest {
            stream: false,
            speaking_pace: 1.0,
            text: "hello",
            output: "out.wav".into(),
            language: "English",
            speaker: Some("Aiden"),
            voice_description: None,
            reference_audio: None,
            reference_text: None,
        };
        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["speaker"], "Aiden");
        assert!(value.get("reference_audio").is_none());
        assert!(value.get("reference_text").is_none());
    }

    #[test]
    fn mode_and_paths_are_held_by_the_validated_config() {
        let config = Qwen3TtsConfig {
            python: PathBuf::from("python"),
            worker: PathBuf::from("worker.py"),
            model: PathBuf::from("model"),
            mode: Qwen3TtsMode::VoiceDesign,
            device: "cpu".into(),
            dtype: "float32".into(),
            language: "English".into(),
            speaker: None,
            voice_description: Some("A calm documentary narrator".into()),
        };
        assert!(matches!(config.mode, Qwen3TtsMode::VoiceDesign));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failed_startup_kills_and_reaps_the_worker() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("bad-worker.py");
        let pid_file = temporary.path().join("pid");
        std::fs::write(
            &worker,
            format!(
                "import os, pathlib, time\npathlib.Path({:?}).write_text(str(os.getpid()))\nprint('unexpected library output', flush=True)\ntime.sleep(30)\n",
                pid_file.to_string_lossy()
            ),
        )
        .unwrap();
        let config = Qwen3TtsConfig {
            python: "/usr/bin/python3".into(),
            worker,
            model: temporary.path().into(),
            mode: Qwen3TtsMode::VoiceClone,
            device: "cpu".into(),
            dtype: "float32".into(),
            language: "English".into(),
            speaker: None,
            voice_description: None,
        };
        assert!(ResidentQwenEngine::load(&config).is_err());
        let pid = std::fs::read_to_string(pid_file).unwrap();
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn persistent_worker_handles_multiple_requests_without_reloading() {
        let temporary = tempfile::tempdir().unwrap();
        let worker = temporary.path().join("fake-worker.py");
        std::fs::write(
            &worker,
            r#"import json, pathlib, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    pathlib.Path(request["output"]).write_bytes(b"RIFFfake")
    print(json.dumps({"ok": True}), flush=True)
"#,
        )
        .unwrap();
        let model = temporary.path().join("model");
        std::fs::create_dir(&model).unwrap();
        let config = Qwen3TtsConfig {
            python: "/usr/bin/python3".into(),
            worker,
            model,
            mode: Qwen3TtsMode::CustomVoice,
            device: "cpu".into(),
            dtype: "float32".into(),
            language: "English".into(),
            speaker: Some("Aiden".into()),
            voice_description: None,
        };
        let mut engine = ResidentQwenEngine::load(&config).unwrap();
        for name in ["first.wav", "second.wav"] {
            let output = temporary.path().join(name);
            engine
                .synthesize(&SynthesisRequest {
                    text: "hello",
                    output: &output,
                    reference_audio: None,
                    reference_text: None,
                    speaking_pace: 1.0,
                })
                .unwrap();
            assert_eq!(std::fs::read(output).unwrap(), b"RIFFfake");
        }
    }
}
