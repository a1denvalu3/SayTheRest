use crate::{Qwen3TtsConfig, Qwen3TtsMode, SynthesisRequest};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

pub struct ResidentQwenEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    config: Qwen3TtsConfig,
}

#[derive(Deserialize)]
struct ReadyResponse {
    ready: bool,
}

#[derive(Deserialize)]
struct SynthesisResponse {
    ok: bool,
    error: Option<String>,
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
}

impl ResidentQwenEngine {
    pub fn load(config: &Qwen3TtsConfig) -> Result<Self> {
        let mode = match config.mode {
            Qwen3TtsMode::CustomVoice => "custom-voice",
            Qwen3TtsMode::VoiceDesign => "voice-design",
            Qwen3TtsMode::VoiceClone => "voice-clone",
        };
        let mut child = Command::new(&config.python)
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
        let stdin = child.stdin.take().context("Qwen worker has no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().context("Qwen worker has no stdout")?);
        let mut line = String::new();
        anyhow::ensure!(
            stdout.read_line(&mut line)? > 0,
            "Qwen worker exited before loading the model"
        );
        let ready: ReadyResponse = serde_json::from_str(&line)
            .context("Qwen worker returned an invalid ready response")?;
        anyhow::ensure!(ready.ready, "Qwen worker did not become ready");
        Ok(Self {
            child,
            stdin,
            stdout,
            config: config.clone(),
        })
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<()> {
        if request.text.trim().is_empty() {
            bail!("text must not be empty");
        }
        if let Some(parent) = request.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = WorkerRequest {
            text: request.text,
            output: request.output.to_string_lossy().into_owned(),
            language: &self.config.language,
            speaker: self.config.speaker.as_deref(),
            voice_description: self.config.voice_description.as_deref(),
            reference_audio: request
                .reference_audio
                .map(|path| path.to_string_lossy().into_owned()),
        };
        serde_json::to_writer(&mut self.stdin, &payload)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut line = String::new();
        anyhow::ensure!(
            self.stdout.read_line(&mut line)? > 0,
            "Qwen worker exited during synthesis"
        );
        let response: SynthesisResponse =
            serde_json::from_str(&line).context("Qwen worker returned an invalid response")?;
        if !response.ok {
            bail!(
                "Qwen synthesis failed: {}",
                response.error.as_deref().unwrap_or("unknown worker error")
            );
        }
        anyhow::ensure!(
            request.output.is_file(),
            "Qwen worker did not create output audio"
        );
        Ok(())
    }
}

impl Drop for ResidentQwenEngine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn worker_request_never_serializes_absent_reference_audio() {
        let request = WorkerRequest {
            text: "hello",
            output: "out.wav".into(),
            language: "English",
            speaker: Some("Aiden"),
            voice_description: None,
            reference_audio: None,
        };
        let value = serde_json::to_value(request).unwrap();
        assert_eq!(value["speaker"], "Aiden");
        assert!(value.get("reference_audio").is_none());
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
                    speaking_pace: 1.0,
                })
                .unwrap();
            assert_eq!(std::fs::read(output).unwrap(), b"RIFFfake");
        }
    }
}
