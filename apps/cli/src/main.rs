use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use reqwest::blocking::Client;
use sayit_core::{BenchmarkRunner, EngineConfig, SherpaOnnxEngine, SynthesisRequest, TtsEngine};
use sayit_protocol::{QueuePolicy, SpeechSource, SpeechSubmission};
use std::{
    io::{self, Read},
    path::PathBuf,
    sync::OnceLock,
};

static API_TOKEN: OnceLock<String> = OnceLock::new();

#[derive(Parser)]
#[command(version, about = "Private, local text-to-speech for Linux and Windows")]
struct Cli {
    #[arg(long, global = true, default_value = "http://127.0.0.1:55391/v1")]
    service_url: String,
    /// Override the per-user service token (normally discovered automatically).
    #[arg(long, global = true)]
    token: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Speak text through the persistent background service.
    Speak {
        text: Option<String>,
        #[arg(long)]
        voice_profile_id: Option<uuid::Uuid>,
        #[arg(long, value_enum, default_value_t = CliQueuePolicy::Replace)]
        queue: CliQueuePolicy,
        /// Confirm processing when text exceeds the configured long-text threshold.
        #[arg(long)]
        confirm_long_text: bool,
        /// Native generation pace (0.75, 0.9, 1, 1.1, 1.25, or 1.5).
        #[arg(long)]
        pace: Option<f64>,
    },
    /// Show service and playback state.
    Status,
    /// List current and recent jobs.
    Jobs,
    /// List, install, select, cancel, or remove models.
    Models {
        #[command(subcommand)]
        action: Option<ModelAction>,
    },
    /// List or manage locally imported voice profiles.
    Voices {
        #[command(subcommand)]
        action: Option<VoiceAction>,
    },
    /// List or manage persisted speech history.
    History {
        #[command(subcommand)]
        action: Option<HistoryAction>,
    },
    /// Pause current playback.
    Pause,
    /// Resume current playback.
    Resume,
    /// Stop current playback.
    Stop,
    /// Clear active speech and continue with queued work.
    Clear,
    /// Seek to an absolute position in seconds.
    Seek { seconds: f64 },
    /// Skip forward or backward by a signed number of seconds.
    Skip {
        #[arg(allow_hyphen_values = true)]
        seconds: f64,
    },
    /// Change playback speed (0.5 to 3.0).
    Rate { rate: f64 },
    /// Change playback volume (0.0 to 1.0).
    Volume { volume: f64 },
    /// Generate a WAV directly, bypassing the service (diagnostics).
    Synth {
        text: String,
        #[arg(short, long, default_value = "sayit.wav")]
        output: PathBuf,
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Native synthesis pace for supported models.
        #[arg(long, default_value_t = 1.0)]
        pace: f64,
    },
    /// Benchmark direct cold-process inference (diagnostics).
    Benchmark {
        #[arg(default_value = "The quick brown fox jumps over the lazy dog.")]
        text: String,
        #[arg(short, long, default_value_t = 5)]
        iterations: usize,
        #[arg(short, long)]
        config: Option<PathBuf>,
        #[arg(short, long, default_value = "output/benchmark.wav")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum CliQueuePolicy {
    Replace,
    Append,
    Interrupt,
}

#[derive(Subcommand)]
enum ModelAction {
    Install {
        id: String,
    },
    Update {
        id: String,
    },
    Select {
        id: String,
    },
    Voice {
        id: String,
        voice: String,
    },
    Unload {
        id: String,
    },
    Cancel {
        id: String,
    },
    Remove {
        id: String,
    },
    Benchmark {
        id: String,
    },
    ImportLocal {
        directory: PathBuf,
        display_name: String,
        #[arg(long)]
        license: String,
        #[arg(long)]
        license_url: String,
        /// Confirm the model is untested and its source license was reviewed.
        #[arg(long)]
        acknowledged: bool,
    },
    ImportHuggingFace {
        repository: String,
        display_name: String,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long)]
        license: String,
        #[arg(long)]
        license_url: String,
        #[arg(long)]
        access_token: Option<String>,
        /// Confirm the model is untested and its model-card license was reviewed.
        #[arg(long)]
        acknowledged: bool,
    },
}

#[derive(Subcommand)]
enum VoiceAction {
    Clone {
        name: String,
        reference_wav: PathBuf,
        /// Confirm the recorded speaker granted permission for voice cloning.
        #[arg(long)]
        speaker_permission_confirmed: bool,
    },
    Select {
        id: String,
    },
    Preview {
        id: uuid::Uuid,
    },
    Rename {
        id: uuid::Uuid,
        name: String,
    },
    Tune {
        id: uuid::Uuid,
        #[arg(value_parser = clap::value_parser!(u32).range(3..=8))]
        refinement_steps: u32,
    },
    Remove {
        id: uuid::Uuid,
    },
}

#[derive(Subcommand)]
enum HistoryAction {
    Replay { id: uuid::Uuid },
    Regenerate { id: uuid::Uuid },
    Pin { id: uuid::Uuid },
    Unpin { id: uuid::Uuid },
    Remove { id: uuid::Uuid },
}

impl From<CliQueuePolicy> for QueuePolicy {
    fn from(value: CliQueuePolicy) -> Self {
        match value {
            CliQueuePolicy::Replace => Self::Replace,
            CliQueuePolicy::Append => Self::Append,
            CliQueuePolicy::Interrupt => Self::Interrupt,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse_from(normalized_args());
    let _ = API_TOKEN.set(cli.token.map(Ok).unwrap_or_else(load_api_token)?);
    let base = cli.service_url.trim_end_matches('/');
    match cli.command {
        Command::Speak {
            text,
            voice_profile_id,
            queue,
            confirm_long_text,
            pace,
        } => {
            let text = text.map(Ok).unwrap_or_else(read_stdin)?;
            let response = authorized(Client::new().post(format!("{base}/jobs")))?
                .json(&SpeechSubmission {
                    text,
                    voice_profile_id,
                    source: SpeechSource::Cli,
                    queue_policy: queue.into(),
                    confirmed_long_text: confirm_long_text,
                    speaking_pace: pace,
                })
                .send()
                .context("cannot connect to sayIt service")?;
            print_response(response)?;
        }
        Command::Status => get_and_print(base, "state")?,
        Command::Jobs => get_and_print(base, "jobs")?,
        Command::Models { action: None } => get_and_print(base, "models")?,
        Command::Models {
            action: Some(ModelAction::Install { id }),
        } => post_and_print(base, &format!("models/{id}/install"), None)?,
        Command::Models {
            action: Some(ModelAction::Update { id }),
        } => post_and_print(base, &format!("models/{id}/update"), None)?,
        Command::Models {
            action: Some(ModelAction::Select { id }),
        } => post_and_print(base, &format!("models/{id}/select"), None)?,
        Command::Models {
            action: Some(ModelAction::Voice { id, voice }),
        } => post_and_print(base, &format!("models/{id}/voices/{voice}/select"), None)?,
        Command::Models {
            action: Some(ModelAction::Unload { id }),
        } => post_and_print(base, &format!("models/{id}/unload"), None)?,
        Command::Models {
            action: Some(ModelAction::Cancel { id }),
        } => post_and_print(base, &format!("models/{id}/cancel"), None)?,
        Command::Models {
            action: Some(ModelAction::Remove { id }),
        } => delete_and_print(base, &format!("models/{id}"))?,
        Command::Models {
            action: Some(ModelAction::Benchmark { id }),
        } => post_and_print(base, &format!("models/{id}/benchmark"), None)?,
        Command::Models {
            action:
                Some(ModelAction::ImportLocal {
                    directory,
                    display_name,
                    license,
                    license_url,
                    acknowledged,
                }),
        } => post_and_print(
            base,
            "models/imports/local",
            Some(serde_json::json!({
                "directory": directory,
                "display_name": display_name,
                "license": license,
                "license_url": license_url,
                "untested_model_and_license_review_confirmed": acknowledged
            })),
        )?,
        Command::Models {
            action:
                Some(ModelAction::ImportHuggingFace {
                    repository,
                    display_name,
                    revision,
                    license,
                    license_url,
                    access_token,
                    acknowledged,
                }),
        } => post_and_print(
            base,
            "models/imports/huggingface",
            Some(serde_json::json!({
                "repository": repository,
                "revision": revision,
                "display_name": display_name,
                "license": license,
                "license_url": license_url,
                "access_token": access_token,
                "untested_model_and_license_review_confirmed": acknowledged
            })),
        )?,
        Command::Voices { action: None } => get_and_print(base, "voices")?,
        Command::Voices {
            action:
                Some(VoiceAction::Clone {
                    name,
                    reference_wav,
                    speaker_permission_confirmed,
                }),
        } => post_and_print(
            base,
            "voices",
            Some(serde_json::json!({
                "name": name,
                "reference_audio_path": reference_wav,
                "speaker_permission_confirmed": speaker_permission_confirmed
            })),
        )?,
        Command::Voices {
            action: Some(VoiceAction::Select { id }),
        } => post_and_print(base, &format!("voices/{id}/select"), None)?,
        Command::Voices {
            action: Some(VoiceAction::Preview { id }),
        } => post_and_print(base, &format!("voices/{id}/preview"), None)?,
        Command::Voices {
            action: Some(VoiceAction::Rename { id, name }),
        } => post_and_print(
            base,
            &format!("voices/{id}/rename"),
            Some(serde_json::json!({"name": name})),
        )?,
        Command::Voices {
            action:
                Some(VoiceAction::Tune {
                    id,
                    refinement_steps,
                }),
        } => post_and_print(
            base,
            &format!("voices/{id}/tune"),
            Some(serde_json::json!({"refinement_steps": refinement_steps})),
        )?,
        Command::Voices {
            action: Some(VoiceAction::Remove { id }),
        } => delete_and_print(base, &format!("voices/{id}"))?,
        Command::History { action: None } => get_and_print(base, "history")?,
        Command::History {
            action: Some(HistoryAction::Replay { id }),
        } => post_and_print(base, &format!("history/{id}/replay"), None)?,
        Command::History {
            action: Some(HistoryAction::Regenerate { id }),
        } => post_and_print(base, &format!("history/{id}/regenerate"), None)?,
        Command::History {
            action: Some(HistoryAction::Pin { id }),
        } => post_and_print(
            base,
            &format!("history/{id}/pin"),
            Some(serde_json::json!({"pinned": true})),
        )?,
        Command::History {
            action: Some(HistoryAction::Unpin { id }),
        } => post_and_print(
            base,
            &format!("history/{id}/pin"),
            Some(serde_json::json!({"pinned": false})),
        )?,
        Command::History {
            action: Some(HistoryAction::Remove { id }),
        } => delete_and_print(base, &format!("history/{id}"))?,
        Command::Pause => post_and_print(base, "playback/pause", None)?,
        Command::Resume => post_and_print(base, "playback/resume", None)?,
        Command::Stop => post_and_print(base, "playback/stop", None)?,
        Command::Clear => post_and_print(base, "playback/stop", None)?,
        Command::Seek { seconds } => post_and_print(
            base,
            "playback/seek",
            Some(serde_json::json!({ "seconds": seconds })),
        )?,
        Command::Skip { seconds } => {
            let state = get_json(base, "state")?;
            let position = state
                .pointer("/playback/position_seconds")
                .and_then(serde_json::Value::as_f64)
                .context("service state did not include a playback position")?;
            post_and_print(
                base,
                "playback/seek",
                Some(serde_json::json!({ "seconds": (position + seconds).max(0.0) })),
            )?;
        }
        Command::Rate { rate } => post_and_print(
            base,
            "playback/rate",
            Some(serde_json::json!({ "rate": rate })),
        )?,
        Command::Volume { volume } => post_and_print(
            base,
            "playback/volume",
            Some(serde_json::json!({ "volume": volume })),
        )?,
        Command::Synth {
            text,
            output,
            config,
            pace,
        } => {
            let engine = load_engine(config)?;
            engine.synthesize(&SynthesisRequest {
                text: &text,
                output: &output,
                reference_audio: None,
                speaking_pace: pace,
            })?;
            println!("{}", output.display());
        }
        Command::Benchmark {
            text,
            iterations,
            config,
            output,
        } => {
            let engine = load_engine(config)?;
            let report = BenchmarkRunner::run(&engine, &text, &output, iterations)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}

fn normalized_args() -> Vec<String> {
    normalize_args(std::env::args().collect())
}

fn normalize_args(mut args: Vec<String>) -> Vec<String> {
    let known = [
        "speak",
        "status",
        "jobs",
        "models",
        "history",
        "voices",
        "pause",
        "resume",
        "stop",
        "clear",
        "seek",
        "skip",
        "rate",
        "volume",
        "synth",
        "benchmark",
        "help",
    ];
    let mut index = 1;
    while index < args.len() {
        if args[index] == "--service-url" || args[index] == "--token" {
            index += 2;
        } else if args[index].starts_with("--service-url=")
            || args[index].starts_with("--token=")
            || args[index].starts_with('-')
        {
            index += 1;
        } else {
            if !known.contains(&args[index].as_str()) {
                args.insert(index, "speak".into());
            }
            break;
        }
    }
    args
}

fn read_stdin() -> Result<String> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    if text.trim().is_empty() {
        bail!("provide text as an argument or on standard input");
    }
    Ok(text)
}

fn get_and_print(base: &str, resource: &str) -> Result<()> {
    let response = authorized(Client::new().get(format!("{base}/{resource}")))?
        .send()
        .context("cannot connect to sayIt service")?;
    print_response(response)
}

fn get_json(base: &str, resource: &str) -> Result<serde_json::Value> {
    let response = authorized(Client::new().get(format!("{base}/{resource}")))?
        .send()
        .context("cannot connect to sayIt service")?;
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        bail!("service returned {status}: {body}");
    }
    serde_json::from_str(&body).context("service returned invalid JSON")
}

fn post_and_print(base: &str, resource: &str, body: Option<serde_json::Value>) -> Result<()> {
    let request = authorized(Client::new().post(format!("{base}/{resource}")))?;
    let response = match body {
        Some(body) => request.json(&body),
        None => request,
    }
    .send()
    .context("cannot connect to sayIt service")?;
    let status = response.status();
    let text = response.text()?;
    if !status.is_success() {
        bail!("service returned {status}: {text}");
    }
    if !text.is_empty() {
        println!("{text}");
    }
    Ok(())
}

fn delete_and_print(base: &str, resource: &str) -> Result<()> {
    let response = authorized(Client::new().delete(format!("{base}/{resource}")))?
        .send()
        .context("cannot connect to sayIt service")?;
    let status = response.status();
    let text = response.text()?;
    if !status.is_success() {
        bail!("service returned {status}: {text}");
    }
    if !text.is_empty() {
        println!("{text}");
    }
    Ok(())
}

fn print_response(response: reqwest::blocking::Response) -> Result<()> {
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        bail!("service returned {status}: {body}");
    }
    let json: serde_json::Value = serde_json::from_str(&body)?;
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

fn load_engine(path: Option<PathBuf>) -> Result<SherpaOnnxEngine> {
    let path = match path {
        Some(path) => path,
        None if PathBuf::from("sayit.json").is_file() => PathBuf::from("sayit.json"),
        None if PathBuf::from("say-the-rest.json").is_file() => PathBuf::from("say-the-rest.json"),
        None => std::env::current_exe()?
            .parent()
            .and_then(|parent| {
                ["sayit.json", "say-the-rest.json"]
                    .into_iter()
                    .map(|name| parent.join(name))
                    .find(|candidate| candidate.is_file())
            })
            .ok_or_else(|| anyhow::anyhow!("cannot determine application directory"))?,
    };
    Ok(SherpaOnnxEngine::from_config(EngineConfig::from_path(
        &path,
    )?))
}

fn authorized(
    request: reqwest::blocking::RequestBuilder,
) -> Result<reqwest::blocking::RequestBuilder> {
    let token = API_TOKEN
        .get()
        .context("service token was not initialized")?;
    Ok(request.bearer_auth(token))
}

fn load_api_token() -> Result<String> {
    if let Ok(token) = std::env::var("SAYIT_TOKEN").or_else(|_| std::env::var("SAY_THE_REST_TOKEN"))
    {
        return Ok(token);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir)
                    .join(".local/state")
            })
    };
    let current = base.join("sayit");
    let legacy = base.join(if cfg!(windows) {
        "SayTheRest"
    } else {
        "say-the-rest"
    });
    let data_dir = if current.exists() || !legacy.exists() {
        current
    } else {
        legacy
    };
    std::fs::read_to_string(data_dir.join("api-token"))
        .map(|token| token.trim().to_owned())
        .context("cannot read the local service token; start the service first or pass --token")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_every_parity_command_family() {
        let cases: &[&[&str]] = &[
            &["sayit", "speak", "hello"],
            &["sayit", "status"],
            &["sayit", "jobs"],
            &["sayit", "models"],
            &["sayit", "models", "install", "piper-en-us-lessac-medium"],
            &["sayit", "voices"],
            &["sayit", "history"],
            &["sayit", "pause"],
            &["sayit", "resume"],
            &["sayit", "stop"],
            &["sayit", "clear"],
            &["sayit", "seek", "12.5"],
            &["sayit", "skip", "-15"],
            &["sayit", "rate", "1.25"],
            &["sayit", "volume", "0.8"],
        ];
        for arguments in cases {
            Cli::try_parse_from(*arguments)
                .unwrap_or_else(|error| panic!("failed to parse {arguments:?}: {error}"));
        }
    }

    #[test]
    fn bare_text_is_normalized_to_the_default_speak_command() {
        let arguments = vec!["sayit".into(), "Read this".into()];
        let cli = Cli::try_parse_from(normalize_args(arguments)).unwrap();
        assert!(matches!(cli.command, Command::Speak { .. }));
    }
}
