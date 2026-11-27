use anyhow::Result;
use clap::Parser;
use sayit_core::{
    EngineConfig, ResidentSherpaEngine, SynthesisRequest,
    pcm_codec::{FORMAT_PCM_F32LE_B64, write_audio_event},
    qwen_onnx::{NativeQwenPipeline, NativeRequest},
};
use std::{
    io::{self, BufRead, BufWriter, Write},
    path::PathBuf,
};

mod phonon;

#[derive(Parser)]
struct Options {
    #[arg(long, required_unless_present_any = ["sherpa_config", "phonon_config"])]
    model: Option<PathBuf>,
    #[arg(long, required_unless_present_any = ["sherpa_config", "phonon_config"])]
    runtime: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["model", "runtime", "phonon_config"])]
    sherpa_config: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["model", "runtime", "sherpa_config"])]
    phonon_config: Option<PathBuf>,
    #[arg(long, default_value = "int4")]
    precision: String,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Inference threads (`0` selects the CPU topology optimum automatically).
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

fn main() -> Result<()> {
    let options = Options::parse();
    if let Some(path) = options.phonon_config {
        return phonon::run(EngineConfig::from_path(&path)?);
    }
    if let Some(path) = options.sherpa_config {
        return run_sherpa(EngineConfig::from_path(&path)?);
    }
    let mut pipeline = NativeQwenPipeline::load(
        &options.model.unwrap(),
        &options.runtime.unwrap(),
        &options.precision,
        &options.device,
        options.threads,
    )?;
    let stdout = io::stdout();
    let mut output = BufWriter::with_capacity(64 * 1024, stdout.lock());
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "ready": true, "device": pipeline.device(), "dtype": "float32",
            "backend": "qwen3-tts-onnx", "quantization": (options.precision == "int4").then_some("int4"),
        })
    )?;
    output.flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = serde_json::from_str::<NativeRequest>(&line)
            .map_err(anyhow::Error::from)
            .and_then(|request| {
                let streaming = request.stream;
                let use_b64 = request.stream_format.as_deref() == Some(FORMAT_PCM_F32LE_B64);
                pipeline.synthesize_streaming(&request, &mut |samples, sample_rate| {
                    if streaming {
                        write_audio_event(&mut output, sample_rate, samples, use_b64)?;
                        output.flush()?;
                    }
                    Ok(())
                })
            });
        let response = match result {
            Ok(()) => serde_json::json!({"ok": true}),
            Err(error) => serde_json::json!({"ok": false, "error": format!("{error:#}")}),
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
    Ok(())
}

fn run_sherpa(config: EngineConfig) -> Result<()> {
    let mut engine = ResidentSherpaEngine::load(&config)?;
    eprintln!("SAYIT_SHERPA_READY");
    let stdout = io::stdout();
    let mut output = BufWriter::with_capacity(64 * 1024, stdout.lock());
    writeln!(
        output,
        "{}",
        serde_json::json!({"ready": true, "device": config.provider(), "backend": "sherpa-onnx"})
    )?;
    output.flush()?;
    for line in io::stdin().lock().lines() {
        let result = (|| -> Result<()> {
            let request: serde_json::Value = serde_json::from_str(&line?)?;
            let text = request["text"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing speech text"))?;
            let destination = request["output"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing output path"))?;
            let reference = request["reference_audio"].as_str().map(PathBuf::from);
            let streaming = request["stream"].as_bool().unwrap_or(false);
            let use_b64 = request["stream_format"].as_str() == Some(FORMAT_PCM_F32LE_B64);
            engine.synthesize_streaming(
                &SynthesisRequest {
                    generation: None,
                    text,
                    output: std::path::Path::new(destination),
                    reference_audio: reference.as_deref(),
                    reference_text: request["reference_text"].as_str(),
                    speaking_pace: request["speaking_pace"].as_f64().unwrap_or(1.0),
                },
                None,
                &mut |samples, sample_rate| {
                    if streaming {
                        write_audio_event(&mut output, sample_rate, samples, use_b64)?;
                        output.flush()?;
                    }
                    Ok(())
                },
            )
        })();
        let response = match result {
            Ok(()) => serde_json::json!({"ok": true}),
            Err(error) => serde_json::json!({"ok": false, "error": format!("{error:#}")}),
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
    Ok(())
}
