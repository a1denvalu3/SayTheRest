use anyhow::Result;
use clap::Parser;
use sayit_core::qwen_onnx::{NativeQwenPipeline, NativeRequest};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

#[derive(Parser)]
struct Options {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long, default_value = "int4")]
    precision: String,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

fn main() -> Result<()> {
    let options = Options::parse();
    let mut pipeline = NativeQwenPipeline::load(
        &options.model,
        &options.runtime,
        &options.precision,
        &options.device,
        options.threads,
    )?;
    let stdout = io::stdout();
    let mut output = stdout.lock();
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
            .and_then(|request| pipeline.synthesize(&request));
        let response = match result {
            Ok(()) => serde_json::json!({"ok": true}),
            Err(error) => serde_json::json!({"ok": false, "error": format!("{error:#}")}),
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
    Ok(())
}
