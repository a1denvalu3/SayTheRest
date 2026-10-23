//! Native CPU/Q8 Phonon. Model loading and synthesis stay in this process so
//! cancellation can terminate inference without blocking the service.
use anyhow::{Context, Result, ensure};
use ptts::{
    checkpoint::{Checkpoint, ResolveOptions},
    preprocess::Lang,
    synth::{DeviceKind, Quant, SpeechOptions, Synth},
};
use sayit_core::EngineConfig;
use sha2::{Digest, Sha256};
use std::{
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

pub fn run(config: EngineConfig) -> Result<()> {
    let EngineConfig::Phonon(config) = config else {
        anyhow::bail!("Phonon worker requires a Phonon configuration");
    };
    let checkpoint = Checkpoint::resolve(
        &config.model,
        ResolveOptions {
            quant: Quant::Q80,
            weights: None,
        },
    )?;
    ensure!(
        checkpoint.config.languages.iter().any(|lang| lang == "en"),
        "Phonon model does not support English"
    );
    let mut synth = checkpoint
        .builder(Lang::En)
        .device(DeviceKind::Cpu)
        .build()?;
    // Load the selected voice strictly: an unreadable optional voice must never
    // silently become unconditioned speech or a different voice.
    let selected_voice = config.voice_path()?;
    sayit_core::phonon::read_voice(&selected_voice)?;
    synth.add_voice_file(&config.voice, &selected_voice)?;
    let options = SpeechOptions::default().voice(config.voice.clone());
    let mut custom_voice: Option<(PathBuf, Vec<u8>)> = None;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    writeln!(
        output,
        "{}",
        serde_json::json!({
            "ready":true, "device":"cpu", "dtype":"float32", "backend":"phonon", "quantization":"q8_0"
        })
    )?;
    output.flush()?;
    for line in io::stdin().lock().lines() {
        let result = (|| -> Result<()> {
            let request: serde_json::Value = serde_json::from_str(&line?)?;
            let text = request["text"].as_str().context("missing speech text")?;
            ensure!(!text.trim().is_empty(), "text must not be empty");
            let request_options = if request["reference_audio"].is_null() {
                options.clone()
            } else {
                let path = PathBuf::from(
                    request["reference_audio"]
                        .as_str()
                        .context("invalid voice file path")?,
                );
                let bytes = sayit_core::phonon::read_voice(&path)?;
                let fingerprint = Sha256::digest(&bytes).to_vec();
                let key = (path.clone(), fingerprint);
                if custom_voice.as_ref() != Some(&key) {
                    synth.add_voice_file("__sayit_custom", &path)?;
                    custom_voice = Some(key);
                }
                SpeechOptions::default().voice("__sayit_custom")
            };
            let pace = request["speaking_pace"].as_f64().unwrap_or(1.0);
            ensure!(
                pace.is_finite() && (pace - 1.0).abs() < f64::EPSILON,
                "Phonon does not support native speaking pace; use playback speed"
            );
            let destination =
                PathBuf::from(request["output"].as_str().context("missing output path")?);
            let streaming = request["stream"].as_bool().unwrap_or(false);
            synthesize(
                &synth,
                &request_options,
                text,
                &destination,
                &mut |samples, rate| {
                    if streaming {
                        writeln!(
                            output,
                            "{}",
                            serde_json::json!({"audio":{"sample_rate":rate,"samples":samples}})
                        )?;
                        output.flush()?;
                    }
                    Ok(())
                },
            )
        })();
        let response = match result {
            Ok(()) => serde_json::json!({"ok":true}),
            Err(error) => serde_json::json!({"ok":false,"error":format!("{error:#}")}),
        };
        writeln!(output, "{response}")?;
        output.flush()?;
    }
    Ok(())
}

fn synthesize(
    synth: &Synth,
    options: &SpeechOptions,
    text: &str,
    destination: &Path,
    sink: &mut sayit_core::AudioSink<'_>,
) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    // The parent owns this temporary path and publishes only a successful WAV.
    // Avoid child-owned temporary files that could leak when this process is
    // terminated during synthesis.
    let rate = synth.sample_rate();
    let mut archive = hound::WavWriter::create(
        destination,
        hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    let mut frames = 0;
    for block in synth.stream_with(text, options)? {
        let block = block?;
        ensure!(
            block.iter().all(|sample| sample.is_finite()),
            "Phonon returned non-finite audio"
        );
        for sample in &block {
            archive.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
        frames += block.len();
        sink(&block, rate)?;
    }
    ensure!(frames > 0, "Phonon returned empty audio");
    archive.finalize()?;
    Ok(())
}
