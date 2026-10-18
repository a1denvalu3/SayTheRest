use crate::{SynthesisRequest, TtsEngine, wav_duration_seconds};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkReport {
    pub engine: String,
    pub iterations: usize,
    pub cold_start_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_rtf: f64,
    pub audio_seconds: f64,
}

pub struct BenchmarkRunner;

impl BenchmarkRunner {
    pub fn run(
        engine: &dyn TtsEngine,
        text: &str,
        output: &Path,
        iterations: usize,
    ) -> Result<BenchmarkReport> {
        Self::run_with_reference(engine, text, output, iterations, None, None)
    }

    pub fn run_with_reference(
        engine: &dyn TtsEngine,
        text: &str,
        output: &Path,
        iterations: usize,
        reference_audio: Option<&Path>,
        reference_text: Option<&str>,
    ) -> Result<BenchmarkReport> {
        anyhow::ensure!(iterations > 0, "iterations must be at least 1");
        let mut samples = Vec::with_capacity(iterations);
        let mut rtfs = Vec::with_capacity(iterations);
        let mut audio_seconds = 0.0;
        for _ in 0..iterations {
            let started = Instant::now();
            engine.synthesize(&SynthesisRequest {
                text,
                output,
                reference_audio,
                reference_text,
                speaking_pace: 1.0,
            })?;
            let elapsed = started.elapsed().as_secs_f64();
            audio_seconds = wav_duration_seconds(output)?;
            anyhow::ensure!(audio_seconds > 0.0, "engine generated empty audio");
            samples.push(elapsed * 1000.0);
            rtfs.push(elapsed / audio_seconds);
        }
        let cold_start_ms = samples[0] + engine.startup_ms();
        samples.sort_by(f64::total_cmp);
        Ok(BenchmarkReport {
            engine: engine.name().to_owned(),
            iterations,
            cold_start_ms,
            p50_ms: percentile(&samples, 0.50),
            p95_ms: percentile(&samples, 0.95),
            p99_ms: percentile(&samples, 0.99),
            mean_rtf: rtfs.iter().sum::<f64>() / rtfs.len() as f64,
            audio_seconds,
        })
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ResidentTestEngine;
    impl TtsEngine for ResidentTestEngine {
        fn startup_ms(&self) -> f64 {
            5000.0
        }
        fn name(&self) -> &'static str {
            "test"
        }
        fn synthesize(&self, request: &SynthesisRequest<'_>) -> Result<()> {
            let mut writer = hound::WavWriter::create(
                request.output,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            for _ in 0..160 {
                writer.write_sample(1000i16)?;
            }
            writer.finalize()?;
            Ok(())
        }
    }

    #[test]
    fn resident_model_initialization_is_included_in_cold_start() {
        let directory = tempfile::tempdir().unwrap();
        let report = BenchmarkRunner::run(
            &ResidentTestEngine,
            "Hello",
            &directory.path().join("speech.wav"),
            3,
        )
        .unwrap();
        assert!(report.cold_start_ms >= 5000.0);
        assert_eq!(report.iterations, 3);
        assert_eq!(report.audio_seconds, 0.01);
    }
}
