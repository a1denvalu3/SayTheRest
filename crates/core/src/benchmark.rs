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
        Self::run_with_reference(engine, text, output, iterations, None)
    }

    pub fn run_with_reference(
        engine: &dyn TtsEngine,
        text: &str,
        output: &Path,
        iterations: usize,
        reference_audio: Option<&Path>,
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
                speaking_pace: 1.0,
            })?;
            let elapsed = started.elapsed().as_secs_f64();
            audio_seconds = wav_duration_seconds(output)?;
            anyhow::ensure!(audio_seconds > 0.0, "engine generated empty audio");
            samples.push(elapsed * 1000.0);
            rtfs.push(elapsed / audio_seconds);
        }
        let cold_start_ms = samples[0];
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
