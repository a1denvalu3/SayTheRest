mod benchmark;
mod config;
mod engine;
mod resident;
mod text;
mod wav;

pub use benchmark::{BenchmarkReport, BenchmarkRunner};
pub use config::{EngineConfig, KokoroConfig, PocketConfig, VitsConfig};
pub use engine::{SherpaOnnxEngine, SynthesisRequest, TtsEngine};
pub use resident::ResidentSherpaEngine;
pub use text::{CleanedText, TextCleaningOptions, clean_text};
pub use wav::wav_duration_seconds;
