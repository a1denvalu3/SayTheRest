mod benchmark;
mod config;
mod device;
mod engine;
mod inference;
pub mod onnx_runtime;
mod qwen;
pub mod qwen_onnx;
pub mod qwen_runtime;
mod resident;
mod text;
mod wav;

pub use benchmark::{BenchmarkReport, BenchmarkRunner};
pub use config::{
    EngineConfig, KittenConfig, KokoroConfig, PhononConfig, PocketConfig, Qwen3TtsConfig,
    Qwen3TtsMode, QwenOnnxConfig, VitsConfig, native_worker_path,
};
pub use device::{CudaDevice, TorchFlavor};
pub use engine::{AudioSink, SherpaOnnxEngine, SynthesisRequest, TtsEngine};
pub use inference::{InferenceDevice, InferenceRuntime, ManagedEngine};
pub use qwen::{ResidentQwenEngine, ResidentSpeechWorker};
pub use resident::ResidentSherpaEngine;
pub use text::{CleanedText, TextCleaningOptions, clean_text};
pub use wav::wav_duration_seconds;
