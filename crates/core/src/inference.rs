use crate::qwen_runtime::QwenRuntimeManager;
use crate::{
    CudaDevice, EngineConfig, ResidentQwenEngine, ResidentSherpaEngine, SherpaOnnxEngine,
    SynthesisRequest, TorchFlavor, TtsEngine,
};
use anyhow::{Context, Result, bail};
use bzip2::read::BzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    ffi::OsString,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceDevice {
    pub provider: String,
    pub device: String,
    pub dtype: Option<String>,
    pub fallback_reason: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
}

#[derive(Clone)]
pub struct InferenceRuntime {
    root: PathBuf,
    qwen: QwenRuntimeManager,
    cuda: Option<CudaDevice>,
    setup_failure: Arc<Mutex<HashMap<&'static str, String>>>,
}

struct PreparedInference {
    config: EngineConfig,
    environment: Vec<(OsString, OsString)>,
    fallback_reason: Option<String>,
}

pub fn packaged_manifest_dir() -> PathBuf {
    let packaged = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|p| p.join("runtime/generative")));
    if let Some(path) = packaged
        && path.join("qwen_worker.py").is_file()
    {
        return path;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime")
}

impl InferenceRuntime {
    pub fn reset_failures(&self) {
        if let Ok(mut failure) = self.setup_failure.try_lock() {
            failure.clear();
        }
    }
    pub fn new(data_dir: &Path) -> Self {
        Self::with_manifest(data_dir, packaged_manifest_dir())
    }

    pub fn with_manifest(data_dir: &Path, manifest: PathBuf) -> Self {
        let root = data_dir.join("runtimes");
        Self {
            qwen: QwenRuntimeManager::new(root.join("qwen"), manifest),
            root,
            cuda: CudaDevice::detect(),
            setup_failure: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn for_current_user() -> Self {
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
        Self::new(&base.join("sayit"))
    }

    fn prepare(&self, config: &EngineConfig, prefer_gpu: bool) -> Result<PreparedInference> {
        if matches!(config, EngineConfig::Phonon(_)) {
            return Ok(PreparedInference {
                config: config.clone(),
                environment: Vec::new(),
                fallback_reason: prefer_gpu
                    .then(|| "Phonon Q8 uses the native CPU runtime.".into()),
            });
        }
        if let EngineConfig::Qwen3TtsOnnx(native) = config {
            let mut native = native.clone();
            let runtime =
                crate::onnx_runtime::NativeRuntimeManager::new(self.root.parent().unwrap());
            let gpu = prefer_gpu
                && self
                    .cuda
                    .is_some_and(|device| device.driver_version >= 12060);
            let mut reason = (prefer_gpu && self.cuda.is_some() && !gpu)
                .then(|| "NVIDIA driver does not support the packaged CUDA 12.6 libraries".into());
            native.device = if gpu { "cuda" } else { "cpu" }.into();
            if native.runtime.is_none() {
                native.runtime = Some(match runtime.install(gpu) {
                    Ok(path) => path,
                    Err(error) if gpu => {
                        reason = Some(format!("native CUDA runtime setup failed: {error:#}"));
                        native.device = "cpu".into();
                        runtime.install(false)?
                    }
                    Err(error) => return Err(error),
                });
            }
            return Ok(PreparedInference {
                config: EngineConfig::Qwen3TtsOnnx(native),
                environment: Vec::new(),
                fallback_reason: reason,
            });
        }
        let mut prepared = config.cpu();
        let managed_qwen =
            matches!(config, EngineConfig::Qwen3Tts(q) if self.qwen.owns_python(&q.python));
        if let EngineConfig::Qwen3Tts(qwen) = &mut prepared
            && !managed_qwen
        {
            if prefer_gpu {
                qwen.device = "auto".into();
                qwen.dtype = "auto".into();
            }
            return Ok(PreparedInference {
                config: prepared,
                environment: Vec::new(),
                fallback_reason: None,
            });
        }
        let flavor = prefer_gpu
            .then_some(self.cuda)
            .flatten()
            .and_then(CudaDevice::torch_flavor);
        let mut fallback = None;
        if let Some(flavor) = flavor {
            let family = if matches!(config, EngineConfig::Qwen3Tts(_)) {
                "qwen"
            } else {
                "onnx"
            };
            // ONNX on Linux only needs NVIDIA's runtime libraries. Avoid
            // downloading PyTorch and Qwen for Piper/Kokoro/Kitten/PocketTTS.
            // Windows PyTorch wheels bundle the matching CUDA DLLs.
            let libraries_flavor =
                if cfg!(target_os = "linux") && !matches!(config, EngineConfig::Qwen3Tts(_)) {
                    TorchFlavor::CudaLibraries
                } else {
                    flavor
                };
            let mut failure = self.setup_failure.lock().unwrap();
            let setup = if let Some(reason) = failure.get(family) {
                Err(anyhow::anyhow!(reason.clone()))
            } else {
                self.qwen.install_for(libraries_flavor)
            };
            match setup {
                Ok(()) => {
                    if let EngineConfig::Qwen3Tts(qwen) = &mut prepared {
                        if managed_qwen {
                            qwen.python = self.qwen.python_for(flavor);
                        }
                        qwen.device = "auto".into();
                        qwen.dtype = "auto".into();
                        return Ok(PreparedInference {
                            config: prepared,
                            environment: Vec::new(),
                            fallback_reason: None,
                        });
                    }
                    match self.install_sherpa_cuda().and_then(|executable| {
                        let environment = self.cuda_environment(libraries_flavor, &executable)?;
                        Ok((executable, environment))
                    }) {
                        Ok((executable, environment)) => {
                            prepared.set_sherpa_provider("cuda", Some(&executable));
                            return Ok(PreparedInference {
                                config: prepared,
                                environment,
                                fallback_reason: None,
                            });
                        }
                        Err(error) => {
                            fallback = Some(format!("GPU runtime setup failed: {error:#}"));
                        }
                    }
                }
                Err(error) => fallback = Some(format!("GPU runtime setup failed: {error:#}")),
            }
            if let Some(reason) = &fallback {
                failure.insert(family, reason.clone());
            }
        } else if prefer_gpu && self.cuda.is_some() {
            fallback = Some("The NVIDIA driver is too old for the packaged CUDA runtime (requires CUDA 12.6 or newer).".into());
        }
        if let EngineConfig::Qwen3Tts(qwen) = &mut prepared
            && managed_qwen
        {
            self.qwen.install_for(TorchFlavor::Cpu)?;
            qwen.python = self.qwen.python_for(TorchFlavor::Cpu);
        }
        Ok(PreparedInference {
            config: prepared,
            environment: Vec::new(),
            fallback_reason: fallback,
        })
    }

    fn cuda_environment(
        &self,
        flavor: TorchFlavor,
        executable: &Path,
    ) -> Result<Vec<(OsString, OsString)>> {
        let env = self.qwen.environment(flavor);
        let packages = env.join(if cfg!(windows) {
            "Lib/site-packages"
        } else {
            "lib/python3.12/site-packages"
        });
        let mut directories = vec![
            executable.parent().unwrap().parent().unwrap().join("lib"),
            packages.join("torch/lib"),
        ];
        if let Ok(entries) = fs::read_dir(packages.join("nvidia")) {
            for entry in entries {
                let path = entry?.path();
                for name in ["lib", "bin"] {
                    if path.join(name).is_dir() {
                        directories.push(path.join(name));
                    }
                }
            }
        }
        let key = if cfg!(windows) {
            "PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        if let Some(existing) = std::env::var_os(key) {
            directories.extend(std::env::split_paths(&existing));
        }
        Ok(vec![(key.into(), std::env::join_paths(directories)?)])
    }

    fn install_sherpa_cuda(&self) -> Result<PathBuf> {
        let (asset, digest) = if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            (
                "sherpa-onnx-v1.13.4-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2",
                "75bfb51f58a3f7e0e1be66f702ddb509463444f0f4b11a823f85039b6e3b9119",
            )
        } else if cfg!(all(windows, target_arch = "x86_64")) {
            (
                "sherpa-onnx-v1.13.4-cuda-12.x-cudnn-9.x-win-x64-cuda.tar.bz2",
                "11b56076060c109e16d85eba3acfb03f6d0c4a738ecd2a99dd45eb689b2051a4",
            )
        } else {
            bail!("No packaged CUDA ONNX runtime for this platform");
        };
        fs::create_dir_all(&self.root)?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join("sherpa-install.lock"))?;
        lock.lock()?;
        let destination = self.root.join("sherpa-cuda-1.13.4");
        if fs::read_to_string(destination.join("sha256")).is_ok_and(|value| value == digest) {
            return find_executable(&destination);
        }
        eprintln!("Preparing CUDA inference for Piper, Kokoro, Kitten, and PocketTTS");
        let staging = tempfile::tempdir_in(&self.root)?;
        let archive = staging.path().join("runtime.tar.bz2");
        let mut response = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(1200))
            .build()?
            .get(format!(
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/{asset}"
            ))
            .send()?
            .error_for_status()?;
        let mut output = fs::File::create(&archive)?;
        let mut hash = Sha256::new();
        let mut total = 0;
        let mut buffer = [0u8; 128 * 1024];
        loop {
            let count = response.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            total += count;
            anyhow::ensure!(
                total <= 400 * 1024 * 1024,
                "GPU runtime exceeds download limit"
            );
            output.write_all(&buffer[..count])?;
            hash.update(&buffer[..count]);
        }
        output.sync_all()?;
        anyhow::ensure!(
            format!("{:x}", hash.finalize()) == digest,
            "GPU runtime checksum mismatch"
        );
        let extracted = staging.path().join("extracted");
        fs::create_dir(&extracted)?;
        let mut tar = tar::Archive::new(BzDecoder::new(fs::File::open(archive)?));
        for entry in tar.entries()? {
            anyhow::ensure!(
                entry?.unpack_in(&extracted)?,
                "unsafe GPU runtime archive path"
            );
        }
        find_executable(&extracted)?;
        fs::write(extracted.join("sha256"), digest)?;
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(extracted, &destination)?;
        find_executable(&destination)
    }
}

fn find_executable(directory: &Path) -> Result<PathBuf> {
    let name = if cfg!(windows) {
        "sherpa-onnx-offline-tts.exe"
    } else {
        "sherpa-onnx-offline-tts"
    };
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Ok(path) = find_executable(&entry.path()) {
                return Ok(path);
            }
        } else if entry.file_name() == name {
            return Ok(entry.path());
        }
    }
    bail!("GPU runtime is missing {name}")
}

enum Backend {
    Sherpa(ResidentSherpaEngine),
    // Native provider initialization may abort instead of returning an error.
    // Isolate CUDA in the supplied executable so CPU fallback remains possible.
    SherpaCuda(SherpaOnnxEngine),
    Worker(ResidentQwenEngine),
}

impl Backend {
    fn load(
        config: &EngineConfig,
        environment: Vec<(OsString, OsString)>,
    ) -> Result<(Self, InferenceDevice)> {
        if config.is_qwen() || matches!(config, EngineConfig::Phonon(_)) {
            let engine = match config {
                EngineConfig::Qwen3Tts(qwen) => ResidentQwenEngine::load(qwen)?,
                EngineConfig::Qwen3TtsOnnx(qwen) => ResidentQwenEngine::load_native(qwen)?,
                EngineConfig::Phonon(phonon) => ResidentQwenEngine::load_phonon(phonon)?,
                _ => unreachable!(),
            };
            let device = engine.device().to_owned();
            let provider = device.split(':').next().unwrap_or("cpu").to_owned();
            let info = InferenceDevice {
                provider,
                device,
                dtype: Some(engine.dtype().to_owned()),
                fallback_reason: None,
                backend: Some(engine.backend().into()),
                quantization: engine.quantization().map(str::to_owned),
            };
            return Ok((Self::Worker(engine), info));
        }
        let provider = config.provider().to_owned();
        let info = InferenceDevice {
            device: provider.clone(),
            provider: provider.clone(),
            dtype: None,
            fallback_reason: None,
            backend: Some("sherpa-onnx".into()),
            quantization: None,
        };
        let backend = if provider == "cuda" {
            Self::SherpaCuda(
                SherpaOnnxEngine::from_config(config.clone()).with_environment(environment),
            )
        } else {
            Self::Sherpa(ResidentSherpaEngine::load(config)?)
        };
        Ok((backend, info))
    }

    fn synthesize_with_sink(
        &mut self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
        sink: &mut Option<&mut crate::AudioSink<'_>>,
        delivered: &mut bool,
    ) -> Result<()> {
        if let Some(sink) = sink.as_deref_mut() {
            let mut forward = |samples: &[f32], rate| {
                anyhow::ensure!(
                    cancelled.is_none_or(|flag| !flag.load(Ordering::Relaxed)),
                    "speech generation cancelled"
                );
                if !samples.is_empty() {
                    *delivered = true;
                    sink(samples, rate)?;
                }
                Ok(())
            };
            return match self {
                Self::Sherpa(engine) => {
                    engine.synthesize_streaming(request, cancelled, &mut forward)
                }
                Self::SherpaCuda(engine) => {
                    engine.synthesize_streaming(request, cancelled, &mut forward)
                }
                Self::Worker(engine) => {
                    engine.synthesize_streaming(request, cancelled, &mut forward)
                }
            };
        }
        match self {
            Self::Sherpa(engine) => engine.synthesize(request),
            Self::SherpaCuda(engine) => engine.synthesize(request),
            Self::Worker(engine) => engine.synthesize_cancellable(request, cancelled),
        }
    }
}

struct EngineState {
    backend: Option<Backend>,
    device: InferenceDevice,
}

pub struct ManagedEngine {
    state: Mutex<EngineState>,
    cpu_config: EngineConfig,
    startup_ms: f64,
}

impl ManagedEngine {
    pub fn load(
        config: &EngineConfig,
        runtime: &InferenceRuntime,
        prefer_gpu: bool,
    ) -> Result<Self> {
        let PreparedInference {
            config: prepared,
            environment,
            fallback_reason: fallback,
        } = runtime.prepare(config, prefer_gpu)?;
        let started = Instant::now();
        // A CUDA-enabled PyTorch environment can run CPU fallback too.
        let mut cpu_config = config.cpu();
        if let EngineConfig::Qwen3TtsOnnx(native) = &mut cpu_config
            && native.runtime.is_none()
        {
            native.runtime = Some(
                crate::onnx_runtime::NativeRuntimeManager::new(runtime.root.parent().unwrap())
                    .install(false)?,
            );
        }
        if let (EngineConfig::Qwen3Tts(cpu), EngineConfig::Qwen3Tts(selected)) =
            (&mut cpu_config, &prepared)
        {
            cpu.python = selected.python.clone();
        }
        let (backend, mut device) = match Backend::load(&prepared, environment) {
            Ok(loaded) => loaded,
            Err(error) if prepared.provider() != "cpu" => {
                let (backend, mut info) = Backend::load(&cpu_config, Vec::new())?;
                info.fallback_reason = Some(format!("GPU initialization failed: {error:#}"));
                (backend, info)
            }
            Err(error) => return Err(error),
        };
        if device.fallback_reason.is_none() {
            device.fallback_reason = fallback;
        }
        if prefer_gpu && device.provider == "cpu" && device.fallback_reason.is_none() {
            device.fallback_reason =
                Some("No compatible GPU is available in the selected runtime.".into());
        }
        eprintln!(
            "Inference device: {}{}",
            device.device,
            device
                .fallback_reason
                .as_ref()
                .map(|r| format!(" ({r})"))
                .unwrap_or_default()
        );
        Ok(Self {
            state: Mutex::new(EngineState {
                backend: Some(backend),
                device,
            }),
            cpu_config,
            startup_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn device(&self) -> InferenceDevice {
        self.state.lock().unwrap().device.clone()
    }
}

impl TtsEngine for ManagedEngine {
    fn synthesize(&self, request: &SynthesisRequest<'_>) -> Result<()> {
        self.synthesize_cancellable(request, None)
    }

    fn name(&self) -> &'static str {
        if matches!(self.cpu_config, EngineConfig::Phonon(_)) {
            "phonon"
        } else if matches!(self.cpu_config, EngineConfig::Qwen3TtsOnnx(_)) {
            "qwen3-tts-onnx"
        } else if self.cpu_config.is_qwen() {
            "qwen3-tts"
        } else {
            "sherpa-onnx"
        }
    }

    fn startup_ms(&self) -> f64 {
        self.startup_ms
    }
}

impl ManagedEngine {
    pub fn synthesize_cancellable(
        &self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        self.run_synthesis(request, cancelled, None)
    }

    pub fn synthesize_streaming(
        &self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
        sink: &mut crate::AudioSink<'_>,
    ) -> Result<()> {
        self.run_synthesis(request, cancelled, Some(sink))
    }

    fn run_synthesis(
        &self,
        request: &SynthesisRequest<'_>,
        cancelled: Option<&AtomicBool>,
        mut sink: Option<&mut crate::AudioSink<'_>>,
    ) -> Result<()> {
        anyhow::ensure!(
            cancelled.is_none_or(|cancelled| !cancelled.load(Ordering::Relaxed)),
            "speech generation cancelled"
        );
        anyhow::ensure!(!request.text.trim().is_empty(), "text must not be empty");
        anyhow::ensure!(
            request.generation.is_none() || self.cpu_config.is_qwen(),
            "generation tuning is supported only by Qwen models"
        );
        let parent = request
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        // Keep the path reserved, but close its handle before another process
        // writes/replaces it (especially important on Windows).
        let pending = tempfile::NamedTempFile::new_in(parent)?.into_temp_path();
        let bounded = SynthesisRequest {
            output: pending.as_ref(),
            ..request.clone()
        };
        let mut state = self.state.lock().unwrap();
        if state.backend.is_none() {
            let (backend, mut device) = Backend::load(&self.cpu_config, Vec::new())?;
            device.fallback_reason = state.device.fallback_reason.clone().or_else(|| {
                (state.device.provider != "cpu")
                    .then(|| "Previous inference failed; reloaded on CPU.".into())
            });
            state.device = device;
            state.backend = Some(backend);
        }
        let mut delivered = false;
        let result = state
            .backend
            .as_mut()
            .context("inference engine unavailable")?
            .synthesize_with_sink(&bounded, cancelled, &mut sink, &mut delivered)
            .and_then(|()| {
                anyhow::ensure!(
                    crate::wav_duration_seconds(pending.as_ref())? > 0.0,
                    "inference generated empty audio"
                );
                Ok(())
            });
        if let Err(error) = result {
            if delivered
                || state.device.provider == "cpu"
                || cancelled.is_some_and(|cancelled| cancelled.load(Ordering::Relaxed))
            {
                state.backend = None;
                return Err(error);
            }
            let reason = format!("GPU inference failed; using CPU: {error:#}");
            eprintln!("{reason}");
            // Drop GPU tensors before allocating the CPU model.
            state.backend = None;
            let (backend, mut device) = Backend::load(&self.cpu_config, Vec::new())?;
            device.fallback_reason = Some(reason);
            state.backend = Some(backend);
            state.device = device;
            state.backend.as_mut().unwrap().synthesize_with_sink(
                &bounded,
                cancelled,
                &mut sink,
                &mut delivered,
            )?;
        }
        anyhow::ensure!(
            crate::wav_duration_seconds(pending.as_ref())? > 0.0,
            "inference generated empty audio"
        );
        // Publish only successful complete audio, including after GPU fallback.
        pending
            .persist(request.output)
            .map_err(|error| error.error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Qwen3TtsConfig, Qwen3TtsMode};

    #[test]
    #[ignore = "requires real Phonon Q8 assets and native worker via SAYIT_PHONON_TEST_MODEL/WORKER"]
    fn real_phonon_stream_archive_cancellation_and_recovery() {
        let directory = tempfile::tempdir().unwrap();
        let config = EngineConfig::Phonon(crate::PhononConfig {
            model: PathBuf::from(std::env::var_os("SAYIT_PHONON_TEST_MODEL").unwrap()),
            worker: PathBuf::from(std::env::var_os("SAYIT_PHONON_TEST_WORKER").unwrap()),
            voice: "default".into(),
        });
        config.validate().unwrap();
        let engine =
            ManagedEngine::load(&config, &InferenceRuntime::new(directory.path()), true).unwrap();
        assert_eq!(engine.device().backend.as_deref(), Some("phonon"));
        assert_eq!(engine.device().quantization.as_deref(), Some("q8_0"));
        assert_eq!(engine.device().provider, "cpu");
        let text = "The morning is quiet and the words are clear. This synthetic voice is used only for local testing.";
        for name in ["cold.wav", "warm.wav"] {
            let output = directory.path().join(name);
            let start = Instant::now();
            let mut first = None;
            let mut streamed = Vec::new();
            engine
                .synthesize_streaming(
                    &SynthesisRequest {
                        generation: None,
                        text,
                        output: &output,
                        reference_audio: None,
                        reference_text: None,
                        speaking_pace: 1.0,
                    },
                    None,
                    &mut |samples, rate| {
                        assert!(!output.exists(), "partial WAV must stay private");
                        assert_eq!(rate, 24_000);
                        first.get_or_insert(start.elapsed());
                        streamed.extend_from_slice(samples);
                        Ok(())
                    },
                )
                .unwrap();
            let elapsed = start.elapsed();
            assert!(elapsed.saturating_sub(first.unwrap()) > Duration::from_millis(100));
            assert!(streamed.iter().all(|sample| sample.is_finite()));
            assert!(streamed.iter().any(|sample| sample.abs() > 0.01));
            let archived = hound::WavReader::open(&output)
                .unwrap()
                .into_samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let expected = streamed
                .iter()
                .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                .collect::<Vec<_>>();
            assert_eq!(archived, expected);
            eprintln!(
                "Phonon {name}: first PCM {:.3}s, complete {:.3}s, {} samples",
                first.unwrap().as_secs_f64(),
                elapsed.as_secs_f64(),
                streamed.len()
            );
        }
        let cancelled = AtomicBool::new(false);
        let output = directory.path().join("cancelled.wav");
        let request = SynthesisRequest {
            generation: None,
            text,
            output: &output,
            reference_audio: None,
            reference_text: None,
            speaking_pace: 1.0,
        };
        let result = engine.synthesize_streaming(&request, Some(&cancelled), &mut |_, _| {
            cancelled.store(true, Ordering::Relaxed);
            Ok(())
        });
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        assert!(!output.exists());
        assert!(
            !fs::read_dir(directory.path()).unwrap().any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp")),
            "killed worker must not leak partial audio files"
        );
        let recovered = directory.path().join("recovered.wav");
        engine
            .synthesize(&SynthesisRequest {
                output: &recovered,
                ..request
            })
            .unwrap();
        assert!(crate::wav_duration_seconds(&recovered).unwrap() > 1.0);
    }

    #[test]
    #[ignore = "requires installed native Qwen assets, runtime, worker, and a permitted reference via SAYIT_NATIVE_TEST_* variables"]
    fn real_native_qwen_stream_precedes_atomic_wav_publication() {
        let asset = |name| {
            PathBuf::from(
                std::env::var_os(format!("SAYIT_NATIVE_TEST_{name}")).expect("test asset path"),
            )
        };
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("speech.wav");
        let config = EngineConfig::Qwen3TtsOnnx(crate::QwenOnnxConfig {
            model: asset("MODEL"),
            runtime: Some(asset("RUNTIME")),
            worker: asset("WORKER"),
            precision: "int4".into(),
            num_threads: 4,
            device: "cpu".into(),
            language: "English".into(),
        });
        let runtime = InferenceRuntime::new(directory.path());
        let engine = ManagedEngine::load(&config, &runtime, false).unwrap();
        let reference = asset("REFERENCE");
        let transcript = std::env::var("SAYIT_NATIVE_TEST_TRANSCRIPT").unwrap();
        let start = Instant::now();
        let mut first_audio = None;
        let mut delivered = 0;
        engine.synthesize_streaming(&SynthesisRequest { generation: None,
            text: "Streaming audio arrives while the next sentence is generated. The player keeps the natural pitch of the voice. We can pause and resume the speech. This final sentence completes our test.",
            output: &output, reference_audio: Some(&reference), reference_text: Some(&transcript), speaking_pace: 1.0,
        }, None, &mut |samples, rate| {
            assert!(!output.exists(), "only the complete WAV may be published");
            assert_eq!(rate, 24_000);
            first_audio.get_or_insert(start.elapsed());
            delivered += samples.len();
            Ok(())
        }).unwrap();
        let complete = start.elapsed();
        let first_audio = first_audio.expect("streaming PCM");
        let wav = hound::WavReader::open(&output).unwrap();
        assert_eq!(delivered, wav.len() as usize);
        assert!(
            complete.saturating_sub(first_audio) > Duration::from_millis(100),
            "audio must precede the remaining generation"
        );
        eprintln!(
            "Native Qwen real-model acceptance: first PCM {:.2}s, complete {:.2}s, {} samples",
            first_audio.as_secs_f64(),
            complete.as_secs_f64(),
            delivered
        );
    }

    #[test]
    fn cpu_fallback_preserves_native_weight_quantization() {
        let config = EngineConfig::Qwen3TtsOnnx(crate::QwenOnnxConfig {
            worker: "worker".into(),
            model: "model".into(),
            runtime: None,
            device: "cuda".into(),
            precision: "int4".into(),
            num_threads: 4,
            language: "Auto".into(),
        });
        let EngineConfig::Qwen3TtsOnnx(cpu) = config.cpu() else {
            panic!()
        };
        assert_eq!(cpu.device, "cpu");
        assert_eq!(cpu.precision, "int4");
        assert!(config.requires_reference());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_gpu_failure_retries_cpu_without_installing_python() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("native-worker");
        fs::write(&worker,r#"#!/usr/bin/python3
import json,pathlib,sys,wave
device=sys.argv[sys.argv.index('--device')+1]
model=pathlib.Path(sys.argv[sys.argv.index('--model')+1])
with (model/'starts').open('a') as out: out.write(device+'\n')
if device=='cuda': sys.exit(1)
print(json.dumps({'ready':True,'device':'cpu','dtype':'float32','backend':'qwen3-tts-onnx','quantization':'int4'}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 with wave.open(request['output'],'wb') as out:
  out.setparams((1,2,24000,0,'NONE','NONE'));out.writeframes(b'\0\0'*240)
 print(json.dumps({'ok':True}),flush=True)
"#).unwrap();
        fs::set_permissions(&worker, fs::Permissions::from_mode(0o755)).unwrap();
        let library = directory.path().join("runtime");
        fs::write(&library, []).unwrap();
        let config = EngineConfig::Qwen3TtsOnnx(crate::QwenOnnxConfig {
            worker,
            model: directory.path().into(),
            runtime: Some(library),
            device: "auto".into(),
            precision: "int4".into(),
            num_threads: 4,
            language: "Auto".into(),
        });
        let mut runtime = InferenceRuntime::new(directory.path());
        runtime.cuda = Some(CudaDevice {
            driver_version: 12060,
            count: 1,
        });
        let engine = ManagedEngine::load(&config, &runtime, true).unwrap();
        let output = directory.path().join("speech.wav");
        engine
            .synthesize(&SynthesisRequest {
                generation: None,
                text: "hello",
                output: &output,
                reference_audio: None,
                reference_text: None,
                speaking_pace: 1.0,
            })
            .unwrap();
        assert_eq!(engine.device().provider, "cpu");
        assert_eq!(engine.device().backend.as_deref(), Some("qwen3-tts-onnx"));
        assert_eq!(engine.device().quantization.as_deref(), Some("int4"));
        assert!(engine.device().fallback_reason.is_some());
        assert_eq!(
            fs::read_to_string(directory.path().join("starts")).unwrap(),
            "cuda\ncpu\n"
        );
        assert!(!directory.path().join("runtimes/qwen").exists());
    }

    #[test]
    fn provider_selection_covers_every_onnx_model_family() {
        for value in [
            serde_json::json!({"engine":"sherpa-onnx-vits", "model":"model", "tokens":"tokens"}),
            serde_json::json!({"engine":"sherpa-onnx-kokoro", "model":"model", "tokens":"tokens", "voices":"voices", "data_dir":"data", "lexicons":["lexicon"]}),
            serde_json::json!({"engine":"sherpa-onnx-kitten", "model":"model", "tokens":"tokens", "voices":"voices", "data_dir":"data"}),
            serde_json::json!({"engine":"sherpa-onnx-pocket", "lm_flow":"flow", "lm_main":"main", "encoder":"encoder", "decoder":"decoder", "text_conditioner":"text", "vocab_json":"vocab", "token_scores_json":"scores"}),
        ] {
            let mut config: EngineConfig = serde_json::from_value(value).unwrap();
            assert_eq!(config.provider(), "auto");
            config.set_sherpa_provider("cuda", Some(Path::new("gpu/bin/tts")));
            assert_eq!(config.provider(), "cuda");
            assert_eq!(
                SherpaOnnxEngine::from_config(config.clone()).executable(),
                Path::new("gpu/bin/tts")
            );
            assert_eq!(config.cpu().provider(), "cpu");
        }
    }

    #[cfg(target_os = "linux")]
    fn fake_qwen(directory: &Path) -> (EngineConfig, InferenceRuntime) {
        let worker = directory.join("worker.py");
        fs::write(&worker, r#"import json, pathlib, sys, wave
device = sys.argv[sys.argv.index('--device') + 1]
root = pathlib.Path(sys.argv[sys.argv.index('--model') + 1])
with (root / 'starts').open('a') as log:
    log.write(device + '\n')
if device == 'auto' and (root / 'fail-startup').exists():
    raise RuntimeError('GPU initialization failed')
gpu = device == 'auto'
print(json.dumps({'ready': True, 'device': 'cuda:0' if gpu else 'cpu', 'dtype': 'bfloat16' if gpu else 'float32'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if gpu or request['text'] == 'fail on CPU too':
        print(json.dumps({'ok': False, 'error': 'out of memory' if gpu else 'CPU failure'}), flush=True)
        continue
    with wave.open(request['output'], 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wav.writeframes(b'\0\0' * 160)
    print(json.dumps({'ok': True}), flush=True)
"#).unwrap();
        let config = EngineConfig::Qwen3Tts(Qwen3TtsConfig {
            python: "/usr/bin/python3".into(),
            worker,
            model: directory.into(),
            mode: Qwen3TtsMode::VoiceClone,
            device: "auto".into(),
            dtype: "auto".into(),
            language: "Auto".into(),
            speaker: None,
            voice_description: None,
        });
        let mut runtime = InferenceRuntime::new(directory);
        runtime.cuda = None; // Never install software or require GPU hardware in unit tests.
        (config, runtime)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn gpu_generation_failure_releases_worker_and_reuses_cpu_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let (config, runtime) = fake_qwen(directory.path());
        let engine = ManagedEngine::load(&config, &runtime, true).unwrap();
        assert_eq!(engine.device().provider, "cuda");
        let output = directory.path().join("speech.wav");
        for _ in 0..2 {
            engine
                .synthesize(&SynthesisRequest {
                    generation: None,
                    text: "A short sentence.",
                    output: &output,
                    reference_audio: None,
                    reference_text: Some("Reference words."),
                    speaking_pace: 1.0,
                })
                .unwrap();
            assert!(crate::wav_duration_seconds(&output).unwrap() > 0.0);
        }
        assert_eq!(engine.device().provider, "cpu");
        assert_eq!(engine.device().dtype.as_deref(), Some("float32"));
        assert!(
            engine
                .device()
                .fallback_reason
                .unwrap()
                .contains("out of memory")
        );
        assert_eq!(
            fs::read_to_string(directory.path().join("starts")).unwrap(),
            "auto\ncpu\n"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn gpu_startup_failure_and_explicit_cpu_preference_are_reported_correctly() {
        for fail_startup in [true, false] {
            let directory = tempfile::tempdir().unwrap();
            let (config, runtime) = fake_qwen(directory.path());
            if fail_startup {
                fs::write(directory.path().join("fail-startup"), []).unwrap();
            }
            let engine = ManagedEngine::load(&config, &runtime, fail_startup).unwrap();
            assert_eq!(engine.device().provider, "cpu");
            assert_eq!(engine.device().fallback_reason.is_some(), fail_startup);
            assert_eq!(
                fs::read_to_string(directory.path().join("starts")).unwrap(),
                if fail_startup { "auto\ncpu\n" } else { "cpu\n" }
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn failed_gpu_and_cpu_attempts_preserve_existing_output() {
        let directory = tempfile::tempdir().unwrap();
        let (config, runtime) = fake_qwen(directory.path());
        let engine = ManagedEngine::load(&config, &runtime, true).unwrap();
        let output = directory.path().join("speech.wav");
        fs::write(&output, b"previous audio").unwrap();
        assert!(
            engine
                .synthesize(&SynthesisRequest {
                    generation: None,
                    text: "fail on CPU too",
                    output: &output,
                    reference_audio: None,
                    reference_text: None,
                    speaking_pace: 1.0,
                })
                .is_err()
        );
        assert_eq!(fs::read(&output).unwrap(), b"previous audio");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_failed_gpu_stream_never_replays_already_delivered_audio_on_cpu() {
        let directory = tempfile::tempdir().unwrap();
        let (config, runtime) = fake_qwen(directory.path());
        let worker = directory.path().join("worker.py");
        let source = fs::read_to_string(&worker).unwrap().replace(
            "if gpu or request['text'] == 'fail on CPU too':",
            "if request.get('stream'):\n        print(json.dumps({'audio': {'sample_rate': 16000, 'samples': [0.2] * 160}}), flush=True)\n    if gpu or request['text'] == 'fail on CPU too':");
        fs::write(worker, source).unwrap();
        let engine = ManagedEngine::load(&config, &runtime, true).unwrap();
        let output = directory.path().join("speech.wav");
        fs::write(&output, b"previous complete audio").unwrap();
        let mut frames = 0;
        let result = engine.synthesize_streaming(
            &SynthesisRequest {
                generation: None,
                text: "A short sentence.",
                output: &output,
                reference_audio: None,
                reference_text: None,
                speaking_pace: 1.0,
            },
            None,
            &mut |samples, _| {
                frames += samples.len();
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(frames, 160);
        assert_eq!(
            fs::read_to_string(directory.path().join("starts")).unwrap(),
            "auto\n"
        );
        assert_eq!(fs::read(&output).unwrap(), b"previous complete audio");
    }
}
