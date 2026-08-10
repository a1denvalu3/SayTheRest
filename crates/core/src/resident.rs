use crate::{EngineConfig, SynthesisRequest};
use anyhow::{Context, Result, bail};
use libloading::Library;
use std::{
    ffi::CString,
    os::raw::c_char,
    path::{Path, PathBuf},
    ptr,
};

#[repr(C)]
struct VitsConfig {
    model: *const c_char,
    lexicon: *const c_char,
    tokens: *const c_char,
    data_dir: *const c_char,
    noise_scale: f32,
    noise_scale_w: f32,
    length_scale: f32,
    dict_dir: *const c_char,
}
#[repr(C)]
struct MatchaConfig {
    acoustic_model: *const c_char,
    vocoder: *const c_char,
    lexicon: *const c_char,
    tokens: *const c_char,
    data_dir: *const c_char,
    noise_scale: f32,
    length_scale: f32,
    dict_dir: *const c_char,
}
#[repr(C)]
struct KokoroConfig {
    model: *const c_char,
    voices: *const c_char,
    tokens: *const c_char,
    data_dir: *const c_char,
    length_scale: f32,
    dict_dir: *const c_char,
    lexicon: *const c_char,
    lang: *const c_char,
}
#[repr(C)]
struct KittenConfig {
    model: *const c_char,
    voices: *const c_char,
    tokens: *const c_char,
    data_dir: *const c_char,
    length_scale: f32,
}
#[repr(C)]
struct ZipVoiceConfig {
    tokens: *const c_char,
    encoder: *const c_char,
    decoder: *const c_char,
    vocoder: *const c_char,
    data_dir: *const c_char,
    lexicon: *const c_char,
    feat_scale: f32,
    t_shift: f32,
    target_rms: f32,
    guidance_scale: f32,
}
#[repr(C)]
struct PocketConfig {
    lm_flow: *const c_char,
    lm_main: *const c_char,
    encoder: *const c_char,
    decoder: *const c_char,
    text_conditioner: *const c_char,
    vocab_json: *const c_char,
    token_scores_json: *const c_char,
    voice_embedding_cache_capacity: i32,
}
#[repr(C)]
struct SupertonicConfig {
    duration_predictor: *const c_char,
    text_encoder: *const c_char,
    vector_estimator: *const c_char,
    vocoder: *const c_char,
    tts_json: *const c_char,
    unicode_indexer: *const c_char,
    voice_style: *const c_char,
}
#[repr(C)]
struct ModelConfig {
    vits: VitsConfig,
    num_threads: i32,
    debug: i32,
    provider: *const c_char,
    matcha: MatchaConfig,
    kokoro: KokoroConfig,
    kitten: KittenConfig,
    zipvoice: ZipVoiceConfig,
    pocket: PocketConfig,
    supertonic: SupertonicConfig,
}
#[repr(C)]
struct TtsConfig {
    model: ModelConfig,
    rule_fsts: *const c_char,
    max_num_sentences: i32,
    rule_fars: *const c_char,
    silence_scale: f32,
}
#[repr(C)]
struct GenerationConfig {
    silence_scale: f32,
    speed: f32,
    sid: i32,
    reference_audio: *const f32,
    reference_audio_len: i32,
    reference_sample_rate: i32,
    reference_text: *const c_char,
    num_steps: i32,
    extra: *const c_char,
}
#[repr(C)]
struct GeneratedAudio {
    samples: *const f32,
    n: i32,
    sample_rate: i32,
}
#[repr(C)]
struct OfflineTts {
    _private: [u8; 0],
}

type Create = unsafe extern "C" fn(*const TtsConfig) -> *const OfflineTts;
type Destroy = unsafe extern "C" fn(*const OfflineTts);
type Generate = unsafe extern "C" fn(
    *const OfflineTts,
    *const c_char,
    *const GenerationConfig,
    Option<unsafe extern "C" fn(*const f32, i32, f32, *mut std::ffi::c_void) -> i32>,
    *mut std::ffi::c_void,
) -> *const GeneratedAudio;
type DestroyAudio = unsafe extern "C" fn(*const GeneratedAudio);
type WriteWave = unsafe extern "C" fn(*const f32, i32, i32, *const c_char) -> i32;

pub struct ResidentSherpaEngine {
    tts: *const OfflineTts,
    destroy: Destroy,
    generate: Generate,
    destroy_audio: DestroyAudio,
    write_wave: WriteWave,
    speaker_id: i32,
    num_steps: i32,
    needs_reference: bool,
    _library: Library,
}

// The service serializes all access to one engine. sherpa-onnx documents an
// instance as usable from one inference thread; it is never shared concurrently.
unsafe impl Send for ResidentSherpaEngine {}

impl ResidentSherpaEngine {
    pub fn load(config: &EngineConfig) -> Result<Self> {
        if matches!(config, EngineConfig::Qwen3Tts(_)) {
            bail!("Qwen models require the persistent generative worker");
        }
        let executable = executable(config);
        let library_path = c_api_library(&executable);
        // SAFETY: The library is shipped with the matching executable and remains
        // owned by this object until after the native TTS instance is destroyed.
        let library = unsafe { load_c_api_library(&library_path) }
            .with_context(|| format!("failed to load {}", library_path.display()))?;
        let create: Create = unsafe { *library.get(b"SherpaOnnxCreateOfflineTts\0")? };
        let destroy: Destroy = unsafe { *library.get(b"SherpaOnnxDestroyOfflineTts\0")? };
        let generate: Generate =
            unsafe { *library.get(b"SherpaOnnxOfflineTtsGenerateWithConfig\0")? };
        let destroy_audio: DestroyAudio =
            unsafe { *library.get(b"SherpaOnnxDestroyOfflineTtsGeneratedAudio\0")? };
        let write_wave: WriteWave = unsafe { *library.get(b"SherpaOnnxWriteWave\0")? };

        let mut strings = Vec::<CString>::new();
        let mut keep = |value: String| -> Result<*const c_char> {
            strings.push(CString::new(value).context("native TTS path contains a NUL byte")?);
            Ok(strings.last().unwrap().as_ptr())
        };
        let mut native: TtsConfig = unsafe { std::mem::zeroed() };
        let (speaker_id, num_steps, needs_reference) = match config {
            EngineConfig::SherpaOnnxVits(value) => {
                native.model.vits.model = keep(path_string(&value.model))?;
                native.model.vits.tokens = keep(path_string(&value.tokens))?;
                native.model.vits.data_dir = optional_path(&mut keep, value.data_dir.as_deref())?;
                native.model.vits.lexicon = optional_path(&mut keep, value.lexicon.as_deref())?;
                native.model.vits.noise_scale = 0.667;
                native.model.vits.noise_scale_w = 0.8;
                native.model.vits.length_scale = 1.0;
                native.model.num_threads = value.num_threads as i32;
                native.model.provider = keep(value.provider.clone())?;
                (value.speaker_id as i32, 0, false)
            }
            EngineConfig::SherpaOnnxKokoro(value) => {
                native.model.kokoro.model = keep(path_string(&value.model))?;
                native.model.kokoro.voices = keep(path_string(&value.voices))?;
                native.model.kokoro.tokens = keep(path_string(&value.tokens))?;
                native.model.kokoro.data_dir = keep(path_string(&value.data_dir))?;
                native.model.kokoro.lexicon = keep(
                    value
                        .lexicons
                        .iter()
                        .map(|path| path_string(path))
                        .collect::<Vec<_>>()
                        .join(","),
                )?;
                native.model.kokoro.length_scale = 1.0;
                native.model.num_threads = value.num_threads as i32;
                native.model.provider = keep(value.provider.clone())?;
                (value.speaker_id as i32, 0, false)
            }
            EngineConfig::SherpaOnnxKitten(value) => {
                native.model.kitten.model = keep(path_string(&value.model))?;
                native.model.kitten.voices = keep(path_string(&value.voices))?;
                native.model.kitten.tokens = keep(path_string(&value.tokens))?;
                native.model.kitten.data_dir = keep(path_string(&value.data_dir))?;
                native.model.kitten.length_scale = 1.0;
                native.model.num_threads = value.num_threads as i32;
                native.model.provider = keep(value.provider.clone())?;
                (value.speaker_id as i32, 0, false)
            }
            EngineConfig::SherpaOnnxPocket(value) => {
                native.model.pocket.lm_flow = keep(path_string(&value.lm_flow))?;
                native.model.pocket.lm_main = keep(path_string(&value.lm_main))?;
                native.model.pocket.encoder = keep(path_string(&value.encoder))?;
                native.model.pocket.decoder = keep(path_string(&value.decoder))?;
                native.model.pocket.text_conditioner = keep(path_string(&value.text_conditioner))?;
                native.model.pocket.vocab_json = keep(path_string(&value.vocab_json))?;
                native.model.pocket.token_scores_json =
                    keep(path_string(&value.token_scores_json))?;
                native.model.pocket.voice_embedding_cache_capacity = 8;
                native.model.num_threads = value.num_threads as i32;
                native.model.provider = keep(value.provider.clone())?;
                (0, value.num_steps as i32, true)
            }
            EngineConfig::Qwen3Tts(_) => unreachable!(),
        };
        native.max_num_sentences = 2;
        native.silence_scale = 0.2;
        let tts = unsafe { create(&native) };
        if tts.is_null() {
            bail!("sherpa-onnx could not load the selected model")
        }
        Ok(Self {
            tts,
            destroy,
            generate,
            destroy_audio,
            write_wave,
            speaker_id,
            num_steps,
            needs_reference,
            _library: library,
        })
    }

    pub fn synthesize(&mut self, request: &SynthesisRequest<'_>) -> Result<()> {
        if request.text.trim().is_empty() {
            bail!("text must not be empty")
        }
        let text = CString::new(request.text).context("speech text contains a NUL byte")?;
        let output =
            CString::new(path_string(request.output)).context("output path contains a NUL byte")?;
        let (reference, sample_rate) = if self.needs_reference {
            read_reference(
                request
                    .reference_audio
                    .context("the selected model requires a voice profile")?,
            )?
        } else {
            (Vec::new(), 0)
        };
        let generation = GenerationConfig {
            silence_scale: 0.2,
            speed: request.speaking_pace as f32,
            sid: self.speaker_id,
            reference_audio: if reference.is_empty() {
                ptr::null()
            } else {
                reference.as_ptr()
            },
            reference_audio_len: i32::try_from(reference.len())
                .context("reference audio is too long")?,
            reference_sample_rate: sample_rate,
            reference_text: ptr::null(),
            num_steps: self.num_steps,
            extra: ptr::null(),
        };
        let audio =
            unsafe { (self.generate)(self.tts, text.as_ptr(), &generation, None, ptr::null_mut()) };
        if audio.is_null() {
            bail!("sherpa-onnx synthesis failed")
        }
        let audio_ref = unsafe { &*audio };
        let wrote = if audio_ref.n > 0 && audio_ref.sample_rate > 0 && !audio_ref.samples.is_null()
        {
            unsafe {
                (self.write_wave)(
                    audio_ref.samples,
                    audio_ref.n,
                    audio_ref.sample_rate,
                    output.as_ptr(),
                )
            }
        } else {
            0
        };
        unsafe { (self.destroy_audio)(audio) };
        if wrote != 1 || !request.output.is_file() {
            bail!("sherpa-onnx did not write generated audio")
        }
        Ok(())
    }
}

impl Drop for ResidentSherpaEngine {
    fn drop(&mut self) {
        if !self.tts.is_null() {
            unsafe { (self.destroy)(self.tts) }
        }
    }
}

fn executable(config: &EngineConfig) -> PathBuf {
    match config {
        EngineConfig::SherpaOnnxVits(v) => v.executable.clone(),
        EngineConfig::SherpaOnnxKokoro(v) => v.executable.clone(),
        EngineConfig::SherpaOnnxKitten(v) => v.executable.clone(),
        EngineConfig::SherpaOnnxPocket(v) => v.executable.clone(),
        EngineConfig::Qwen3Tts(v) => v.python.clone(),
    }
}
fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
fn optional_path(
    keep: &mut impl FnMut(String) -> Result<*const c_char>,
    path: Option<&Path>,
) -> Result<*const c_char> {
    path.map(|path| keep(path_string(path)))
        .transpose()
        .map(|value| value.unwrap_or(ptr::null()))
}
fn c_api_library(executable: &Path) -> PathBuf {
    let bin = executable.parent().unwrap_or_else(|| Path::new("."));
    if cfg!(windows) {
        bin.parent()
            .unwrap_or(bin)
            .join("lib")
            .join("sherpa-onnx-c-api.dll")
    } else if cfg!(target_os = "macos") {
        bin.parent()
            .unwrap_or(bin)
            .join("lib")
            .join("libsherpa-onnx-c-api.dylib")
    } else {
        bin.parent()
            .unwrap_or(bin)
            .join("lib")
            .join("libsherpa-onnx-c-api.so")
    }
}

#[cfg(windows)]
unsafe fn load_c_api_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        Library as WindowsLibrary,
    };

    // Sherpa's C API depends on adjacent ONNX Runtime and provider DLLs. Loading
    // with an absolute path alone does not add that directory to dependency
    // resolution on Windows, so opt into the safe DLL search flags explicitly.
    unsafe {
        WindowsLibrary::load_with_flags(
            path,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
        .map(Into::into)
    }
}

#[cfg(not(windows))]
unsafe fn load_c_api_library(path: &Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(path) }
}

fn read_reference(path: &Path) -> Result<(Vec<f32>, i32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    anyhow::ensure!(spec.channels == 1, "reference WAV must be mono");
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample.saturating_sub(1) as u32)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    Ok((samples, spec.sample_rate as i32))
}

#[cfg(test)]
mod tests {
    use super::{ResidentSherpaEngine, c_api_library};
    use crate::{EngineConfig, KittenConfig, SynthesisRequest, wav_duration_seconds};
    use std::path::{Path, PathBuf};

    #[test]
    fn resolves_c_api_next_to_the_packaged_runtime() {
        let path = c_api_library(Path::new("bundle/runtime/bin/sherpa-onnx-offline-tts"));
        let expected = if cfg!(windows) {
            "sherpa-onnx-c-api.dll"
        } else if cfg!(target_os = "macos") {
            "libsherpa-onnx-c-api.dylib"
        } else {
            "libsherpa-onnx-c-api.so"
        };
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some(expected)
        );
        assert_eq!(path.parent(), Some(Path::new("bundle/runtime/lib")));
    }

    #[test]
    #[ignore = "requires extracted Kitten weights and a packaged sherpa runtime"]
    fn kitten_models_synthesize_through_resident_runtime() {
        let models = PathBuf::from(std::env::var_os("SAY_THE_REST_KITTEN_TEST_ROOT").unwrap());
        let executable = PathBuf::from(std::env::var_os("SAY_THE_REST_TTS_TEST_RUNTIME").unwrap());
        for (directory, model) in [
            ("kitten-mini-en-v0_8", "model.onnx"),
            ("kitten-nano-en-v0_8-int8", "model.int8.onnx"),
        ] {
            let root = models.join(directory);
            let config = EngineConfig::SherpaOnnxKitten(KittenConfig {
                executable: executable.clone(),
                model: root.join(model),
                voices: root.join("voices.bin"),
                tokens: root.join("tokens.txt"),
                data_dir: root.join("espeak-ng-data"),
                provider: "cpu".into(),
                num_threads: 4,
                speaker_id: 1,
            });
            config.validate().unwrap();
            let mut engine = ResidentSherpaEngine::load(&config).unwrap();
            let temporary = tempfile::tempdir().unwrap();
            let output = temporary.path().join(format!("{directory}.wav"));
            engine
                .synthesize(&SynthesisRequest {
                    text: "Kitten speech runs through the resident native engine.",
                    output: &output,
                    reference_audio: None,
                    speaking_pace: 1.0,
                })
                .unwrap();
            assert!(wav_duration_seconds(&output).unwrap() > 1.0);
        }
    }
}
