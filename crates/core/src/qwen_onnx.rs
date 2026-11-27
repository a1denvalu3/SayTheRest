//! Native execution of the pinned Qwen3-TTS Base ONNX component graphs.
//! The prompt layout follows QwenLM/Qwen3-TTS and ONNX Community's inference.py.
//! No Python code or remote tokenizer code is executed by this module.

use crate::{
    cpu_topology::{self, CpuTopology},
    qwen_code_predictor::FastCodePredictor,
};
use anyhow::{Context, Result, bail, ensure};
use ort::{
    environment::{Environment, GlobalThreadPoolOptions},
    ep,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{Session, builder::GraphOptimizationLevel},
    value::{DynValue, Tensor},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::mpsc,
    thread::{self, JoinHandle},
    time::Instant,
};
use tokenizers::{
    AddedToken, SplitDelimiterBehavior, Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
    },
};

const SAMPLE_RATE: usize = 24_000;
const GROUPS: usize = 16;
const DECODE_FRAMES: usize = 25;

#[derive(Clone, Deserialize)]
pub struct NativeRequest {
    pub text: String,
    pub output: PathBuf,
    #[serde(default)]
    pub reference_audio: Option<PathBuf>,
    pub reference_text: Option<String>,
    #[serde(default = "auto_language")]
    pub language: String,
    pub seed: Option<u64>,
    #[serde(default = "max_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "sampling")]
    pub do_sample: bool,
    #[serde(flatten)]
    pub sampling: crate::QwenSamplingOptions,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_format: Option<String>,
}
fn auto_language() -> String {
    "Auto".into()
}
fn max_tokens() -> usize {
    1024
}
fn sampling() -> bool {
    true
}

#[derive(Clone)]
struct ReferencePrompt {
    key: Vec<u8>,
    speaker: Vec<f32>,
    codes: Vec<f32>,
}

struct CachedEmbeddings {
    bos: Vec<f32>,
    eos: Vec<f32>,
    pad: Vec<f32>,
    codec_pad: Vec<f32>,
    codec_bos: Vec<f32>,
}

enum PredictorBackend {
    Fast(Box<FastCodePredictor>),
    Ort(Session),
}

enum DecoderMessage {
    Decode {
        codes: Vec<i64>,
        reply: mpsc::Sender<Result<Vec<f32>>>,
    },
    Shutdown,
}

struct PipelinedDecoder {
    tx: mpsc::Sender<DecoderMessage>,
    handle: Option<JoinHandle<()>>,
}

impl PipelinedDecoder {
    fn spawn(mut session: Session, main_cpu: Option<usize>) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<DecoderMessage>();
        let handle = thread::Builder::new()
            .name("qwen-tok-decoder".into())
            .spawn(move || {
                if let Some(cpu) = main_cpu {
                    cpu_topology::set_current_thread_affinity(&[cpu]);
                }
                while let Ok(msg) = rx.recv() {
                    match msg {
                        DecoderMessage::Decode { codes, reply } => {
                            let res = decode_chunk_session(&mut session, &codes);
                            let _ = reply.send(res);
                        }
                        DecoderMessage::Shutdown => break,
                    }
                }
            })
            .context("cannot spawn pipelined Qwen audio decoder thread")?;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    fn enqueue(&self, codes: Vec<i64>, reply: mpsc::Sender<Result<Vec<f32>>>) -> Result<()> {
        self.tx
            .send(DecoderMessage::Decode { codes, reply })
            .map_err(|_| anyhow::anyhow!("pipelined audio decoder thread exited"))
    }
}

impl Drop for PipelinedDecoder {
    fn drop(&mut self) {
        let _ = self.tx.send(DecoderMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

enum DecoderBackend {
    Inline(Session),
    Pipelined(PipelinedDecoder),
}

pub struct NativeQwenPipeline {
    env: Environment,
    directory: PathBuf,
    gpu: bool,
    threads: usize,
    gen_cpus: Vec<usize>,
    text_embed: Session,
    codec_embed: Session,
    residual_embed: Session,
    talker: Session,
    predictor: PredictorBackend,
    encoder: Option<Session>,
    decoder: DecoderBackend,
    speaker: Option<Session>,
    tokenizer: Tokenizer,
    config: Value,
    hidden: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    past_names: Vec<String>,
    device: String,
    reference: Option<ReferencePrompt>,
    cached_embeddings: CachedEmbeddings,
    sample_candidates: Vec<(usize, f32)>,
    sample_weights: Vec<f64>,
}

fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape, data))?.into_dyn())
}
fn tensor_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape, data))?.into_dyn())
}
fn run(session: &mut Session, inputs: Vec<(&str, DynValue)>) -> Result<Vec<DynValue>> {
    Ok(session
        .run(inputs)?
        .into_iter()
        .map(|(_, value)| value)
        .collect())
}
fn floats(value: &DynValue) -> Result<Vec<f32>> {
    Ok(value.try_extract_tensor::<f32>()?.1.to_vec())
}
fn embed(session: &mut Session, input: &str, ids: &[i64]) -> Result<Vec<f32>> {
    floats(
        &run(
            session,
            vec![(input, tensor_i64(vec![1, ids.len()], ids.to_vec())?)],
        )?[0],
    )
}

fn build_shared_session(
    env: &Environment,
    directory: &Path,
    name: &str,
    gpu: bool,
    threads: usize,
    use_global_pool: bool,
) -> Result<Session> {
    let mut builder = Session::builder(env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if !use_global_pool {
        builder = builder
            .with_independent_thread_pool()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    if gpu {
        builder = builder
            .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    builder
        .commit_from_file(directory.join(format!("{name}.onnx")))
        .with_context(|| format!("cannot load native Qwen {name}"))
}

fn build_decoder_session(
    env: &Environment,
    directory: &Path,
    gpu: bool,
    threads: usize,
    decoder_cpus: &[usize],
    use_global_pool: bool,
) -> Result<Session> {
    if !gpu && !decoder_cpus.is_empty() {
        let dec_threads = decoder_cpus.len();
        let mut builder = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_independent_thread_pool()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(dec_threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_op_spinning(false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_flush_to_zero()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some(affinity) = CpuTopology::ort_intra_affinity(decoder_cpus, dec_threads) {
            builder = builder
                .with_config_entry("session.intra_op_thread_affinities", affinity)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        return builder
            .commit_from_file(directory.join("tok_decoder.onnx"))
            .context("cannot load native Qwen tok_decoder");
    }
    build_shared_session(env, directory, "tok_decoder", gpu, threads, use_global_pool)
}

fn decode_chunk_session(decoder: &mut Session, chunk: &[i64]) -> Result<Vec<f32>> {
    ensure!(
        !chunk.is_empty() && chunk.len().is_multiple_of(GROUPS),
        "invalid generated audio codes"
    );
    let frames = chunk.len() / GROUPS;
    ensure!(
        frames <= DECODE_FRAMES,
        "decoder chunk exceeds DECODE_FRAMES"
    );
    let padded = if frames == DECODE_FRAMES {
        chunk.to_vec()
    } else {
        (0..DECODE_FRAMES * GROUPS)
            .map(|i| chunk[i % chunk.len()])
            .collect()
    };
    let output = run(
        decoder,
        vec![(
            "audio_codes",
            tensor_i64(vec![1, DECODE_FRAMES, GROUPS], padded)?,
        )],
    )?;
    let samples = output[0].try_extract_tensor::<f32>()?.1;
    let count = (samples.len() * frames + DECODE_FRAMES / 2) / DECODE_FRAMES;
    let slice = &samples[..count];
    ensure!(
        slice.iter().all(|sample| sample.is_finite()),
        "decoder generated invalid samples"
    );
    Ok(slice.to_vec())
}

fn init_environment(runtime: &Path, gpu: bool, threads: usize, gen_cpus: &[usize]) -> Result<(Environment, bool)> {
    static PROCESS_ENV: std::sync::OnceLock<(Environment, bool)> = std::sync::OnceLock::new();
    if let Some((env, has_pool)) = PROCESS_ENV.get() {
        return Ok((env.clone(), *has_pool && !gpu));
    }
    let mut builder = ort::init_from(runtime)?
        .with_name("sayit-qwen-onnx")
        .with_telemetry(false);
    if !gpu {
        let mut pool = GlobalThreadPoolOptions::default()
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_spin_control(true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_flush_to_zero()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some(affinity) = CpuTopology::ort_intra_affinity(gen_cpus, threads) {
            pool = pool
                .with_intra_affinity(affinity)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        builder = builder.with_global_thread_pool(pool);
    }
    let env = builder.build()?;
    let _ = PROCESS_ENV.set((env.clone(), !gpu));
    Ok((env, !gpu))
}

impl NativeQwenPipeline {
    pub fn load(
        model: &Path,
        runtime: &Path,
        precision: &str,
        device: &str,
        threads: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(precision, "int4" | "fp32"),
            "native Qwen precision must be int4 or fp32"
        );
        ensure!(
            matches!(device, "cpu" | "cuda" | "cuda:0"),
            "unsupported native Qwen device"
        );
        let topo = cpu_topology::detect();
        let threads = if threads == 0 {
            topo.recommended_qwen_onnx_threads()
        } else {
            threads
        };
        ensure!(
            threads > 0 && threads <= 64,
            "inference threads must be between 1 and 64"
        );
        let gpu = device.starts_with("cuda");
        let gen_cpus = if gpu {
            Vec::new()
        } else {
            topo.select_worker_cpus(threads)
        };
        let decoder_cpus = if gpu {
            Vec::new()
        } else {
            topo.pipelined_decoder_cpus(&gen_cpus)
        };
        let (env, use_global_pool) = init_environment(runtime, gpu, threads, &gen_cpus)?;
        let directory = model.join(precision);

        // Validate that lazy-loaded optional reference-encoding graphs exist up front.
        for required in ["speaker_encoder", "tok_encoder"] {
            let path = directory.join(format!("{required}.onnx"));
            ensure!(
                path.is_file(),
                "cannot load native Qwen {required}: {} does not exist",
                path.display()
            );
        }

        let config: Value = serde_json::from_slice(&fs::read(model.join("model.json"))?)?;
        ensure!(
            config["model_type"] == "qwen3_tts" && config["tts_model_type"] == "base",
            "native worker requires Qwen3-TTS Base"
        );
        let dimension = |key: &str| -> Result<usize> {
            config["talker_config"][key]
                .as_u64()
                .map(|v| v as usize)
                .filter(|v| *v > 0 && *v <= 32768)
                .with_context(|| format!("invalid Qwen dimension: {key}"))
        };
        let hidden = dimension("hidden_size")?;
        let layers = dimension("num_hidden_layers")?;
        let kv_heads = dimension("num_key_value_heads")?;
        let head_dim = dimension("head_dim")?;

        // Load independent heavy graphs concurrently to cut cold startup time.
        let (talker, decoder_session, predictor, text_embed, codec_embed, residual_embed, tokenizer) =
            thread::scope(|scope| -> Result<_> {
                let decoder_handle = scope.spawn(|| {
                    build_decoder_session(
                        &env,
                        &directory,
                        gpu,
                        threads,
                        &decoder_cpus,
                        use_global_pool,
                    )
                });
                let talker_handle = scope.spawn(|| {
                    build_shared_session(
                        &env,
                        &directory,
                        "talker_cache",
                        gpu,
                        threads,
                        use_global_pool,
                    )
                });
                let predictor_handle = scope.spawn(|| -> Result<PredictorBackend> {
                    if !gpu
                        && precision == "int4"
                        && let Some(fast) = FastCodePredictor::try_load(
                            &directory.join("code_predictor.onnx"),
                            threads,
                            &gen_cpus,
                        )?
                    {
                        return Ok(PredictorBackend::Fast(Box::new(fast)));
                    }
                    Ok(PredictorBackend::Ort(build_shared_session(
                        &env,
                        &directory,
                        "code_predictor",
                        gpu,
                        threads,
                        use_global_pool,
                    )?))
                });
                let embeds_handle = scope.spawn(|| -> Result<_> {
                    let text_embed = build_shared_session(
                        &env,
                        &directory,
                        "text_embed",
                        gpu,
                        threads,
                        use_global_pool,
                    )?;
                    let codec_embed = build_shared_session(
                        &env,
                        &directory,
                        "codec_embed",
                        gpu,
                        threads,
                        use_global_pool,
                    )?;
                    let residual_embed = build_shared_session(
                        &env,
                        &directory,
                        "residual_embed",
                        gpu,
                        threads,
                        use_global_pool,
                    )?;
                    Ok((text_embed, codec_embed, residual_embed))
                });
                let tokenizer = load_tokenizer(model)?;
                let (text_embed, codec_embed, residual_embed) = embeds_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("embedding loader panicked"))??;
                let predictor = predictor_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("predictor loader panicked"))??;
                let talker = talker_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("talker loader panicked"))??;
                let decoder_session = decoder_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("decoder loader panicked"))??;
                Ok((
                    talker,
                    decoder_session,
                    predictor,
                    text_embed,
                    codec_embed,
                    residual_embed,
                    tokenizer,
                ))
            })?;

        ensure!(
            talker.inputs().len() >= 3,
            "Qwen cache graph has too few inputs"
        );
        let past_names = talker.inputs()[3..]
            .iter()
            .map(|input| input.name().to_owned())
            .collect::<Vec<_>>();
        ensure!(
            past_names.len() == 2 * layers,
            "Qwen cache graph does not match its architecture"
        );
        ensure!(
            talker.outputs().len() == 2 * layers + 2,
            "Qwen cache graph has unexpected outputs"
        );

        let mut text_embed = text_embed;
        let mut codec_embed = codec_embed;
        let config_id = |key: &str| -> Result<i64> {
            config[key]
                .as_i64()
                .or_else(|| config["talker_config"][key].as_i64())
                .with_context(|| format!("missing Qwen token: {key}"))
        };
        let special = embed(
            &mut text_embed,
            "text_ids",
            &[
                config_id("tts_bos_token_id")?,
                config_id("tts_eos_token_id")?,
                config_id("tts_pad_token_id")?,
            ],
        )?;
        ensure!(special.len() == 3 * hidden, "invalid special text embeddings");
        let codec_special = embed(
            &mut codec_embed,
            "codec_ids",
            &[config_id("codec_pad_id")?, config_id("codec_bos_id")?],
        )?;
        ensure!(
            codec_special.len() == 2 * hidden,
            "invalid special codec embeddings"
        );
        let cached_embeddings = CachedEmbeddings {
            bos: special[..hidden].to_vec(),
            eos: special[hidden..2 * hidden].to_vec(),
            pad: special[2 * hidden..3 * hidden].to_vec(),
            codec_pad: codec_special[..hidden].to_vec(),
            codec_bos: codec_special[hidden..2 * hidden].to_vec(),
        };

        let decoder = if !gpu && !decoder_cpus.is_empty() {
            DecoderBackend::Pipelined(PipelinedDecoder::spawn(
                decoder_session,
                decoder_cpus.first().copied(),
            )?)
        } else {
            DecoderBackend::Inline(decoder_session)
        };

        Ok(Self {
            env,
            directory,
            gpu,
            threads,
            gen_cpus,
            text_embed,
            codec_embed,
            residual_embed,
            talker,
            predictor,
            encoder: None,
            decoder,
            speaker: None,
            tokenizer,
            config,
            hidden,
            layers,
            kv_heads,
            head_dim,
            past_names,
            device: if gpu { "cuda:0" } else { "cpu" }.into(),
            reference: None,
            cached_embeddings,
            sample_candidates: Vec::with_capacity(3072),
            sample_weights: Vec::with_capacity(128),
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    fn speaker_session(&mut self) -> Result<&mut Session> {
        if self.speaker.is_none() {
            let session = build_shared_session(
                &self.env,
                &self.directory,
                "speaker_encoder",
                self.gpu,
                self.threads,
                !self.gpu,
            )?;
            self.speaker = Some(session);
        }
        Ok(self.speaker.as_mut().unwrap())
    }

    fn encoder_session(&mut self) -> Result<&mut Session> {
        if self.encoder.is_none() {
            let session = build_shared_session(
                &self.env,
                &self.directory,
                "tok_encoder",
                self.gpu,
                self.threads,
                !self.gpu,
            )?;
            self.encoder = Some(session);
        }
        Ok(self.encoder.as_mut().unwrap())
    }

    fn id(&self, key: &str) -> Result<i64> {
        self.config[key]
            .as_i64()
            .or_else(|| self.config["talker_config"][key].as_i64())
            .with_context(|| format!("missing Qwen token: {key}"))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i64>> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        Ok(encoded.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    fn reference(&mut self, request: &NativeRequest) -> Result<ReferencePrompt> {
        let Some(path) = &request.reference_audio else {
            ensure!(
                request
                    .reference_text
                    .as_ref()
                    .is_none_or(|text| text.trim().is_empty()),
                "a reference transcript requires reference audio"
            );
            return Ok(ReferencePrompt {
                key: Vec::new(),
                speaker: Vec::new(),
                codes: Vec::new(),
            });
        };
        let bytes = fs::read(path).context("cannot read reference recording")?;
        let mut hash = Sha256::new();
        hash.update(&bytes);
        hash.update(
            request
                .reference_text
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        );
        let key = hash.finalize().to_vec();
        if let Some(reference) = &self.reference
            && reference.key == key
        {
            return Ok(reference.clone());
        }
        let wav = read_reference(path)?;
        let speaker_out = run(
            self.speaker_session()?,
            vec![("audio", tensor_f32(vec![1, wav.len()], wav.clone())?)],
        )?;
        let speaker = floats(&speaker_out[0])?;
        ensure!(
            speaker.len() == self.hidden,
            "speaker embedding has unexpected dimensions"
        );
        let mut codes = Vec::new();
        if request
            .reference_text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        {
            let mut all_codec_ids = Vec::new();
            for chunk in wav.chunks(SAMPLE_RATE) {
                let mut padded = vec![0.0; SAMPLE_RATE];
                padded[..chunk.len()].copy_from_slice(chunk);
                let encoded = run(
                    self.encoder_session()?,
                    vec![("audio", tensor_f32(vec![1, 1, SAMPLE_RATE], padded)?)],
                )?;
                let (_, data) = encoded[0].try_extract_tensor::<i64>()?;
                ensure!(data.len() % GROUPS == 0, "invalid reference audio codes");
                all_codec_ids.extend_from_slice(data);
            }
            if !all_codec_ids.is_empty() {
                let total_frames = all_codec_ids.len() / GROUPS;
                let output = run(
                    &mut self.residual_embed,
                    vec![(
                        "codec_ids",
                        tensor_i64(vec![total_frames, GROUPS], all_codec_ids)?,
                    )],
                )?;
                codes = floats(&output[0])?;
            }
        }
        let reference = ReferencePrompt {
            key,
            speaker,
            codes,
        };
        // One cached reference only: bounded memory, invalidated by audio/transcript changes.
        self.reference = Some(reference.clone());
        Ok(reference)
    }

    fn prompt(
        &mut self,
        request: &NativeRequest,
        text: &str,
        reference: &ReferencePrompt,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let ids = self.tokenize(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))?;
        ensure!(ids.len() >= 9, "text is too short to tokenize");
        let h = self.hidden;
        let language = crate::language::qwen_language(&request.language)?
            .name
            .to_lowercase();
        let language_id =
            self.config["talker_config"]["codec_language_id"][language.as_str()].as_i64();
        ensure!(
            language == "auto" || language_id.is_some(),
            "native model has no token for the selected language"
        );
        let prefix = if let Some(id) = language_id {
            vec![
                self.id("codec_think_id")?,
                self.id("codec_think_bos_id")?,
                id,
                self.id("codec_think_eos_id")?,
            ]
        } else {
            vec![
                self.id("codec_nothink_id")?,
                self.id("codec_think_bos_id")?,
                self.id("codec_think_eos_id")?,
            ]
        };
        let mut codec = embed(&mut self.codec_embed, "codec_ids", &prefix)?;
        codec.extend_from_slice(&reference.speaker);
        codec.extend_from_slice(&self.cached_embeddings.codec_pad);
        codec.extend_from_slice(&self.cached_embeddings.codec_bos);
        let mut prompt = embed(&mut self.text_embed, "text_ids", &ids[..3])?;
        let frames = codec.len() / h;
        for (i, row) in codec[..codec.len() - h].chunks_exact(h).enumerate() {
            let text_vec = if i == frames - 2 {
                &self.cached_embeddings.bos
            } else {
                &self.cached_embeddings.pad
            };
            prompt.extend(add(row, text_vec)?);
        }
        if reference.speaker.is_empty() {
            // Qwen's unconditioned prompt supplies text alongside generated
            // codec frames. Reference: Qwen3TTS.prepareGenerationInputs in
            // Blaizzy/mlx-audio-swift at 624fd5410b2fc8ea06f955500c9ac078e405261f.
            let body = embed(&mut self.text_embed, "text_ids", &ids[3..ids.len() - 5])?;
            ensure!(body.len() >= h, "text has no generation tokens");
            prompt.extend(add(&body[..h], &codec[codec.len() - h..])?);
            let mut trailing = body[h..].to_vec();
            trailing.extend_from_slice(&self.cached_embeddings.eos);
            return Ok((prompt, self.cached_embeddings.pad.clone(), trailing));
        }
        let mut body_ids = Vec::new();
        let icl = request
            .reference_text
            .as_deref()
            .filter(|text| !text.trim().is_empty());
        if let Some(transcript) = icl {
            let ids = self.tokenize(&format!("<|im_start|>assistant\n{transcript}<|im_end|>\n"))?;
            ensure!(ids.len() >= 5, "invalid reference transcript");
            body_ids.extend_from_slice(&ids[3..ids.len() - 2]);
        }
        body_ids.extend_from_slice(&ids[3..ids.len() - 5]);
        let mut body = embed(&mut self.text_embed, "text_ids", &body_ids)?;
        body.extend_from_slice(&self.cached_embeddings.eos);
        for row in body.chunks_exact(h) {
            prompt.extend(add(row, &self.cached_embeddings.codec_pad)?);
        }
        prompt.extend(add(
            &self.cached_embeddings.codec_bos,
            &self.cached_embeddings.pad,
        )?);
        if icl.is_some() {
            for row in reference.codes.chunks_exact(h) {
                prompt.extend(add(row, &self.cached_embeddings.pad)?);
            }
        }
        Ok((prompt, self.cached_embeddings.pad.clone(), Vec::new()))
    }

    fn generate_and_stream<F>(
        &mut self,
        request: &NativeRequest,
        text: &str,
        reference: &ReferencePrompt,
        rng: &mut StdRng,
        mut on_audio: F,
    ) -> Result<()>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let (input, pad, trailing_text) = self.prompt(request, text, reference)?;
        #[cfg(debug_assertions)]
        trace(
            "native-prompt.json",
            &serde_json::json!({"prompt": input, "pad": pad, "speaker": reference.speaker, "reference": reference.codes, "ids": self.tokenize(&format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"))?}),
        )?;
        let h = self.hidden;
        let prefill_length = input.len() / h;
        let mut current = input;
        let mut past = (0..self.layers * 2)
            .map(|_| tensor_f32(vec![1, self.kv_heads, 0, self.head_dim], Vec::new()))
            .collect::<Result<Vec<_>>>()?;
        let mut pending_codes = Vec::with_capacity(DECODE_FRAMES * GROUPS);
        let mut generated_frames = 0usize;
        #[cfg(debug_assertions)]
        let mut trace_codes = Vec::new();
        let mut previous = HashSet::new();
        let eos = self.id("codec_eos_token_id")? as usize;
        let vocab = self.config["talker_config"]["vocab_size"]
            .as_u64()
            .context("missing codec vocabulary")? as usize;
        ensure!(
            vocab > 1024 && vocab <= 65536 && eos < vocab,
            "invalid Qwen codec vocabulary"
        );

        let (decode_tx, decode_rx) = mpsc::channel::<Result<Vec<f32>>>();
        let mut in_flight_decodes = 0usize;
        let mut first_logits = vec![0.0f32; vocab];

        for step in 0..request.max_new_tokens {
            // Drain any completed pipelined audio chunks non-blockingly so
            // streaming playback receives audio as soon as the E-core decoder
            // finishes a 25-frame block.
            while in_flight_decodes > 0 {
                match decode_rx.try_recv() {
                    Ok(decoded) => {
                        in_flight_decodes -= 1;
                        let samples = decoded?;
                        on_audio(&samples)?;
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        bail!("pipelined audio decoder disconnected");
                    }
                }
            }

            let total = prefill_length + step;
            let cur = current.len() / h;
            let positions: Vec<i64> = if cur == 1 {
                let p = (total - 1) as i64;
                vec![p, p, p]
            } else {
                (0..3)
                    .flat_map(|_| ((total - cur)..total).map(|p| p as i64))
                    .collect()
            };
            let mut inputs = vec![
                ("inputs_embeds", tensor_f32(vec![1, cur, h], current)?),
                ("position_ids", tensor_i64(vec![3, 1, cur], positions)?),
                (
                    "attention_mask",
                    tensor_i64(vec![1, total], vec![1; total])?,
                ),
            ];
            for (name, value) in self.past_names.iter().zip(past) {
                inputs.push((name.as_str(), value));
            }
            let outputs = if self.device.starts_with("cuda") {
                let mut binding = self.talker.create_binding()?;
                for (name, value) in &inputs {
                    binding.bind_input(*name, value)?;
                }
                for (index, output) in self.talker.outputs().iter().enumerate() {
                    let device = if index < 2 {
                        AllocationDevice::CPU
                    } else {
                        AllocationDevice::CUDA
                    };
                    binding.bind_output_to_device(
                        output.name(),
                        &MemoryInfo::new(device, 0, AllocatorType::Device, MemoryType::Default)?,
                    )?;
                }
                self.talker
                    .run_binding(&binding)?
                    .into_iter()
                    .map(|(_, value)| value)
                    .collect()
            } else {
                run(&mut self.talker, inputs)?
            };
            let mut out = outputs.into_iter();
            let logits = out.next().context("missing Qwen logits")?;
            let hidden = out.next().context("missing Qwen hidden state")?;
            past = out.collect();
            let (_, logits) = logits.try_extract_tensor::<f32>()?;
            ensure!(logits.len() >= vocab, "invalid Qwen logits");
            first_logits.copy_from_slice(&logits[logits.len() - vocab..]);
            let eos_logit = first_logits[eos];
            first_logits[vocab - 1024..].fill(f32::NEG_INFINITY);
            first_logits[eos] = eos_logit;
            let penalty = request.sampling.repetition_penalty;
            if (penalty - 1.0).abs() > f32::EPSILON {
                for &id in &previous {
                    let logit = &mut first_logits[id];
                    *logit = if *logit < 0.0 {
                        *logit * penalty
                    } else {
                        *logit / penalty
                    };
                }
            }
            let code0 = sample_with_scratch(
                &first_logits,
                request.do_sample,
                &request.sampling,
                rng,
                &mut self.sample_candidates,
                &mut self.sample_weights,
            )?;
            if code0 == eos {
                ensure!(generated_frames > 0, "Qwen produced no speech");
                if !pending_codes.is_empty() {
                    match &mut self.decoder {
                        DecoderBackend::Pipelined(worker) => {
                            worker.enqueue(std::mem::take(&mut pending_codes), decode_tx.clone())?;
                            in_flight_decodes += 1;
                        }
                        DecoderBackend::Inline(session) => {
                            let samples = decode_chunk_session(session, &pending_codes)?;
                            pending_codes.clear();
                            on_audio(&samples)?;
                        }
                    }
                }
                while in_flight_decodes > 0 {
                    let decoded = decode_rx
                        .recv()
                        .map_err(|_| anyhow::anyhow!("pipelined audio decoder exited"))??;
                    in_flight_decodes -= 1;
                    on_audio(&decoded)?;
                }
                return Ok(());
            }
            previous.insert(code0);
            let (_, state) = hidden.try_extract_tensor::<f32>()?;
            ensure!(state.len() >= h, "invalid Qwen hidden state");
            let state = &state[state.len() - h..];
            let frame: [i64; GROUPS] = match &mut self.predictor {
                PredictorBackend::Fast(fast) => {
                    let do_sample = request.do_sample;
                    let sampling = &request.sampling;
                    let candidates = &mut self.sample_candidates;
                    let weights = &mut self.sample_weights;
                    fast.predict_frame(state, code0 as i64, |group_logits| {
                        sample_with_scratch(
                            group_logits,
                            do_sample,
                            sampling,
                            rng,
                            candidates,
                            weights,
                        )
                    })?
                }
                PredictorBackend::Ort(predictor) => {
                    let mut frame = [0i64; GROUPS];
                    frame[0] = code0 as i64;
                    for group in 1..GROUPS {
                        let predicted = run(
                            predictor,
                            vec![
                                ("talker_hidden", tensor_f32(vec![1, h], state.to_vec())?),
                                ("codec_ids", tensor_i64(vec![1, GROUPS], frame.to_vec())?),
                            ],
                        )?;
                        let (shape, logits) = predicted[0].try_extract_tensor::<f32>()?;
                        let vocabulary =
                            *shape.last().context("invalid predictor shape")? as usize;
                        let start = (group - 1) * vocabulary;
                        ensure!(
                            start + vocabulary <= logits.len(),
                            "invalid predictor logits"
                        );
                        frame[group] = sample_with_scratch(
                            &logits[start..start + vocabulary],
                            request.do_sample,
                            &request.sampling,
                            rng,
                            &mut self.sample_candidates,
                            &mut self.sample_weights,
                        )? as i64;
                    }
                    frame
                }
            };
            pending_codes.extend_from_slice(&frame);
            generated_frames += 1;
            #[cfg(debug_assertions)]
            if step < 3 {
                trace_codes.extend_from_slice(&frame);
                trace("native-codes.json", &trace_codes)?;
            }
            if pending_codes.len() == DECODE_FRAMES * GROUPS {
                match &mut self.decoder {
                    DecoderBackend::Pipelined(worker) => {
                        let chunk = std::mem::replace(
                            &mut pending_codes,
                            Vec::with_capacity(DECODE_FRAMES * GROUPS),
                        );
                        worker.enqueue(chunk, decode_tx.clone())?;
                        in_flight_decodes += 1;
                    }
                    DecoderBackend::Inline(session) => {
                        let samples = decode_chunk_session(session, &pending_codes)?;
                        pending_codes.clear();
                        on_audio(&samples)?;
                    }
                }
            }
            let output = run(
                &mut self.residual_embed,
                vec![("codec_ids", tensor_i64(vec![1, GROUPS], frame.to_vec())?)],
            )?;
            let text = trailing_text.get(step * h..(step + 1) * h).unwrap_or(&pad);
            current = add(&floats(&output[0])?, text)?;
            if (step + 1) % 50 == 0 {
                eprintln!("Native Qwen: {} audio frames", step + 1);
            }
        }
        bail!("Qwen reached its generation limit before finishing; use a shorter text chunk")
    }

    pub fn synthesize(&mut self, request: &NativeRequest) -> Result<()> {
        self.synthesize_streaming(request, &mut |_, _| Ok(()))
    }

    pub fn synthesize_streaming(
        &mut self,
        request: &NativeRequest,
        sink: &mut crate::AudioSink<'_>,
    ) -> Result<()> {
        ensure!(!request.text.trim().is_empty(), "text must not be empty");
        crate::language::qwen_language(&request.language)?;
        request.sampling.validate()?;
        ensure!(
            (1..=2048).contains(&request.max_new_tokens),
            "generation limit must be between 1 and 2048"
        );
        ensure!(
            request
                .reference_text
                .as_ref()
                .is_none_or(|text| text.chars().count() <= 2000),
            "reference transcript exceeds 2000 characters"
        );
        // Pin the caller thread to the primary performance core for the
        // duration of synthesis so it does not compete with ORT / predictor
        // worker threads pinned to gen_cpus[1..].
        let _affinity_guard = self
            .gen_cpus
            .first()
            .copied()
            .and_then(|cpu| cpu_topology::pin_current_thread(&[cpu]));
        let start = Instant::now();
        let reference = self.reference(request)?;
        let mut rng = request
            .seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_os_rng);
        if let Some(parent) = request
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let mut output = hound::WavWriter::create(
            &request.output,
            hound::WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE as u32,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for text in text_chunks(&request.text, 160) {
            self.generate_and_stream(request, &text, &reference, &mut rng, |samples| {
                for chunk in samples.chunks(4096) {
                    for &sample in chunk {
                        output
                            .write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
                    }
                    sink(chunk, SAMPLE_RATE as u32)?;
                }
                Ok(())
            })?;
        }
        output.finalize()?;
        eprintln!(
            "Native Qwen synthesis completed in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }
}

// Opt-in local diagnostics for comparing this port with the published ONNX reference.
// Release builds never write intermediate voice tensors.
#[cfg(debug_assertions)]
fn trace(name: &str, value: &impl serde::Serialize) -> Result<()> {
    if let Some(directory) = std::env::var_os("SAYIT_QWEN_TRACE_DIR") {
        let directory = PathBuf::from(directory);
        fs::create_dir_all(&directory)?;
        fs::write(directory.join(name), serde_json::to_vec(value)?)?;
    }
    Ok(())
}

pub fn load_tokenizer(model: &Path) -> Result<Tokenizer> {
    let bpe = BPE::from_file(
        &model.join("vocab.json").to_string_lossy(),
        &model.join("merges.txt").to_string_lossy(),
    )
    .build()
    .map_err(|e| anyhow::anyhow!("cannot load Qwen BPE: {e}"))?;
    let mut tokenizer = Tokenizer::new(bpe);
    // Qwen2Tokenizer's exact Unicode regex; ByteLevel handles the byte alphabet only.
    let split = Split::new(SplitPattern::Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+".into()), SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| anyhow::anyhow!("invalid Qwen tokenization pattern: {e}"))?;
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, true, false).into(),
    ])));
    let config: Value = serde_json::from_slice(&fs::read(model.join("tokenizer_config.json"))?)?;
    let mut added = config["added_tokens_decoder"]
        .as_object()
        .context("Qwen tokenizer has no added tokens")?
        .iter()
        .map(|(id, value)| Ok((id.parse::<u32>()?, value)))
        .collect::<Result<Vec<_>>>()?;
    added.sort_by_key(|(id, _)| *id);
    for (id, token) in added {
        let content = token["content"].as_str().context("invalid added token")?;
        let added = AddedToken::from(content, token["special"].as_bool().unwrap_or(false))
            .normalized(token["normalized"].as_bool().unwrap_or(false))
            .single_word(token["single_word"].as_bool().unwrap_or(false))
            .lstrip(token["lstrip"].as_bool().unwrap_or(false))
            .rstrip(token["rstrip"].as_bool().unwrap_or(false));
        tokenizer.add_tokens(&[added]);
        ensure!(
            tokenizer.token_to_id(content) == Some(id),
            "Qwen tokenizer token ID mismatch for {content}"
        );
    }
    Ok(tokenizer)
}

fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    ensure!(a.len() == b.len(), "Qwen embedding dimensions do not match");
    Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
}

#[cfg(test)]
fn sample(logits: &[f32], do_sample: bool, rng: &mut StdRng) -> Result<usize> {
    sample_with_options(
        logits,
        do_sample,
        &crate::QwenSamplingOptions::default(),
        rng,
    )
}

#[cfg(test)]
fn sample_with_options(
    logits: &[f32],
    do_sample: bool,
    options: &crate::QwenSamplingOptions,
    rng: &mut StdRng,
) -> Result<usize> {
    let mut candidates = Vec::new();
    let mut weights = Vec::new();
    sample_with_scratch(logits, do_sample, options, rng, &mut candidates, &mut weights)
}

fn sample_with_scratch(
    logits: &[f32],
    do_sample: bool,
    options: &crate::QwenSamplingOptions,
    rng: &mut StdRng,
    candidates: &mut Vec<(usize, f32)>,
    weights: &mut Vec<f64>,
) -> Result<usize> {
    if !do_sample {
        let mut best: Option<(usize, f32)> = None;
        for (idx, &v) in logits.iter().enumerate() {
            if v.is_finite()
                && best.is_none_or(|(_, bv)| v.total_cmp(&bv).is_gt())
            {
                best = Some((idx, v));
            }
        }
        return best
            .map(|(idx, _)| idx)
            .context("model produced no finite token logits");
    }

    candidates.clear();
    candidates.extend(
        logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite()),
    );
    ensure!(
        !candidates.is_empty(),
        "model produced no finite token logits"
    );

    if options.top_k > 0 && options.top_k < candidates.len() {
        let k = options.top_k;
        candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let maximum = candidates[0].1;
    let inv_temp = 1.0f32 / options.temperature;
    weights.clear();
    let mut total_weight = 0.0f64;
    for &(_, v) in candidates.iter() {
        let w = f64::from((v - maximum) * inv_temp).exp();
        weights.push(w);
        total_weight += w;
    }
    if options.top_p < 1.0 {
        let limit = f64::from(options.top_p) * total_weight;
        let mut accumulated = 0.0;
        let keep = weights
            .iter()
            .position(|weight| {
                accumulated += weight;
                accumulated >= limit
            })
            .map_or(weights.len(), |index| index + 1);
        weights.truncate(keep);
        candidates.truncate(keep);
        total_weight = weights.iter().sum();
    }
    let mut target = rng.random::<f64>() * total_weight;
    for ((id, _), &weight) in candidates.iter().zip(weights.iter()) {
        target -= weight;
        if target <= 0.0 {
            return Ok(*id);
        }
    }
    Ok(candidates.last().unwrap().0)
}

fn read_reference(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.channels > 0 && spec.sample_rate > 0,
        "invalid reference WAV"
    );
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|s| s as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let mono = samples
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
        .collect::<Vec<_>>();
    let duration = mono.len() as f64 / f64::from(spec.sample_rate);
    ensure!(
        (3.0..=45.0).contains(&duration),
        "reference recording must contain 3–45 seconds of speech"
    );
    ensure!(
        mono.iter().all(|v| v.is_finite()),
        "invalid reference audio samples"
    );
    Ok(resample(&mono, spec.sample_rate as usize, SAMPLE_RATE))
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn resample_point(input: &[f32], position: f64, cutoff: f64) -> f32 {
    let center = position.floor() as isize;
    let (mut sum, mut norm) = (0.0, 0.0);
    for j in center - 32..=center + 32 {
        if j < 0 || j as usize >= input.len() {
            continue;
        }
        let distance = position - j as f64;
        if distance.abs() > 32.0 {
            continue;
        }
        let x = std::f64::consts::PI * distance * cutoff;
        let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
        let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / 32.0).cos();
        let weight = sinc * window * cutoff;
        sum += f64::from(input[j as usize]) * weight;
        norm += weight;
    }
    (sum / norm) as f32
}

fn resample(input: &[f32], source: usize, target: usize) -> Vec<f32> {
    if source == target || input.is_empty() {
        return input.to_vec();
    }
    let count = input.len() * target / source;
    let cutoff = (target as f64 / source as f64).min(1.0) * 0.95;

    // Polyphase precomputed Hann-windowed sinc table for standard rational
    // sample-rate ratios (covers 8k/11.025k/16k/22.05k/32k/44.1k/48k/96k -> 24k).
    let g = gcd(source, target);
    let phases = target / g;
    const HALF_TAPS: usize = 32;
    const TAPS: usize = HALF_TAPS * 2 + 1; // 65
    if phases <= 1024 && input.len() > TAPS {
        let step_num = source / g;
        let mut table = vec![0.0f32; phases * TAPS];
        for phase in 0..phases {
            let frac = ((phase * step_num) % phases) as f64 / phases as f64;
            let row = &mut table[phase * TAPS..(phase + 1) * TAPS];
            let mut weights = [0.0f64; TAPS];
            let mut norm = 0.0f64;
            for (k, w) in weights.iter_mut().enumerate() {
                let offset = k as isize - HALF_TAPS as isize;
                let distance = frac - offset as f64;
                if distance.abs() <= 32.0 {
                    let x = std::f64::consts::PI * distance * cutoff;
                    let sinc = if x.abs() < 1e-12 { 1.0 } else { x.sin() / x };
                    let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / 32.0).cos();
                    let weight = sinc * window * cutoff;
                    *w = weight;
                    norm += weight;
                }
            }
            let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for (dst, &w) in row.iter_mut().zip(&weights) {
                *dst = (w * inv_norm) as f32;
            }
        }
        let mut output = Vec::with_capacity(count);
        let max_center = input.len() - 1 - HALF_TAPS;
        for i in 0..count {
            let center = (i * step_num) / phases;
            if center >= HALF_TAPS && center <= max_center {
                let phase = i % phases;
                let row = &table[phase * TAPS..(phase + 1) * TAPS];
                let slice = &input[center - HALF_TAPS..center + HALF_TAPS + 1];
                let mut sum = 0.0f32;
                for (&s, &w) in slice.iter().zip(row) {
                    sum = s.mul_add(w, sum);
                }
                output.push(sum);
            } else {
                let position = i as f64 * source as f64 / target as f64;
                output.push(resample_point(input, position, cutoff));
            }
        }
        return output;
    }

    (0..count)
        .map(|i| {
            let position = i as f64 * source as f64 / target as f64;
            resample_point(input, position, cutoff)
        })
        .collect()
}

fn text_chunks(text: &str, limit: usize) -> Vec<String> {
    let mut remaining = text.trim();
    let mut result = Vec::new();
    while !remaining.is_empty() {
        let Some((end, _)) = remaining.char_indices().nth(limit) else {
            result.push(remaining.to_owned());
            break;
        };
        let prefix = &remaining[..end];
        let boundary = prefix
            .char_indices()
            .filter(|(_, c)| matches!(c, '.' | '!' | '?' | ';' | '。' | '！' | '？' | '；' | '\n'))
            .map(|(i, c)| i + c.len_utf8())
            .next_back()
            .or_else(|| {
                prefix
                    .char_indices()
                    .filter(|(_, c)| c.is_whitespace())
                    .map(|(i, _)| i)
                    .next_back()
            })
            .filter(|&i| i > 0)
            .unwrap_or(end);
        let (chunk, rest) = remaining.split_at(boundary);
        result.push(chunk.trim().to_owned());
        remaining = rest.trim();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires real native Qwen assets/runtime through SAYIT_ONNX_TEST_MODEL and SAYIT_ONNX_TEST_RUNTIME"]
    fn free_voice_sampling_is_repeatable_and_isolated_from_saved_references() {
        let model = PathBuf::from(std::env::var_os("SAYIT_ONNX_TEST_MODEL").unwrap());
        let runtime = PathBuf::from(std::env::var_os("SAYIT_ONNX_TEST_RUNTIME").unwrap());
        let mut pipeline = NativeQwenPipeline::load(&model, &runtime, "int4", "cpu", 4).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let request = NativeRequest {
            text: "Every voice gives familiar words a new color, character, and rhythm.".into(),
            output: directory.path().join("first.wav"),
            reference_audio: None,
            reference_text: None,
            language: "English".into(),
            seed: Some(417),
            max_new_tokens: 256,
            do_sample: true,
            sampling: crate::QwenSamplingOptions::default(),
            stream: false,
            stream_format: None,
        };
        let mut streamed = Vec::new();
        pipeline
            .synthesize_streaming(&request, &mut |pcm, rate| {
                assert_eq!(rate, 24_000);
                assert!(pcm.iter().all(|sample| sample.is_finite()));
                streamed.extend_from_slice(pcm);
                Ok(())
            })
            .unwrap();
        assert!(streamed.iter().any(|sample| sample.abs() > 0.001));
        assert_eq!(
            hound::WavReader::open(&request.output).unwrap().len() as usize,
            streamed.len()
        );
        let cloned = NativeRequest {
            text: "This recording uses the generated voice for new words.".into(),
            reference_audio: Some(request.output.clone()),
            reference_text: Some(request.text.clone()),
            output: directory.path().join("cloned.wav"),
            ..request.clone()
        };
        pipeline.synthesize(&cloned).unwrap();
        assert!(crate::wav_duration_seconds(&cloned.output).unwrap() > 0.5);
        let repeat = NativeRequest {
            output: directory.path().join("repeat.wav"),
            ..request.clone()
        };
        pipeline.synthesize(&repeat).unwrap();
        assert_eq!(
            fs::read(&request.output).unwrap(),
            fs::read(&repeat.output).unwrap(),
            "reference conditioning leaked into free generation or seeded generation was not repeatable"
        );
        let different = NativeRequest {
            output: directory.path().join("different.wav"),
            seed: Some(418),
            ..request.clone()
        };
        pipeline.synthesize(&different).unwrap();
        assert_ne!(
            fs::read(&request.output).unwrap(),
            fs::read(&different.output).unwrap(),
            "different seeds must sample different audio"
        );
        let mut speakers = Vec::new();
        for output in [&request.output, &different.output] {
            let reference = NativeRequest {
                reference_audio: Some(output.clone()),
                ..request.clone()
            };
            speakers.push(pipeline.reference(&reference).unwrap().speaker);
        }
        let dot = speakers[0]
            .iter()
            .zip(&speakers[1])
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>();
        let norm = |speaker: &[f32]| {
            speaker
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let cosine = dot / (norm(&speakers[0]) * norm(&speakers[1]));
        assert!(cosine.is_finite());
        eprintln!(
            "Free Qwen candidates: {} source samples, speaker cosine {cosine:.4}",
            streamed.len()
        );
    }

    #[test]
    #[ignore = "requires pinned native model/runtime assets and a permitted reference; set SAYIT_QWEN_TEST_ROOT"]
    fn native_reference_encoder_preserves_every_supported_recording_length() {
        let root =
            PathBuf::from(std::env::var_os("SAYIT_QWEN_TEST_ROOT").expect("SAYIT_QWEN_TEST_ROOT"));
        let runtime = PathBuf::from(
            std::env::var_os("SAYIT_ONNX_TEST_RUNTIME").expect("SAYIT_ONNX_TEST_RUNTIME"),
        );
        let mut pipeline = NativeQwenPipeline::load(&root, &runtime, "int4", "cpu", 4).unwrap();
        let examples: Vec<Value> =
            serde_json::from_str(include_str!("../tests/data/qwen-tokenizer.json")).unwrap();
        for example in examples {
            let text = example["text"].as_str().unwrap();
            let expected: Vec<i64> = serde_json::from_value(example["ids"].clone()).unwrap();
            assert_eq!(
                pipeline.tokenize(text).unwrap(),
                expected,
                "tokenizer differs for {text:?}"
            );
        }
        let mut request: NativeRequest =
            serde_json::from_slice(&fs::read(root.join("request.json")).unwrap()).unwrap();
        assert!(
            request
                .reference_text
                .as_ref()
                .is_some_and(|text| !text.is_empty())
        );
        let source = read_reference(request.reference_audio.as_ref().unwrap()).unwrap();
        let encoded = run(
            pipeline.encoder_session().unwrap(),
            vec![(
                "audio",
                tensor_f32(vec![1, 1, SAMPLE_RATE], source[..SAMPLE_RATE].to_vec()).unwrap(),
            )],
        )
        .unwrap();
        let frames_per_window = encoded[0].try_extract_tensor::<i64>().unwrap().1.len() / GROUPS;
        let directory = tempfile::tempdir().unwrap();
        for seconds in [3, 15, 30, 45] {
            let path = directory.path().join(format!("{seconds}.wav"));
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: SAMPLE_RATE as u32,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for index in 0..seconds * SAMPLE_RATE {
                writer.write_sample(source[index % source.len()]).unwrap();
            }
            writer.finalize().unwrap();
            request.reference_audio = Some(path);
            let prompt = pipeline.reference(&request).unwrap();
            assert_eq!(
                prompt.codes.len() / pipeline.hidden,
                seconds * frames_per_window
            );
            assert_eq!(prompt.speaker.len(), pipeline.hidden);
            assert_eq!(pipeline.reference(&request).unwrap().key, prompt.key);
        }
    }
    #[test]
    fn resampling_preserves_duration_and_dc() {
        for (src, dst) in [(16_000, 24_000), (44_100, 24_000), (48_000, 24_000)] {
            let output = resample(&vec![0.25; src], src, dst);
            assert_eq!(output.len(), dst);
            assert!(output.iter().all(|v| (v - 0.25).abs() < 1e-5));
        }
    }
    #[test]
    fn chunks_preserve_unicode_and_sentence_boundaries() {
        let text = "Hello there. 今日は良い天気ですね。 Another sentence follows.";
        let chunks = text_chunks(text, 25);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 25));
        assert_eq!(chunks.join(" ").replace(' ', ""), text.replace(' ', ""));
    }
    #[test]
    fn sampling_never_selects_suppressed_codes() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..100 {
            assert_eq!(
                sample(&[f32::NEG_INFINITY, 4.0, f32::NAN], true, &mut rng).unwrap(),
                1
            );
        }
        assert!(sample(&[f32::NEG_INFINITY], true, &mut rng).is_err());
    }

    #[test]
    fn top_k_and_nucleus_controls_keep_sampling_inside_the_selected_distribution() {
        let mut rng = StdRng::seed_from_u64(10);
        for options in [
            crate::QwenSamplingOptions {
                top_k: 1,
                ..Default::default()
            },
            crate::QwenSamplingOptions {
                top_k: 0,
                top_p: 0.5,
                ..Default::default()
            },
        ] {
            for _ in 0..100 {
                assert_eq!(
                    sample_with_options(&[8.0, 1.0, f32::NEG_INFINITY], true, &options, &mut rng)
                        .unwrap(),
                    0
                );
            }
        }
        for options in [
            crate::QwenSamplingOptions {
                temperature: f32::NAN,
                ..Default::default()
            },
            crate::QwenSamplingOptions {
                top_p: 1.1,
                ..Default::default()
            },
            crate::QwenSamplingOptions {
                top_k: 101,
                ..Default::default()
            },
            crate::QwenSamplingOptions {
                repetition_penalty: 0.0,
                ..Default::default()
            },
        ] {
            assert!(options.validate().is_err());
        }
    }
}
