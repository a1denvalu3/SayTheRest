//! Native execution of the pinned Qwen3-TTS Base ONNX component graphs.
//! The prompt layout follows QwenLM/Qwen3-TTS and ONNX Community's inference.py.
//! No Python code or remote tokenizer code is executed by this module.

use anyhow::{Context, Result, bail, ensure};
use ort::{
    environment::Environment,
    ep,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
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

#[derive(Deserialize)]
pub struct NativeRequest {
    pub text: String,
    pub output: PathBuf,
    pub reference_audio: PathBuf,
    pub reference_text: Option<String>,
    #[serde(default = "auto_language")]
    pub language: String,
    pub seed: Option<u64>,
    #[serde(default = "max_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "sampling")]
    pub do_sample: bool,
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

pub struct NativeQwenPipeline {
    text_embed: Session,
    codec_embed: Session,
    residual_embed: Session,
    talker: Session,
    predictor: Session,
    encoder: Session,
    decoder: Session,
    speaker: Session,
    tokenizer: Tokenizer,
    config: Value,
    hidden: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    past_names: Vec<String>,
    device: String,
    reference: Option<ReferencePrompt>,
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
        ensure!(
            threads > 0 && threads <= 64,
            "inference threads must be between 1 and 64"
        );
        let env: Environment = ort::init_from(runtime)?
            .with_name("sayit-qwen-onnx")
            .with_telemetry(false)
            .build()?;
        let gpu = device.starts_with("cuda");
        let directory = model.join(precision);
        let load = |name: &str| -> Result<Session> {
            let mut builder = Session::builder(&env)?
                .with_intra_threads(threads)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_inter_threads(1)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .with_intra_op_spinning(false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if gpu {
                builder = builder
                    .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            builder
                .commit_from_file(directory.join(format!("{name}.onnx")))
                .with_context(|| format!("cannot load native Qwen {name}"))
        };
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
        let talker = load("talker_cache")?;
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
        let tokenizer = load_tokenizer(model)?;
        Ok(Self {
            text_embed: load("text_embed")?,
            codec_embed: load("codec_embed")?,
            residual_embed: load("residual_embed")?,
            talker,
            predictor: load("code_predictor")?,
            encoder: load("tok_encoder")?,
            decoder: load("tok_decoder")?,
            speaker: load("speaker_encoder")?,
            tokenizer,
            config,
            hidden,
            layers,
            kv_heads,
            head_dim,
            past_names,
            device: if gpu { "cuda:0" } else { "cpu" }.into(),
            reference: None,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
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
        let bytes =
            fs::read(&request.reference_audio).context("cannot read reference recording")?;
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
        let wav = read_reference(&request.reference_audio)?;
        let speaker = floats(
            &run(
                &mut self.speaker,
                vec![("audio", tensor_f32(vec![1, wav.len()], wav.clone())?)],
            )?[0],
        )?;
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
            for chunk in wav.chunks(SAMPLE_RATE) {
                let mut padded = vec![0.0; SAMPLE_RATE];
                padded[..chunk.len()].copy_from_slice(chunk);
                let encoded = run(
                    &mut self.encoder,
                    vec![("audio", tensor_f32(vec![1, 1, SAMPLE_RATE], padded)?)],
                )?;
                let (_, data) = encoded[0].try_extract_tensor::<i64>()?;
                ensure!(data.len() % GROUPS == 0, "invalid reference audio codes");
                let frames = data.len() / GROUPS;
                let output = run(
                    &mut self.residual_embed,
                    vec![(
                        "codec_ids",
                        tensor_i64(vec![frames, GROUPS], data.to_vec())?,
                    )],
                )?;
                codes.extend(floats(&output[0])?);
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
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let ids = self.tokenize(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))?;
        ensure!(ids.len() >= 9, "text is too short to tokenize");
        let special = [
            self.id("tts_bos_token_id")?,
            self.id("tts_eos_token_id")?,
            self.id("tts_pad_token_id")?,
        ];
        let special = embed(&mut self.text_embed, "text_ids", &special)?;
        let h = self.hidden;
        let (bos, rest) = special.split_at(h);
        let (eos, pad) = rest.split_at(h);
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
        let codec_pad_id = self.id("codec_pad_id")?;
        let codec_bos_id = self.id("codec_bos_id")?;
        codec.extend(embed(
            &mut self.codec_embed,
            "codec_ids",
            &[codec_pad_id, codec_bos_id],
        )?);
        let mut prompt = embed(&mut self.text_embed, "text_ids", &ids[..3])?;
        let frames = codec.len() / h;
        for (i, row) in codec[..codec.len() - h].chunks_exact(h).enumerate() {
            prompt.extend(add(row, if i == frames - 2 { bos } else { pad })?);
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
        body.extend_from_slice(eos);
        let codec_pad = embed(&mut self.codec_embed, "codec_ids", &[codec_pad_id])?;
        for row in body.chunks_exact(h) {
            prompt.extend(add(row, &codec_pad)?);
        }
        let codec_bos = embed(&mut self.codec_embed, "codec_ids", &[codec_bos_id])?;
        prompt.extend(add(&codec_bos, pad)?);
        if icl.is_some() {
            for row in reference.codes.chunks_exact(h) {
                prompt.extend(add(row, pad)?);
            }
        }
        Ok((prompt, pad.to_vec()))
    }

    fn generate(
        &mut self,
        request: &NativeRequest,
        text: &str,
        reference: &ReferencePrompt,
        rng: &mut StdRng,
    ) -> Result<Vec<i64>> {
        let (input, pad) = self.prompt(request, text, reference)?;
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
        let mut codes = Vec::new();
        let mut previous = HashSet::new();
        let eos = self.id("codec_eos_token_id")? as usize;
        let vocab = self.config["talker_config"]["vocab_size"]
            .as_u64()
            .context("missing codec vocabulary")? as usize;
        ensure!(
            vocab > 1024 && vocab <= 65536 && eos < vocab,
            "invalid Qwen codec vocabulary"
        );
        for step in 0..request.max_new_tokens {
            let total = prefill_length + step;
            let cur = current.len() / h;
            let positions = (0..3)
                .flat_map(|_| ((total - cur)..total).map(|p| p as i64))
                .collect();
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
            let mut first = logits[logits.len() - vocab..].to_vec();
            for (id, logit) in first.iter_mut().enumerate() {
                if id >= vocab - 1024 && id != eos {
                    *logit = f32::NEG_INFINITY;
                }
                if previous.contains(&id) {
                    *logit = if *logit < 0.0 {
                        *logit * 1.05
                    } else {
                        *logit / 1.05
                    };
                }
            }
            let code0 = sample(&first, request.do_sample, rng)?;
            if code0 == eos {
                ensure!(!codes.is_empty(), "Qwen produced no speech");
                return Ok(codes);
            }
            previous.insert(code0);
            let (_, state) = hidden.try_extract_tensor::<f32>()?;
            ensure!(state.len() >= h, "invalid Qwen hidden state");
            let state = &state[state.len() - h..];
            let mut frame = vec![0; GROUPS];
            frame[0] = code0 as i64;
            for group in 1..GROUPS {
                let predicted = run(
                    &mut self.predictor,
                    vec![
                        ("talker_hidden", tensor_f32(vec![1, h], state.to_vec())?),
                        ("codec_ids", tensor_i64(vec![1, GROUPS], frame.clone())?),
                    ],
                )?;
                let (shape, logits) = predicted[0].try_extract_tensor::<f32>()?;
                let vocabulary = *shape.last().context("invalid predictor shape")? as usize;
                let start = (group - 1) * vocabulary;
                ensure!(
                    start + vocabulary <= logits.len(),
                    "invalid predictor logits"
                );
                frame[group] =
                    sample(&logits[start..start + vocabulary], request.do_sample, rng)? as i64;
            }
            codes.extend_from_slice(&frame);
            #[cfg(debug_assertions)]
            if step < 3 {
                trace("native-codes.json", &codes)?;
            }
            let output = run(
                &mut self.residual_embed,
                vec![("codec_ids", tensor_i64(vec![1, GROUPS], frame)?)],
            )?;
            current = add(&floats(&output[0])?, &pad)?;
            if (step + 1) % 50 == 0 {
                eprintln!("Native Qwen: {} audio frames", step + 1);
            }
        }
        bail!("Qwen reached its generation limit before finishing; use a shorter text chunk")
    }

    fn decode(&mut self, codes: &[i64]) -> Result<Vec<f32>> {
        ensure!(
            !codes.is_empty() && codes.len().is_multiple_of(GROUPS),
            "invalid generated audio codes"
        );
        let mut wav = Vec::new();
        for chunk in codes.chunks(DECODE_FRAMES * GROUPS) {
            let frames = chunk.len() / GROUPS;
            let padded = (0..DECODE_FRAMES * GROUPS)
                .map(|i| chunk[i % chunk.len()])
                .collect();
            let output = run(
                &mut self.decoder,
                vec![(
                    "audio_codes",
                    tensor_i64(vec![1, DECODE_FRAMES, GROUPS], padded)?,
                )],
            )?;
            let samples = output[0].try_extract_tensor::<f32>()?.1;
            let count = (samples.len() * frames + DECODE_FRAMES / 2) / DECODE_FRAMES;
            wav.extend_from_slice(&samples[..count]);
        }
        ensure!(
            wav.iter().all(|sample| sample.is_finite()),
            "decoder generated invalid samples"
        );
        Ok(wav)
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
            let codes = self.generate(request, &text, &reference, &mut rng)?;
            let samples = self.decode(&codes)?;
            for chunk in samples.chunks(4096) {
                for &sample in chunk {
                    output.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
                }
                sink(chunk, SAMPLE_RATE as u32)?;
            }
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

fn sample(logits: &[f32], do_sample: bool, rng: &mut StdRng) -> Result<usize> {
    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .collect::<Vec<_>>();
    ensure!(
        !candidates.is_empty(),
        "model produced no finite token logits"
    );
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if !do_sample {
        return Ok(candidates[0].0);
    }
    candidates.truncate(50);
    let maximum = candidates[0].1;
    let weights = candidates
        .iter()
        .map(|(_, v)| f64::from((*v - maximum) / 0.9).exp())
        .collect::<Vec<_>>();
    let mut target = rng.random::<f64>() * weights.iter().sum::<f64>();
    for ((id, _), weight) in candidates.iter().zip(weights) {
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

fn resample(input: &[f32], source: usize, target: usize) -> Vec<f32> {
    if source == target {
        return input.to_vec();
    }
    let count = input.len() * target / source;
    let cutoff = (target as f64 / source as f64).min(1.0) * 0.95;
    (0..count)
        .map(|i| {
            let position = i as f64 * source as f64 / target as f64;
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
        let source = read_reference(&request.reference_audio).unwrap();
        let encoded = run(
            &mut pipeline.encoder,
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
            request.reference_audio = path;
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
        let output = resample(&vec![0.25; 16_000], 16_000, 24_000);
        assert_eq!(output.len(), 24_000);
        assert!(output.iter().all(|v| (v - 0.25).abs() < 1e-5));
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
}
