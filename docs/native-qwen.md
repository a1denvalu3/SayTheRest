# Native Qwen3-TTS

Qwen3-TTS 0.6B Base has an opt-in native ONNX backend. A persistent Rust
`sayit-inference-worker` process uses ONNX Runtime 1.24.2. Inference, text
tokenization, reference encoding, and audio decoding require no Python or
PyTorch. The existing service owns the queue, saved voices, history, device
preference, and playback.

## Enable or switch back

Use **Try native INT4** on Qwen's model card, or:

```sh
sayit models backend qwen3-tts-06b-base onnx --precision int4
sayit models
```

Setup is asynchronous. Wait until the download reports `installed`, then select
the model and one of its saved voices. Switching preserves the model ID, voice
UUIDs, recordings, and transcripts. Files are verified before replacing the
active configuration.

For an FP32 comparison or to restore the original implementation:

```sh
sayit models backend qwen3-tts-06b-base onnx --precision fp32
sayit models backend qwen3-tts-06b-base pytorch
```

Returning to PyTorch reuses its original weights/settings when available;
otherwise it downloads and prepares that backend. ONNX remains opt-in while
quality and performance are evaluated across supported hardware.

## Generate a voice locally

With the native backend selected, open **Voice studio → Explore generated
voices**. Choose the sample text, language, sample count, and style, then select
**Generate samples**. Preview a candidate, give it a name, select **Save voice**,
and then **Use this voice**. Saved voices can read new text and survive restarts.
No recording or transcript from a human speaker is needed for this workflow.

Move **Voice character** between Faithful and Expressive, or enable **Surprise
me** to explore different generation settings for each candidate. The optional
seed also makes these recipes repeatable with the same runtime. Each saved voice
keeps the settings used for its sample.

New batches append to the current collection, up to 32 candidates. Cancellation
keeps completed samples. Unsaved candidates are temporary; save favorites before
restarting. A sample that fails reference-quality checks can still be previewed,
but cannot be saved. **Clear candidates** preserves saved voices.

**Generation settings** on a saved Qwen profile controls temperature, top P,
top K, and repetition penalty for future readings. Legacy profiles retain their
previous defaults until edited. Explicitly selecting a profile also restores its
preferred language. Listening speed remains a separate playback control.

## References and precision

New profiles require the exact recording transcript. References retain the
3–45 second limit. The native encoder processes the whole recording in the
export's one-second windows. Legacy profiles without transcripts can use the
speaker-embedding-only path. A bounded cache retains the last reference prompt,
invalidated when the audio or transcript changes.

The INT4 variant quantizes weights; activations and speech codec components
use FP32. CPU execution does not imply FP32 weights. `sayit status` reports
`backend`, `provider`, `dtype`, `quantization`, and `fallback_reason` separately.
CUDA sessions retain the talker's key/value cache on the GPU. Automatic device
selection tries supported NVIDIA CUDA and retries on CPU if preparation,
initialization, or generation fails. CPU-only setups do not download CUDA.

## Assets and distribution

`crates/core/data/qwen-onnx-assets.json` pins the ONNX Community export and
upstream tokenizer/architecture revisions, sizes, and SHA-256 digests. Only
the requested precision is downloaded, including its FP32 audio components.

Release packages include the native worker and CPU runtime. Development builds
require Rust 1.92 or newer and can prepare the pinned CPU runtime during setup.
CUDA preparation downloads a
separate pinned runtime and NVIDIA library wheels, extracting native libraries
without installing Python. Cached inference runs offline. Runtime libraries
load in the worker, separate from sherpa-onnx's library version.

See `QWEN-ONNX-NOTICE.txt`, `Apache-2.0.txt`, and the runtime's notices for
attribution. Sources:

- https://huggingface.co/onnx-community/Qwen3-TTS-12Hz-0.6B-Base
- https://github.com/QwenLM/Qwen3-TTS

## Validation

Unit tests cover runtime fallback, INT4 preservation on CPU, cancellation,
worker cleanup, cached-download integrity, and existing model behavior. Full
model checks require downloaded assets and a local reference, which can be
generated synthetically with the discovery workflow.
Compare tokenization, the complete prompt, speaker embedding, and early greedy
audio tokens against the pinned ONNX reference, then check sampled speech and
long references. Greedy decoding can repeat instead of ending; normal speech
uses sampling with a bounded token budget.

Benchmark cold loading separately from repeated synthesis, including reference
preparation on first use. Check intelligibility, voice similarity, memory use,
and real-time factor before changing the default. NVIDIA CUDA and Windows
execution need validation on their respective hardware; CPU fallback tests
do not establish GPU performance.

On the development Intel Core Ultra 9 185H laptop, the native INT4 prompt,
speaker embedding, and first three greedy codec frames matched the ONNX
reference exactly. Local speech recognition recovered the complete test
sentence from sampled output. Actual model tests verified that 3-, 15-, 30-,
and 45-second references all reach the audio encoder. An isolated service
with Python absent from PATH completed native setup, preserved a saved voice,
cancelled generation, and subsequently benchmarked successfully.

The three-iteration CPU service benchmark measured mean RTF 3.89 (above real
time), with cold start 14.67 s. This establishes a working native implementation,
not a CPU speed guarantee. The INT4 worker used approximately 2 GB RSS in the
repeated-request check. CUDA acceleration and Windows inference have not been
hardware-tested on this development machine.
