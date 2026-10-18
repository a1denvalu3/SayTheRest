# Automatic inference devices

Settings → Inference device defaults to **Automatic**, which prefers a compatible GPU
for Piper, Kokoro, Kitten, PocketTTS, and Qwen. **CPU** forces CPU execution for service
jobs, built-in voice previews, cloned voice previews, and model benchmarks. Diagnostic
CLI `synth` and `benchmark` use the same device policy; set `SAYIT_COMPUTE_DEVICE=cpu`
to force their CPU path.

The managed Windows/Linux x86-64 builds support NVIDIA CUDA. Detection queries the CUDA
driver and respects an empty or `-1` `CUDA_VISIBLE_DEVICES`. Drivers supporting CUDA
12.6 use the pinned cu126 runtime; drivers supporting 12.8 or newer use cu128 for Qwen.
Older drivers and devices without a supported backend use CPU.

On first GPU use, the app prepares a separate cached runtime. Qwen receives CUDA-enabled
PyTorch 2.9.1 and selects bfloat16 when supported, otherwise float16. CPU uses float32.
ONNX models receive sherpa-onnx 1.13.4's CUDA 12/cuDNN 9 runtime, verified against a pinned
SHA-256 digest. Linux ONNX inference downloads NVIDIA libraries without installing Qwen
or PyTorch. Windows obtains the matching CUDA DLLs from the PyTorch runtime. CPU-only
systems keep their small CPU runtime. GPU runtime preparation can require substantial
downloads; model weights and recorded voices remain local.

Qwen uses PyTorch SDPA; the optional external `flash-attn` package is unnecessary.
User-supplied Qwen runtimes can also expose ROCm, Intel XPU, or Apple MPS through PyTorch.
The managed ONNX releases do not supply compatible Intel/AMD GPU providers, so those
devices use CPU. NPU inference is not enabled.

The Models view shows the provider of the loaded model. Settings diagnostics and
`GET /v1/state` expose the last inference device, precision, and any fallback reason.
CUDA ONNX sessions can run unsupported operations on CPU; a CUDA provider label does
not assert that every graph operation ran on the GPU. Provider initialization and
generation failures retry on CPU. CUDA ONNX inference runs in the supplied executable
so a native provider abort cannot take down the service; CPU and Qwen retain resident
models. An ONNX CUDA model therefore reloads for each synthesis request.
Cold-start benchmark timing includes model initialization and first generation, while
excluding one-time runtime downloads. Repeated timings measure subsequent synthesis.

Existing reference profiles remain usable. Qwen now accepts 3–45 seconds of reference
speech and a matching transcript; PocketTTS keeps its 3–15 second limit. The longer
studio passage provides varied sentences and a question. A reference conditions each
generation; creating a profile does not train or fine-tune the model.

Primary references: [sherpa CUDA builds](https://k2-fsa.github.io/sherpa/onnx/install/linux.html),
[PyTorch 2.9.1 runtimes](https://pytorch.org/get-started/previous-versions/#v291),
[PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html).
