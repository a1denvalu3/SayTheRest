# ADR 0001: Benchmarkable inference boundary

Status: accepted for the first implementation slice

## Context

Say It uses MLX Audio, which targets Apple silicon. sayIt targets Windows and Linux across CPUs and GPUs. Choosing a UI framework does not answer the harder deployment question: which model/runtime combination delivers acceptable first-audio latency, real-time factor, memory use, quality, and install size on representative hardware.

## Decision

- Start with `sherpa-onnx-offline-tts` as a replaceable process adapter.
- Start with its CPU provider because it has the broadest deployment baseline.
- Record cold/process-inclusive p50, p95, p99 and real-time factor.
- Keep model files outside application packages and retain their license metadata.
- Replace the process adapter with the native C API only after the baseline proves model loading dominates interactive latency.
- Evaluate Windows WinML and Linux CUDA/OpenVINO builds separately; no accelerator becomes a default without measurements and output-quality comparison.

This slice deliberately does not select the desktop UI toolkit. The core, CLI, engine configuration, and benchmark remain usable from a native or webview shell.

## Acceptance targets

- RTF below 1.0 for default speech on supported baseline hardware.
- Report cold start separately from repeated synthesis.
- No network access during synthesis after explicit model download.
- Generated output must remain intelligible and stable across runtime/provider changes.
