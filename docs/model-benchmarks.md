# Local model benchmark record

Catalog recommendations are derived from measurements stored by the service after a user runs
**Benchmark**. This record captures the acceptance run used while adding Kokoro; it is not a
universal performance claim.

## 2026-08-10 · Linux x86-64 CPU

- Hardware: Intel Core i7-1365U, 10 cores / 12 logical CPUs
- Runtime: sherpa-onnx 1.13.4, CPU provider, four inference threads
- Workload: “Say the Rest keeps your words private and reads them aloud locally.”
- Method: five process-cold iterations, including model initialization and output writing
- Audio: 24 kHz, mono, 16-bit PCM

| Model | Quantization | Cold start | p50 | p95 / p99 | Mean process-inclusive RTF | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Piper Lessac Medium | model release default | 1,770 ms | 1,386 ms | 1,770 / 1,770 ms | 0.415 | 175 MiB |
| Kokoro 82M | INT8 | 8,132 ms | 7,775 ms | 8,353 / 8,353 ms | 2.189 | 352 MiB |

Both models produced valid PCM WAV output. Kokoro speaker IDs 3 (`af_heart`) and 11
(`am_adam`) produced different audio hashes and durations from the same input. On this machine,
Piper is the interactive-speed recommendation; Kokoro remains an opt-in quality/voice-variety
choice. Audio quality still requires a listening evaluation on representative text and languages.

## Resident-engine acceptance

On the same machine, an isolated service loaded Piper through the bundled sherpa C API. Two jobs
completed through the same native engine; the model endpoint reported `resident: true` between
jobs and `resident: false` immediately after the explicit unload request. A separate isolated
service loaded Kokoro through the same resident path and completed a valid synthesis job. The
service stores one optional resident slot, replaces it during model changes, reloads it when a
voice configuration changes, and checks the configured inactivity timeout every 15 seconds.

The rebuilt AppImage was then exercised as the active per-user systemd service. Two queued Kokoro
jobs completed without API errors, both outputs were valid 24 kHz mono 16-bit PCM WAV files, the
catalog reported the selected model as resident between jobs, and an explicit unload changed that
state to false. The Windows resident-loader path is compile-checked for x86-64 and uses
`LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` so sherpa's adjacent ONNX Runtime and provider DLLs resolve from
the packaged runtime directory; an actual Windows-device acceptance run is still required.
