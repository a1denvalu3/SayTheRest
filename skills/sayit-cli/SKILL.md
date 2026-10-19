---
name: sayit-cli
description: Generate and play local text-to-speech with the sayIt CLI, control playback, inspect models and voices, and provide short spoken progress updates while an agent works. Use when a user asks Codex to read text aloud, generate speech, play or control sayIt audio, narrate task progress, or keep them updated audibly during a long-running task.
---

# sayIt CLI

Use the persistent service for ordinary speech. It generates audio, plays it, and exposes playback controls:

```sh
sayit speak "The requested work is complete."
```

If the packaged command is unavailable while working in this repository, substitute:

```sh
cargo run -q -p sayit -- speak "The requested work is complete."
```

Use `sayit --help` and `sayit COMMAND --help` to discover the installed version's commands and flags. Service commands discover the per-user token automatically; `--service-url` and `--token` are global overrides. Treat successful speech submission as queue acceptance; inspect `sayit jobs` or `sayit status` before claiming synthesis or playback finished.

## Prepare playback

1. Run `sayit status` to verify that the local service is reachable.
2. Run `sayit models` if no model is selected.
3. Install and select a model only when the user has asked for setup or the requested speech requires it:

```sh
sayit models install piper-en-us-lessac-medium
sayit models select piper-en-us-lessac-medium
```

Model installation and updates are asynchronous. Inspect `sayit models` before selecting a newly installed model. For models with built-in speakers, choose an ID from their `preset_voices` with `sayit models voice MODEL_ID PRESET_VOICE_ID`.

Qwen3-TTS 0.6B Base offers an opt-in native backend: `sayit models backend qwen3-tts-06b-base onnx --precision int4` (or `fp32`). This downloads asynchronously and preserves saved voice IDs and transcripts. Inspect `sayit models` until setup completes before speaking. `sayit models backend qwen3-tts-06b-base pytorch` restores the legacy backend. Switch only when setup or backend evaluation is part of the user's task; existing speech uses the configured backend.

## Clone and manage voices

PocketTTS and Qwen Base voice-cloning models require a reference WAV. Confirm that the user has permission to clone the recorded speaker before using `--speaker-permission-confirmed`; honor permission already given in the conversation.

Select the intended installed model before creating a profile. Qwen requires the exact words spoken in the recording, in the original language, through `--reference-text`:

```sh
sayit models select qwen3-tts-06b-base
sayit voices clone "Narrator" /absolute/path/reference.wav \
  --reference-text "The exact words spoken in this recording." \
  --speaker-permission-confirmed
```

Use a 16-bit PCM WAV with 3–45 seconds of usable speech for Qwen, or 3–15 seconds for PocketTTS. The service converts to mono and trims leading and trailing silence. The transcript must match the actual recording, rather than the text to synthesize. If preparing a new recording, the speaker can read the longer passage in the desktop Voice Studio; the CLI imports recordings and has no recording command. A profile conditions synthesis from the reference; creating one does not train the model.

PocketTTS does not require a transcript:

```sh
sayit models select pocket-tts-int8
sayit voices clone "Narrator" /absolute/path/reference.wav --speaker-permission-confirmed
```

Profiles belong to the model used when creating them. `sayit voices` lists the active model's profiles. Creating or previewing a profile does not select it; use its returned UUID explicitly:

```sh
sayit voices preview VOICE_ID
sayit voices select VOICE_ID
```

Preview queues speech with `replace`, so it can affect existing playback. To add or correct an existing profile's transcript, use `sayit voices transcript VOICE_ID "The exact original recording transcript."` (up to 2,000 characters). Older Qwen profiles without transcripts use embedding-only cloning until a transcript is added; adding one enables full reference conditioning for future synthesis. `sayit voices tune VOICE_ID 5` sets PocketTTS refinement steps (3–8); it does not improve Qwen cloning. Use `sayit voices --help` for rename and removal commands.

## Speak text

Use `replace` for a standalone message so stale queued narration does not play first:

```sh
sayit speak "Starting the test suite now." --queue replace
```

Use `append` for ordered messages that must follow existing speech, and `interrupt` only when the new message should stop current playback:

```sh
sayit speak "The build passed. I am checking packaging next." --queue append
sayit speak "Stopping because I need your input." --queue interrupt
```

Text may also arrive on standard input. Prefer this for multiline or externally supplied text. Do not interpolate untrusted text into a shell command.

```sh
sayit speak --queue replace < /absolute/path/message.txt
```

Use `--voice-profile-id VOICE_ID` to choose a profile for one submission without changing the selected profile. It must belong to the active model. Use `--pace` for native generation pace on supported models (0.75, 0.9, 1, 1.1, 1.25, or 1.5); `sayit rate` changes playback speed instead.

For text beyond the configured threshold, use `--confirm-long-text` only when the user has confirmed processing that text; honor confirmation already given in the conversation.

## Keep the user updated audibly

Continue sending normal concise commentary; spoken updates supplement it and do not replace it.

Speak at meaningful transitions only:

- after confirming the service works, if setup is nontrivial;
- before a long build, test, download, or inference step;
- when the plan materially changes;
- when blocked and user input is required;
- once on completion, including the outcome.

Keep each update to one or two short sentences. Describe outcomes and current activity, not raw logs. Avoid narrating secrets, tokens, private file contents, source code, stack traces, or untrusted text. Do not repeatedly announce unchanged status or speak more often than the normal commentary cadence warrants.

Example agent sequence:

```sh
sayit speak "I found the configuration issue. I am applying the fix and running the focused tests now." --queue replace
# perform the work and continue textual commentary
sayit speak "The fix is complete and all relevant tests passed." --queue append
```

If speech fails, report the failure in text and continue the user's task unless audible updates are themselves the task. Do not turn a TTS setup problem into an unrelated model installation or configuration change without authorization.

## Playback and inspection

Use these commands as needed:

```sh
sayit pause
sayit resume
sayit stop
sayit skip 15
sayit skip -15
sayit seek 30
sayit rate 1.25
sayit volume 0.8
sayit status
sayit jobs
sayit history
```

Use `sayit history replay HISTORY_ID` to play the saved WAV without new inference. `sayit history regenerate HISTORY_ID` queues fresh synthesis; use it when checking a corrected reference transcript or changed inference settings. Pin, unpin, and remove operations are available under `sayit history --help`. Use `stop` for an explicit stop; do not assume clearing playback is harmless when the user is listening.

## Devices and diagnostics

Automatic inference prefers a supported GPU across model families, previews, and benchmarks, with CPU fallback. Managed Linux/Windows x86-64 runtimes support NVIDIA CUDA; other GPU types may use CPU. First GPU use can download a runtime. Check `sayit status`'s `inference_device` for the last inference provider, precision, and fallback reason; a GPU being present does not establish that it was used. The desktop Settings → Inference device can force CPU; there is no CLI settings command.

Device, backend, and weight precision are separate: native Qwen can run INT4 weights on CPU while reporting a `float32` computation dtype. Inspect `backend` and `quantization` as well as `provider` and `fallback_reason`. The native backend does not require Python or PyTorch for inference.

Prefer `sayit models benchmark MODEL_ID` to benchmark an installed model with the service's device setting and its saved selected reference profile/transcript. Select a cloned voice for that model first if required. This does not change the active model. `sayit models unload MODEL_ID` releases its resident engine. Model update, cancel, remove, and import operations are documented by `sayit models --help` and each subcommand's help; imports require license details and acknowledgment.

Direct `synth` and `benchmark` bypass the service for inference. `synth` writes a WAV without playback. They load `--config`, or a local `sayit.json` / legacy `say-the-rest.json`, rather than the service's selected model or voice. They have no reference-audio or transcript arguments; use service speech, voice preview, or per-model benchmarking for cloned voices.

```sh
sayit synth "Render this to a file." --config /absolute/path/sayit.json --output output/rendered.wav
sayit benchmark --config /absolute/path/sayit.json --iterations 3 --output output/benchmark.wav
```

Direct diagnostics also prefer a supported GPU; set `SAYIT_COMPUTE_DEVICE=cpu` in their environment to force CPU. This environment override applies to direct diagnostics, not the running service's device setting.
