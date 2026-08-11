---
name: say-the-rest-cli
description: Generate and play local text-to-speech with the Say the Rest CLI, control playback, inspect models and voices, and provide short spoken progress updates while an agent works. Use when a user asks Codex to read text aloud, generate speech, play or control Say the Rest audio, narrate task progress, or keep them updated audibly during a long-running task.
---

# Say the Rest CLI

Use the persistent service for ordinary speech. It generates audio, plays it, and exposes playback controls:

```sh
say-the-rest speak "The requested work is complete."
```

If the packaged command is unavailable while working in this repository, substitute:

```sh
cargo run -q -p say-the-rest -- speak "The requested work is complete."
```

Do not use `synth` when the user expects playback. `synth` bypasses the service and only writes a diagnostic WAV:

```sh
say-the-rest synth "Render this to a file." --output output/rendered.wav
```

## Prepare playback

1. Run `say-the-rest status` to verify that the local service is reachable.
2. Run `say-the-rest models` if no model is selected.
3. Install and select a model only when the user has asked for setup or the requested speech requires it:

```sh
say-the-rest models install piper-en-us-lessac-medium
say-the-rest models select piper-en-us-lessac-medium
```

Model installation is asynchronous. Inspect `say-the-rest models` before selecting it. PocketTTS requires a permitted reference WAV and a selected cloned voice:

```sh
say-the-rest models select pocket-tts-int8
say-the-rest voices clone "Narrator" /absolute/path/reference.wav --speaker-permission-confirmed
say-the-rest voices
say-the-rest voices select VOICE_ID
```

Never assert speaker permission on the user's behalf. Use PocketTTS cloning only after the user explicitly confirms permission.

## Speak text

Use `replace` for a standalone message so stale queued narration does not play first:

```sh
say-the-rest speak "Starting the test suite now." --queue replace
```

Use `append` for ordered messages that must follow existing speech, and `interrupt` only when the new message should stop current playback:

```sh
say-the-rest speak "The build passed. I am checking packaging next." --queue append
say-the-rest speak "Stopping because I need your input." --queue interrupt
```

Text may also arrive on standard input. Prefer this for multiline or externally supplied text. Do not interpolate untrusted text into a shell command.

```sh
say-the-rest speak --queue replace < /absolute/path/message.txt
```

For text beyond the configured threshold, ask before using `--confirm-long-text`. Never use that option merely to bypass a user-facing safeguard.

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
say-the-rest speak "I found the configuration issue. I am applying the fix and running the focused tests now." --queue replace
# perform the work and continue textual commentary
say-the-rest speak "The fix is complete and all relevant tests passed." --queue append
```

If speech fails, report the failure in text and continue the user's task unless audible updates are themselves the task. Do not turn a TTS setup problem into an unrelated model installation or configuration change without authorization.

## Playback and inspection

Use these commands as needed:

```sh
say-the-rest pause
say-the-rest resume
say-the-rest stop
say-the-rest skip 15
say-the-rest skip -15
say-the-rest seek 30
say-the-rest rate 1.25
say-the-rest volume 0.8
say-the-rest status
say-the-rest jobs
say-the-rest history
```

Use `say-the-rest history replay HISTORY_ID` to play previously generated audio. Use `stop` for an explicit stop; do not assume clearing playback is harmless when the user is listening.
