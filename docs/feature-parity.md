# SayIt feature-parity contract

This document defines parity by user-visible behavior. A checkbox is complete only when it is covered by an automated test or a documented manual platform test.

## Always-available speech

- [x] Start a per-user background service at login.
- [ ] Select text in any accessible application and invoke a configurable global shortcut.
- [x] Read the clipboard only when its separate configurable shortcut is invoked.
- [x] Preserve clipboard contents when copy-based selection fallback is required.
- [x] Queue, replace, or interrupt speech according to the chosen queue policy.
- [x] Confirm before processing unusually long text.
- [x] Work offline after the chosen model is installed.

### Selection adapters

| Platform | Primary path | Explicit fallback |
|---|---|---|
| Windows | UI Automation TextPattern/TextPattern2 | Preserve clipboard, synthesize Ctrl+C, restore clipboard |
| Linux X11 | AT-SPI selected-text interfaces | Preserve clipboard, synthesize Ctrl+C, restore clipboard |
| Linux Wayland | AT-SPI when exposed by the application/compositor | Portal/compositor-compatible copy shortcut; show unsupported status when neither is permitted |

Wayland intentionally prevents arbitrary global input and selection access in some compositor configurations. Parity here means the app provides the best permitted path, detects failure, explains it, and never passively monitors the clipboard.

For the explicit clipboard shortcut, compositors without a data-control protocol use the XWayland bridge when available, then a user-approved XDG Clipboard portal attached to a Remote Desktop v2 session. The portal is created lazily and retained for the desktop process lifetime, so an approved session is reused rather than prompting on every shortcut press.

The copy fallback snapshots advertised raw formats (including text, HTML/RTF, images, file lists, and application-specific formats) before issuing Copy and restores them together afterward. It refuses the fallback before modifying the clipboard if a format cannot be read or the bounded snapshot cannot be held safely.

## Player and history

- [ ] Tray/menu-bar status and compact player (implemented; Linux native UI acceptance passes, Windows GUI acceptance remains).
- [x] Play, pause, resume, stop, seek, skip forward/back, rate, and volume.
- [x] Progressive PCM playback with buffering and pitch-preserving listening speed.
- [x] Current title, progress, duration, current spoken text, and chunk separators.
- [x] Amplitude waveform, word seeking, and reader follow controls (measured block boundaries with estimated word timing within each block).
- [x] Persistent history with search, pinning, replay, regeneration, deletion, and audio export.
- [x] Configurable history retention and disk quota.

## Models

- [x] Curated catalog with size, speed, quality note, languages, capabilities, and license.
- [x] Hardware-aware recommendations based on measured local performance.
- [x] Download progress, cancellation, integrity validation, install, update, selection, unload, and removal.
- [x] Import compatible community models by repository ID or local archive.
- [x] Keep at most one large model resident and unload it after configurable inactivity.
- [ ] CPU baseline plus independently packaged/tested acceleration providers.

### SayIt model-family coverage

Exact parity tracks SayIt's v0.2.0 bundled catalog at upstream commit
`37d49718bb58853945577a6845dce0cfde84c90d`; a generic model importer is not evidence that a
family works. Each entry requires an installed-model synthesis smoke test on Linux and Windows,
and capability tests for every feature advertised by that model.

- [ ] Phonon 31M English, custom voice files, and voice-pack updates (native Q8 worker implemented and tested on Linux; catalog, voice workflow, and Windows validation remain)
- [ ] Kokoro (synthesis exists; multilingual/preset coverage remains)
- [x] Kitten Mini and Kitten Nano (real synthesis, progressive callbacks, worker transport, and cancellation pass on Linux and Windows)
- [ ] Pocket TTS (cloning exists; preset coverage remains)
- [ ] Soprano
- [ ] Qwen3 TTS Base, VoiceDesign, and CustomVoice
- [ ] Chatterbox, Chatterbox Turbo, and Chatterbox Multilingual
- [ ] OmniVoice
- [ ] Irodori
- [ ] Vyvo
- [ ] Orpheus BF16 and 4-bit
- [ ] Breeze TTS 2 (experimental upstream)

Fish Audio S2 Pro, MOSS TTS, MOSS Nano, IndexTTS, Echo, and Marvis are marked
unavailable upstream. They are not requirements for parity with working upstream models.

## Voices

- [ ] Built-in voice and language selection where supported (presets exist; general language controls remain).
- [x] Per-model voice selection and speaking pace.
- [ ] Voice descriptions where supported.
- [x] Local recording/import, quality analysis, trimming, and voice cloning where supported.
- [ ] Voice discovery/random generation where supported.
- [x] Rename, preview, tune, select, and delete local voice profiles.
- [x] Require an affirmative speaker-permission acknowledgement for cloning.

## Interfaces and operations

- [x] Desktop onboarding, settings, diagnostics, updates, and launch-at-login.
- [ ] CLI parity (core commands exist; per-request model/language/voice overrides, waiting, and skill discovery remain).
- [x] Versioned localhost REST API with scoped bearer tokens, rate limiting, event stream, and OpenAPI document.
- [x] Crash-safe job journal and service recovery.
- [x] Linux packages and Windows installer that need neither Cargo nor Python.
- [x] No analytics, cloud inference, passive clipboard monitoring, or unexpected network access.

## Definition of done

Implementation order, preserving the complete feature and UI scope:

1. Progressive playback and pitch-preserving speed, verified with synthetic DSP signals and real local models.
2. Compact tray player, amplitude waveform, and word-level reader with seeking and follow controls.
3. Phonon and complete voice/language workflows, with real synthesis before advertising support.
4. Guided onboarding, service recovery, diagnostics, API controls, and management settings.
5. Remaining selectable model families, CLI compatibility, and history/export workflows.

Correctness takes priority over checking off an item. A model catalog entry or
passing mock does not establish real-model support; platform and hardware
limitations must stay visible until the corresponding acceptance run succeeds.

The playback implementation uses vendored Sonic for continuous tempo processing.
Tests verify pitch, duration, progressive output, tail preservation, buffering,
pause during generation, seeking, rate changes, and cancellation. Run
`python3 scripts/test-streaming.py --config /path/to/installed/config.json`
after building the service and worker for silent real-model/device acceptance in
an isolated service. This never changes the live app's settings or history.

On 2026-10-23, Linux acceptance with the installed Piper model observed playback
before generation finished (first audio poll about 1.5 s, complete generation
5.06 s, 68.58 s of archived source audio). Pause, seek, 1.5x listening speed, and
stop during the next synthesis passed against an actual audio device at zero
volume, including replay replacing an active synthesis. Real Kitten Mini and
Nano tests covered direct callbacks and the isolated worker, exact archived
sample counts, and cancellation without publishing incomplete audio. A native
INT4 Qwen run with a synthetic Piper reference delivered, after model loading,
first PCM at 52.13 s and completed at 65.36 s; its 245,760 streamed samples matched
the final WAV. This establishes progressive delivery, while making clear that
Qwen CPU first-audio latency still needs improvement. CUDA and Windows hardware
acceptance are not established by these Linux CPU results.

The compact player is a separate 360-pixel native window opened from the tray.
It retains completed recordings for replay and word seeking, offers an actual
amplitude waveform, keyboard navigation, and a reader that follows speech until
the user scrolls away. Text offsets use Unicode scalar positions across service
and desktop. Synthesis records the audio start/end of each short text block;
word timing within a block is approximate, matching the upstream approach rather
than claiming forced alignment. These boundaries persist in history and survive
replay. Older recordings use a fixed-duration estimate.

On 2026-10-23, the native Linux Tauri/WebKit acceptance run passed 20 checks with
an isolated Piper service and a real audio device at zero volume. It checked the
360-pixel layout, idle controls, pause during generation, real amplitudes, measured
blocks, word seeking while paused, manual/follow scrolling, keyboard navigation,
speed changes, volume preview, retained completion, replay without duplicate
history, clear/recent history, and opening management settings. Native screenshots
were inspected in light, dark, and idle states. The suite is reproducible with:

```sh
cargo build -p sayit-service-app -p sayit-desktop --locked
python3 scripts/test-desktop-ui.py --config /path/to/installed/preset/config.json --snapshots /tmp/sayit-ui
node --test apps/desktop/tests/reader-timing.test.cjs
```

The native UI runner currently targets Linux and requires an active display;
it does not establish Windows GUI or compositor-specific tray placement. Linux
AppIndicator exposes the player through its menu because Tauri does not provide
tray click coordinates/events there. Windows uses direct tray clicks. On Wayland,
window placement follows the compositor when explicit positioning is unavailable.

The native Phonon worker uses the pinned `ptts` 0.4.0 CPU/Q8 runtime. A real
Linux run with the immutable 31M assets checked progressive PCM, exact archive
sample values, cancellation without publishing/leaking partial audio, and a
successful request after cancellation. Cold first PCM was 2.919 s and completion
5.693 s; warm first PCM was 0.467 s and completion 3.266 s for 5.84 s of speech.
These are local measurements, not upstream benchmark claims. Run
`scripts/test-phonon.sh <rust-target>` with `SAYIT_TEST_WORKER` set to the built
release worker; the release matrix runs it against the packaged binary. The
runtime requires no Python and makes no model/voice network requests during
synthesis. Native speaking pace is rejected explicitly; listening speed remains
the separate pitch-preserving player control.

Linux and Windows release candidates must pass the same behavior suite. Platform-specific selection tests run against representative native, Chromium, Electron, terminal, and office applications. Model/provider changes must compare audio quality and report cold start, warm p50/p95/p99, peak memory, time to first audio, and real-time factor on the supported hardware matrix.

The job journal is written through a synced temporary file with a last-known-good backup. On
startup, queued work remains queued; interrupted synthesis or playback is requeued once after
partial audio is removed. A second interruption becomes a visible failed history item instead of
causing an endless service crash loop. Automated tests cover malformed and missing primary state,
backup restoration, retry, cleanup, and the repeated-interruption guard.

## Release evidence

Commit `0f03465` passed the complete Linux/Windows release matrix on 2026-10-23,
including workspace tests, real Mini/Nano progressive synthesis through the
packaged resident runtime and isolated worker, cancellation, and package audits.
The Windows run exposed and verified a fix for 32-bit overflow in Sonic's
full-rate pitch comparison; the signal tests retain their original pitch bounds.
GitHub Actions run:
[39947527253](https://github.com/a1denvalu3/SayTheRest/actions/runs/39947527253).

The 2026-08-10 release-candidate matrix for commit `3061cb9` ran the same workspace test suite on
Ubuntu 24.04 and Windows Server 2025, then built and inspected every native package. Before
packaging, both platforms executed a real desktop-clipboard transaction that saved plain text,
canonical HTML, and a custom binary format, exposed temporary selected text to the app, and
verified that all original payloads were restored. This specifically covers Windows standard
numeric clipboard formats rather than re-registering their names as unrelated custom formats.
The Linux audit verified the AppImage, Debian package, portable archive, supervised desktop
service, and bundled inference executable. The Windows audit expanded the portable ZIP and
verified the CLI, service, desktop shortcut host, PowerShell installer, bundled inference
executable, and per-user installer. GitHub Actions run:
[31405332625](https://github.com/a1denvalu3/SayTheRest/actions/runs/31405332625).

Commit `2a36d0b` added pinned Kitten Mini 0.8 and Kitten Nano 0.8 INT8 downloads, all eight
preset voices, process-cold inference, and resident native inference. Release run
[31407920797](https://github.com/a1denvalu3/SayTheRest/actions/runs/31407920797) downloaded both
real model archives independently on Ubuntu and Windows, verified their SHA-256 digests, loaded
them through each packaged sherpa runtime, and produced valid audio before auditing all release
packages. This proves cross-platform synthesis, not SayIt's progressive first-audio behavior;
that remaining capability keeps the model-family parity item open.
