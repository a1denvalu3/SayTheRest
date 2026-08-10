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

- [x] Tray/menu-bar status and compact player.
- [x] Play, pause, resume, stop, seek, skip forward/back, rate, and volume.
- [x] Current title, progress, duration, current spoken text, and chunk separators.
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

Exact parity tracks SayIt's bundled catalog at upstream commit
`e428efeeb2a98425a38c0282f5ea95d9090a250d`; a generic model importer is not evidence that a
family works. Each entry requires an installed-model synthesis smoke test on Linux and Windows,
and capability tests for every feature advertised by that model.

- [x] Kokoro
- [ ] Kitten Mini and Kitten Nano
- [x] Pocket TTS
- [ ] Soprano
- [ ] Qwen3 TTS Base, VoiceDesign, and CustomVoice
- [ ] Chatterbox, Chatterbox Turbo, and Chatterbox Multilingual
- [ ] OmniVoice
- [ ] Fish Audio S2 Pro
- [ ] Irodori
- [ ] Vyvo
- [ ] Orpheus BF16 and 4-bit
- [ ] MOSS TTS and MOSS Nano
- [ ] IndexTTS
- [ ] Echo
- [ ] Marvis

## Voices

- [x] Built-in voice and language selection where supported.
- [x] Per-model voice selection and speaking pace.
- [ ] Voice descriptions where supported.
- [x] Local recording/import, quality analysis, trimming, and voice cloning where supported.
- [ ] Voice discovery/random generation where supported.
- [x] Rename, preview, tune, select, and delete local voice profiles.
- [x] Require an affirmative speaker-permission acknowledgement for cloning.

## Interfaces and operations

- [x] Desktop onboarding, settings, diagnostics, updates, and launch-at-login.
- [x] CLI parity for speech, status, jobs, models, voices, history, and playback control.
- [x] Versioned localhost REST API with scoped bearer tokens, rate limiting, event stream, and OpenAPI document.
- [x] Crash-safe job journal and service recovery.
- [x] Linux packages and Windows installer that need neither Cargo nor Python.
- [x] No analytics, cloud inference, passive clipboard monitoring, or unexpected network access.

## Definition of done

Linux and Windows release candidates must pass the same behavior suite. Platform-specific selection tests run against representative native, Chromium, Electron, terminal, and office applications. Model/provider changes must compare audio quality and report cold start, warm p50/p95/p99, peak memory, time to first audio, and real-time factor on the supported hardware matrix.

The job journal is written through a synced temporary file with a last-known-good backup. On
startup, queued work remains queued; interrupted synthesis or playback is requeued once after
partial audio is removed. A second interruption becomes a visible failed history item instead of
causing an endless service crash loop. Automated tests cover malformed and missing primary state,
backup restoration, retry, cleanup, and the repeated-interruption guard.

## Release evidence

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
