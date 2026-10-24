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

- [x] Phonon 31M English, custom voice files, and voice-pack updates (38 presets, three import formats, exact streamed archives, cancellation, recovery, and cached updates pass Linux and Windows)
- [ ] Kokoro (synthesis exists; multilingual/preset coverage remains)
- [x] Kitten Mini and Kitten Nano (real synthesis, progressive callbacks, worker transport, and cancellation pass on Linux and Windows)
- [x] Pocket TTS (eight presets, optional cloning, cancellation/recovery, and install/update repair pass Linux and Windows; native Linux UI acceptance also passes)
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

- [ ] Built-in voice and language selection where supported (Qwen language controls pass Linux native UI acceptance; Kokoro language coverage and Windows acceptance remain).
- [x] Per-model voice selection and speaking pace.
- [ ] Voice descriptions where supported.
- [x] Local recording/import, quality analysis, trimming, and voice cloning where supported.
- [ ] Voice discovery/random generation where supported (native Qwen candidate generation, preview, save, reuse, and tuning pass Linux native acceptance; automatic stable/per-paragraph reading modes remain).
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

Phonon now installs from pinned, SHA-256-checked assets with all 38 upstream
presets. Linux real-worker acceptance synthesized every preset and compared its
streamed PCM with every archived sample. Imported F32, F16, and BF16 voice files
also produced valid audio. Import validation rejects invalid shapes, unsafe
sizes, non-finite values, and silent latents. Profiles are copied into private
storage and survive restarts; importing a voice file does not require uploading
audio or using a Gradium account inside this application.

The native management UI passed 12 checks covering preset selection, invalid-file
feedback, import, selection, real preview, archive publication, and deletion.
The test exposed and verified fixes for lost service error details and a race
where playback could report completion before its history item was available.
Fresh catalog installation, selected-voice preservation across update, and silent
device playback passed in a separate isolated release service. The latest install
took 38.09 s and a cached update took 1.00 s. A simulated voice-pack revision at
unchanged paths verified that the next request reloads the resident voice.
Failed/cancelled updates preserve the active installation and custom profiles.

```sh
python3 scripts/test-preset-model-install.py --model phonon-31m-en
python3 scripts/test-phonon-voices.py --model /path/to/phonon --worker target/release/sayit-inference-worker
python3 scripts/test-desktop-ui.py --config /path/to/phonon-config.json --window main --script scripts/desktop-phonon-ui-smoke.js --voice-file /path/to/export.safetensors
```

The release matrix covers Phonon installation/update without opening an audio
device, plus all presets, imported formats, progressive delivery, exact archives,
cancellation, and recovery against its packaged worker. Linux native
UI/device evidence does not establish Windows GUI or audio-device behavior.

Pocket installs the eight upstream English preset recordings: Alba, Azelma,
Cosette, Eponine, Fantine, Javert, Jean, and Marius. Each recording is pinned to
an immutable source revision and SHA-256 digest; its recording license and
attribution accompany the installation. The voice list shows each recording's
license separately from the model license. A default preset enables immediate
speech while keeping local voice cloning optional. Existing installations keep
their saved profiles and can update to obtain the preset pack.

Real Linux resident-runtime acceptance synthesized all eight presets, including
24/32/48 kHz reference resampling, compared every streamed sample with the saved
WAV, and verified cancellation, recovery, explicit profile-reference overrides,
and isolated-worker transport. Sherpa's post-generation silence scaling is
disabled so an archived recording cannot differ from audio already streamed to
the player. Pocket uses its natural speaking pace; listening speed remains a
separate player control. The same change passed Piper's real-device progressive
playback regression.

Updates validate installed weight checksums before reuse. A real isolated-service
acceptance run restored a missing selected Eponine recording in 6.53 s without
redownloading valid weights or losing the selected voice. Unit coverage also
checks legacy configurations, optional cloning, damaged weight rejection, and
cancellation preserving the installed files.

The native Linux management UI passed 15 checks with both a fresh download and
verified cached assets, covering default playback, optional profile creation, preset/profile
switching, licensing, previews, reader text, history isolation, and profile
deletion. A separate native run verifies that the visible catalog reflects an
installation started through the API without reopening the model pane. Model
descriptors read installation progress before filesystem state to avoid a
terminal status paired with a stale uninstalled flag; the visible catalog also
refreshes periodically after external changes. The fresh-download repeat passed
after these changes. Both platform release jobs subsequently passed the real
model-family suite; Windows GUI and device acceptance remain separate.

```sh
scripts/test-pocket-model.sh <rust-target>
python3 scripts/test-preset-model-install.py --model pocket-tts-int8
python3 scripts/test-desktop-ui.py --config /path/to/preset/config.json --window main --script scripts/desktop-pocket-ui-smoke.js --voice-file /path/to/permitted-reference.wav --timeout 720
```

Linux and Windows release candidates must pass the same behavior suite. Platform-specific selection tests run against representative native, Chromium, Electron, terminal, and office applications. Model/provider changes must compare audio quality and report cold start, warm p50/p95/p99, peak memory, time to first audio, and real-time factor on the supported hardware matrix.

Qwen language selection is available in Speech & storage for installed models
that support it. It saves a separate default for each model, keeps the selected
voice profile, and accepts automatic detection or one of the ten upstream
language codes. Each queued job captures its effective language; history and
regeneration retain it after the default changes. Unsupported languages fail
before synthesis instead of silently using automatic detection. Native and
PyTorch backends receive the same canonical language names. Both use natural
speaking pace; unsupported per-request pace values are rejected explicitly.

Linux native-worker acceptance produced nonempty, nonsilent 24 kHz audio for
English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish,
Italian, and automatic detection. A rejected unknown language left no output and
the worker remained usable. The desktop passed 12 checks with a real Qwen
model and silent device playback, including Spanish/French selection, profile
preservation, queued-language isolation, and history metadata. A separate real
PyTorch service run produced Spanish audio while respecting natural pace after a
different global pace preference was saved. These checks establish valid audio
and working controls; pronunciation quality still needs listening evaluation.
Native visibility checks also verify that unsupported controls are hidden in
Pocket and Phonon. A shared hidden-state rule prevents grid layout styles from
leaving those controls visible. Windows Qwen synthesis and GUI acceptance remain
separate requirements.

```sh
python3 scripts/test-qwen-languages.py --config /path/to/native-config.json --reference /path/to/reference.wav
python3 scripts/test-qwen-service.py --config /path/to/qwen-config.json --reference /path/to/reference.wav --reference-text 'Exact words in the recording.'
python3 scripts/test-desktop-ui.py --config /path/to/native-config.json --window main --script scripts/desktop-language-ui-smoke.js --voice-file /path/to/reference.wav --reference-text 'Exact words in the recording.' --timeout 240
```

The native Qwen Base worker can also generate a sample without a reference
recording, using the prompt layout in the pinned Mac runtime. Linux real-model
acceptance verified repeatable seeded output, different audio for different
seeds, reuse of a generated sample as a cloning reference, and isolation between
cloned and unconditioned requests. Existing cloned output remained byte-for-byte
equal to its earlier baseline. This establishes the native Base sampling path;
automatic stable/per-paragraph random reading modes and the separate VoiceDesign
and CustomVoice model workflows remain incomplete.

Voice studio now exposes native Qwen candidate generation with editable sample
text, ten languages plus automatic detection, four samples by default, optional
seeds, and Faithful/Natural/Expressive generation presets. New batches append to
the current collection (up to 32 samples). Completed candidates remain available
after cancellation. Preview plays the generated recording and matching reader
text without adding history. Saving copies analyzed reference audio, transcript,
language, and tuning into a durable named profile; simultaneous saves return
the same profile. Clearing candidates leaves saved profiles intact. Unsaved
candidates are private temporary files and are not restored after restart.

Linux real-model service acceptance checked the default four-candidate batch, silent device preview,
two concurrent saves, reading new text with the saved voice, cancellation, and
collection cleanup. Native desktop acceptance passed 25 checks in Spanish,
including editing profile generation settings. The language labels include an
`Intl.DisplayNames` fallback for older webviews. Screenshot review exposed a
legacy grid rule compressing the new form; it was corrected and covered by native layout checks. Light/dark
screenshots were inspected. Rejection tests cover invalid controls/languages,
full collections preserving their prior samples, clearing during generation,
saved profile persistence, language restoration, and legacy profile defaults.
Failed and cancelled attempts release their temporary audio while completed
candidates remain available. Pocket passed 17 native UI checks and Phonon passed
13, including their existing workflows and hiding unsupported discovery.
The release matrix now includes real Qwen CPU tests on both platforms; a new
matrix run must pass before Windows synthesis coverage is claimed.

```sh
python3 scripts/test-voice-discovery.py --config /path/to/native-config.json
python3 scripts/test-desktop-ui.py --config /path/to/native-config.json --window main --script scripts/desktop-discovery-ui-smoke.js --snapshots /tmp/sayit-discovery-ui --timeout 300
scripts/test-qwen-native.sh <rust-target>
```

The last command uses the packaged service, worker, and native runtime, downloads
only pinned/checksummed INT4 model files, verifies seeded free/clone isolation,
managed streaming/archive counts, cancellation/recovery, headless discovery and
concurrent saving, then synthesizes all ten languages and automatic detection
from a locally generated synthetic reference. It requires no recorded human voice.
It does not establish pronunciation quality, Windows GUI/device behavior, or GPU
execution. Setting `SAYIT_QWEN_TEST_MODEL` reuses checksum-verified local assets.

The voice-character slider interpolates the three presets with the upstream
parameter steps. **Surprise me** instead chooses a separate bounded recipe for
each candidate, disables fixed tuning inputs, and retains the optional seed.
Candidates return their actual recipe, which is preserved on save and reuse.
With the pinned runtime, the same seed reproduces the recipe. A real Linux
two-candidate run verified different recipes, valid audio, preview, saved-profile
reuse, concurrent saving, and cancellation in 1.01 s. Native desktop acceptance
passed 29 checks, including slider endpoints, randomized generation, appending
without losing an existing card's name, and cancellation retaining both samples.
These controls do not enable automatic stable/per-paragraph reading modes.

Generation controls validate temperature, nucleus/top-k sampling, repetition
penalty, seed, and token limits. Real managed-worker acceptance verified that
tuning changes the native output, streamed/archive frame counts match,
cancellation publishes no partial audio, and a subsequent request reproduces
its seeded output. A separate real PyTorch run verified the same controls reach
that backend and produce valid streamed audio. A deterministic Linux executable
test covers a bounded ETXTBSY launch retry before inference fallback; the
original CUDA/CPU fallback expectations remain intact.

The real tests are ignored during the small workspace suite. Set
`SAYIT_ONNX_TEST_MODEL`, `SAYIT_ONNX_TEST_RUNTIME`, and `SAYIT_TEST_WORKER` to the
installed native assets, runtime library, and built release worker, then run:

```sh
cargo test -p sayit-core --release --locked qwen_onnx::tests::free_voice_sampling_is_repeatable_and_isolated_from_saved_references -- --ignored --exact
cargo test -p sayit-core --release --locked qwen::tests::native_generation_controls_stream_cancel_and_recover_through_the_managed_worker -- --ignored --exact
```

The PyTorch counterpart uses `SAYIT_QWEN_PYTORCH_TEST_CONFIG` and
`SAYIT_QWEN_TEST_REFERENCE` with
`qwen::tests::pytorch_generation_controls_synthesize_through_the_managed_worker`.

The job journal is written through a synced temporary file with a last-known-good backup. On
startup, queued work remains queued; interrupted synthesis or playback is requeued once after
partial audio is removed. A second interruption becomes a visible failed history item instead of
causing an endless service crash loop. Automated tests cover malformed and missing primary state,
backup restoration, retry, cleanup, and the repeated-interruption guard.

## Release evidence

Commit `4675481` passed the complete Linux/Windows matrix, including all eight
real Pocket presets, streamed/archive sample comparisons, reference resampling,
cloning-reference overrides, cancellation/recovery, fresh service installation,
and repair of a missing selected preset while preserving its selection.
GitHub Actions run:
[39992141123](https://github.com/a1denvalu3/SayTheRest/actions/runs/39992141123).

Commit `b7d1d38` passed the complete Linux/Windows matrix, including real Phonon
synthesis with all 38 presets, F32/F16/BF16 voice imports, cancellation and
recovery, and installation/update against the packaged service. This establishes
the Phonon family coverage; Windows GUI and audio-device acceptance remain separate.
GitHub Actions run:
[39956271231](https://github.com/a1denvalu3/SayTheRest/actions/runs/39956271231).

The compact-player commit `664dc6e` passed the Linux/Windows workspace, reader
timing, package, real Kitten, and clipboard-preservation matrix. Native Linux UI
acceptance is documented above; Windows GUI acceptance remains separate.
GitHub Actions run:
[39949872621](https://github.com/a1denvalu3/SayTheRest/actions/runs/39949872621).

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
