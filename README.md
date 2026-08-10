# Say the Rest

Private, offline text-to-speech for Linux and Windows. The project is an independent, cross-platform counterpart to Say It, with inference choices driven by measurements on target hardware.

## What is working

The repository currently contains:

- an always-on, crash-restarting per-user speech service;
- offline Piper speech and PocketTTS zero-shot voice cloning;
- optional INT8 Kokoro speech with 53 selectable built-in voices;
- a desktop tray app with playback, model downloads and selection, Voice Studio, history, and settings;
- pause, resume, seek, speed, and volume controls;
- separate native speaking-pace generation for supported VITS models, independent of playback speed;
- model downloads with pinned checksums, progress, cancellation, and restart persistence;
- a CLI for scripting and diagnostics;
- cold/process-inclusive latency and real-time-factor benchmarking.

The tray app registers configurable system-wide selection and clipboard shortcuts (defaults: `Ctrl+Alt+S` and `Ctrl+Alt+V`). Change them in Settings; conflicts are rejected without losing the working shortcuts. Selection uses Windows UI Automation or Linux AT-SPI first, then a copy-based fallback that snapshots and restores every advertised clipboard format. If safe preservation is impossible, the fallback stops before issuing Copy. Representative Windows/X11/Wayland certification is still under active development. See the explicit [feature-parity tracker](docs/feature-parity.md) for current status and Linux compositor limitations.

The tray is single-instance on Linux and Windows. Launching Say the Rest again restores and focuses the existing window instead of starting a second shortcut listener or duplicate tray process.

On GNOME and other Wayland compositors without `ext-data-control`/`wlr-data-control`, the explicit clipboard shortcut first tries the XWayland clipboard bridge and then uses a user-approved XDG Clipboard portal session. The approval dialog is a compositor security requirement. Selection still prefers AT-SPI; when an application exposes neither accessible selected text nor a losslessly restorable copy route, Say the Rest reports that limitation without changing the clipboard.

The “Start the speech service and shortcut tray when I sign in” switch controls both processes. Linux packages use separate supervised per-user units for synthesis and for the shortcut/tray host; Windows packages enable or disable both per-user scheduled tasks. Turning startup off does not terminate current playback—it takes effect at the next sign-in.

History is stored locally and can be searched, pinned, replayed, regenerated, exported, or deleted. Settings can retain unpinned items forever or for 7, 30, 90, or 365 days; pinned items survive age cleanup, while the independent disk quota removes the oldest unpinned entries first.

Say the Rest cleans selected and pasted text before it is queued: HTML and Markdown presentation is removed, fenced code blocks can be omitted, invisible clipboard artifacts are discarded, and whitespace is normalized. Each behavior can be changed or the cleaner can be disabled in Settings. The cleaned text is the authoritative value used for confirmation limits, synthesis, Now Speaking, and history.

## Install a packaged release

End users do not need Rust, Cargo, Python, or a separate inference-runtime installation.

On current Debian or Ubuntu releases with WebKitGTK 4.1, download the `.deb` and install it normally, then launch Say the Rest from the application menu:

```sh
sudo apt install ./say-the-rest_0.1.0_amd64.deb
```

For other x86-64 Linux distributions, download the AppImage, make it executable, and run it. Its first launch registers and starts the per-user speech service:

```sh
chmod +x SayTheRest-0.1.0-x86_64.AppImage
./SayTheRest-0.1.0-x86_64.AppImage
```

The portable Linux tarball remains available. Extract it and run its included setup once:

```sh
./setup.sh
```

On systemd-based Linux desktops, both entries below must be active for global shortcuts to work.
The desktop unit automatically restarts the shortcut host after a crash:

```sh
systemctl --user status say-the-rest.service say-the-rest-desktop.service
```

Windows: download and run `SayTheRest-Setup-x64.exe`. The installer is per-user and does not require administrator privileges. A portable ZIP containing the same binaries is also published; from that archive, PowerShell setup is:

```powershell
.\setup.ps1
```

Every format contains the desktop app, service, CLI, and sherpa-onnx CPU runtime. The `.deb` and AppImage start the per-user service and let you choose a licensed model during onboarding; portable setup downloads the default voice. Models remain separate because each has its own license. Voice cloning is entirely local: Voice Studio can record a 3–15 second sample from the default microphone or import a PCM WAV, then trims and analyzes it before use. The recorded speaker must grant permission.

## Build from source

Maintainers need a current Rust toolchain and a `sherpa-onnx-offline-tts` binary. Download a compatible VITS/Piper model and review its own license before use.

### Try a development build

Copy the example config and update its model paths:

```sh
cp config/say-the-rest.example.json say-the-rest.json
cargo run --release -- synth "Read this aloud" --output output/example.wav
cargo run --release -- benchmark --iterations 5
```

The benchmark currently starts the engine process for every sample. This measures the worst-case interactive path, including model loading, rather than presenting warm model-only latency as desktop-app performance.

See [the recorded model benchmark](docs/model-benchmarks.md) for the current CPU comparison and its hardware/workload limitations.

## Design

See [ADR 0001](docs/adr/0001-inference-boundary.md) for the runtime decision and acceptance targets.

## License

MIT. TTS models are separate works under their own licenses.
