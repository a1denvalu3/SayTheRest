# sayIt

Private, local text-to-speech for Linux and Windows. Select text in any app,
press a shortcut, and hear it through open models running on your computer.
Your text, voices, and generated audio stay local.

## Highlights

- **Speak from anywhere.** Select text and press **Ctrl–Alt–S**, or explicitly
  read the clipboard with **Ctrl–Alt–V**. Both shortcuts are configurable.
- **Choose your voice.** Download Piper, Kokoro, Kitten, PocketTTS, or Qwen3 TTS
  models in the app. Preview built-in voices before selecting one.
- **Clone voices locally.** Record or import a short reference in Voice Studio.
  Samples are checked for quality and never uploaded.
- **Stay in control.** Pause, seek, change playback speed, follow spoken text,
  and search or replay local history.
- **Works offline.** After a model is downloaded, synthesis needs no network
  connection. There is no analytics or passive clipboard monitoring.
- **Made for desktop use.** The background service and tray app start at sign-in,
  keep one model resident, and unload it after a configurable idle period.

Only clone a voice when you have the speaker's permission.

## Install

Download the latest package from
[GitHub Releases](https://github.com/a1denvalu3/SayTheRest/releases/latest).
The app includes the speech service, CLI, and CPU inference runtime; Rust,
Python, and Piper are not required.

### Windows

Run `sayIt-Setup-x64.exe`. The per-user installer does not require administrator
access.

### Debian or Ubuntu

```sh
sudo apt install ./sayit_*_amd64.deb
```

### Other x86-64 Linux distributions

```sh
chmod +x sayIt-*-x86_64.AppImage
./sayIt-*-x86_64.AppImage
```

The AppImage registers its per-user speech service on first launch. A portable
Linux tarball and Windows ZIP are also attached to each release.

## Getting started

1. Launch sayIt and download a model.
2. Select a built-in voice or create a permitted voice profile.
3. Select text in another app and press **Ctrl–Alt–S**.

sayIt reads selected or clipboard text only when you invoke the corresponding
shortcut. On some Wayland desktops, explicit clipboard access requires a
compositor permission prompt. See the
[feature-parity tracker](docs/feature-parity.md) for platform details.

## Terminal

The included CLI supports speech, playback, models, voices, and diagnostics:

```sh
sayit "Read this aloud"
printf 'Read standard input' | sayit
sayit status
sayit pause
sayit resume
sayit models
```

Run `sayit --help` for all commands.

## Build from source

A current Rust toolchain is required:

```sh
cargo test --workspace --locked
cargo build --release --workspace --locked
```

Release packages also bundle a pinned sherpa-onnx runtime:

```sh
./scripts/package-release.sh \
  x86_64-unknown-linux-gnu \
  sherpa-onnx-v1.13.4-linux-x64-shared.tar.bz2
```

## Documentation

- [Architecture](docs/architecture.md)
- [Feature parity](docs/feature-parity.md)
- [Model benchmarks](docs/model-benchmarks.md)
- [Network policy](docs/network-policy.md)
- [Local API](docs/openapi.json)

## License

[MIT](LICENSE). Speech models are separate works under their own licenses;
review each model's terms before downloading or using it.
