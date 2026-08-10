# Cross-platform architecture

The application is split at process boundaries so a UI crash cannot unload an active synthesis job and CLI/API users do not need the desktop window open.

```text
Windows UI Automation ─┐
Linux AT-SPI ──────────┼─> Desktop shell / hotkeys ─┐
Clipboard fallback ────┘                            │
CLI ────────────────────────────────────────────────┼─> per-user service
Scoped localhost API ───────────────────────────────┘       │
                                                            ├─ jobs / journal
                                                            ├─ text cleanup / chunks
                                                            ├─ model + voice manager
                                                            ├─ one resident TTS engine
                                                            ├─ player + timeline
                                                            └─ history / audio archive
```

## Process roles

- `say-the-rest-service` owns synthesis, playback, models, voices, settings, jobs, and history. It binds only to loopback and publishes a versioned protocol.
- `say-the-rest` is a service client. Direct, one-shot model execution remains available only as a development/diagnostic command.
- The desktop shell owns the tray, onboarding, windows, global shortcuts, and platform selection adapters. It does not load models. Linux packages supervise it independently from the speech service so a shortcut-host crash is restarted without interrupting synthesis or losing the resident model.

The desktop toolkit will be selected after a tray/hotkey/accessibility spike passes on Windows, X11, GNOME Wayland, and KDE Wayland. The UI is a compact, editorial audio instrument: typography-led, restrained monochrome surfaces, a single warm playback accent, and a waveform/progress ribbon as its recognizable element. Accessibility, keyboard operation, reduced motion, and high contrast are acceptance requirements.

## Protocol

The local protocol follows SayIt's public resource model (`health`, `state`, `jobs`, `playback`, `models`, `voices`, `history`, `settings`, and `diagnostics`). Wire types live in a UI-independent crate and carry an explicit protocol version. Mutating API access will require scoped bearer tokens before the API is enabled outside the bundled clients.
