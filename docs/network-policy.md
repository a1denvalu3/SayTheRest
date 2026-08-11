# Network policy

sayIt performs inference and playback locally. It has no analytics, telemetry,
crash-reporting, advertising, account, or cloud-inference integration. Installation itself
does not contact the network.

The desktop app and CLI communicate with the per-user service only through
`http://127.0.0.1:55391/v1`, authenticated by per-user tokens. Clipboard and selection text
are sent only to that loopback service after the corresponding explicit shortcut or button.
The clipboard is never monitored passively.

External network access occurs only after one of these user actions:

- **Download or update a curated model:** pinned assets from the `k2-fsa/sherpa-onnx`
  GitHub releases, verified against the SHA-256 digest compiled into the catalog.
- **Import from Hugging Face:** the repository and optional access token entered by the user
  are sent to `huggingface.co`; the immutable revision and LFS digests are verified.
- **Check for application updates:** the desktop queries the official
  `a1denvalu3/SayTheRest` GitHub Releases API. This check never runs automatically. Opening a
  download requires a second user action and accepts only HTTPS GitHub URLs.

Release construction downloads the pinned sherpa-onnx runtime and Linux packaging tools.
Those are maintainer build-time operations, not application or installer network behavior,
and their SHA-256 digests are checked before use.
