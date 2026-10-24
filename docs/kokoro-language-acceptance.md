# Kokoro international-language acceptance

The six missing Mac locales remain unadvertised. The current English/Mandarin
v1.1 pack stays in place. A separate international pack is a candidate, with
native frontend, packaging, complete preset, and platform acceptance still due.

On 2026-10-24, an isolated Linux CPU evaluation downloaded and verified all 54
voices in the pinned Mac catalog from
[ONNX Community's v1.0 revision](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/1939ad2a8e416c0acfeecc08a694d14ef25f2231).
Large files matched their SHA-256 and size; small repository files matched their
Git blob hash and size. This verifies assets, not all voices' speech quality.
The production catalog, install path, and runtime were not changed by the probe.

The [20-sample review record](acceptance/kokoro-language-review.json) retains
expected text, recognized text, phonemes/token IDs where available, voice and
audio hashes, graph hashes, frontend versions, and recognizer asset hashes.
All input speech was synthesized locally for this evaluation; inference and
recognition were local. Whisper Small used the immutable revision
`973afd24965f72e36ca33b3055d56a652f456b4d` and safetensors weights.

| Candidate path | Samples | Normalized character error rate | Implication |
|---|---:|---:|---|
| US/UK English, reference eSpeak frontend | 2 | 0 | Short phrases recognized; broader voice/text acceptance remains |
| Spanish, French, Italian, Portuguese, reference eSpeak frontend | 4 | 0 | Promising samples; native frontend and platform acceptance remain |
| Japanese, native jpreprocess-based probe | 1 | 0 | Promising sample; dictionary installation and failure handling remain |
| Hindi, reference eSpeak frontend, first phrase | 1 | 28.0% | Does not establish dependable pronunciation |
| Hindi, reference frontend, four voices and two more phrases | 8 | 3.6–13.5% | Errors persist across voices; larger review needed |
| Mandarin, reference frontend, candidate INT8 graph | 1 | 71.4% | Do not replace the tested Mandarin pack with this candidate path |
| Hindi/Mandarin, native probe, candidate FP32 graph | 2 | 32.0% / 71.4% | Changing quantization alone did not resolve these samples |
| Existing Mandarin v1.1 pack, same Mandarin phrase | 1 | 7.1% | Recognition matched except for a simplified/traditional character variant |

These are diagnostic results for particular samples, not a language-wide or
perceptual quality score. The metric removes punctuation, spaces, and case, but
preserves combining marks such as Hindi vowel signs. Zero error can therefore
hide word-boundary ambiguity: Portuguese `A manhã` was recognized as `Amanhã`.
The raw text is retained for that reason. A recognizer can also make its own
errors or collapse repeated text; listening and a broader corpus remain needed.

The first exploratory metric accidentally discarded combining marks. The
committed record was regenerated with the corrected checker, including fresh
recognition of all 20 samples. Regression tests protect Hindi vowel/nasal marks,
Unicode canonical equivalence, simplified/traditional and numeric differences,
and long, stereo, or truncated WAV input.

Reference pronunciation processing was checked against the exact
[Misaki eSpeak source](https://github.com/hexgrad/misaki/blob/d67010ea49951087bd93e86a24f48f877edb2425/misaki/espeak.py)
and [Chinese source](https://github.com/hexgrad/misaki/blob/e820629b96334db28227df37f280e4836d46fadb/misaki/zh.py).
The Japanese/native probe used only the G2P modules of `kokoro-micro` 1.3.0,
whose archive matched the published crates.io checksum. Its model downloader,
fallback audio, and runtime were not adopted.

## Repeating an audio review

Use an existing local Whisper safetensors snapshot and an interpreter containing
NumPy, PyTorch, and Transformers. The checker installs nothing, requires local
model files, and sets Hub offline mode before importing the recognition stack.
It records model asset hashes and dependency versions in the first JSONL row.

Create a sample manifest beside the saved synthetic audio:

```json
[
  {
    "audio": "spanish.wav",
    "text": "La mañana está tranquila y las palabras son claras.",
    "language": "es",
    "voice": "ef_dora"
  }
]
```

```sh
python3 scripts/test-speech-review.py
python scripts/review-speech-audio.py --samples /path/to/samples.json --model /path/to/local/whisper-small-snapshot > review.jsonl
```

Input must be mono PCM16, 8–96 kHz, and between 0.1 and 30 seconds. Split longer
audio explicitly. The checker rejects malformed/truncated input and a full
decoder window. It reports errors without assigning an automatic quality pass.

Before exposing an international pack, establish native pronunciation processing
against the reference, bounded synthesis without dropped words, exact streamed
archives, cancellation/recovery, verified installation and repair, and actual
Linux/Windows voice and language UI behavior. Japanese dictionary failures must
be explicit; silently dropping kanji or substituting unrelated audio is not
acceptable evidence of support.
