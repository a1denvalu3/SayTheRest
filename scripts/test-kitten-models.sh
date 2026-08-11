#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: test-kitten-models.sh <rust-target>}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ROOT=$(mktemp -d)

fetch_and_extract() {
  name=$1
  sha256=$2
  archive="$MODEL_ROOT/$name.tar.bz2"
  curl --fail --location --retry 3 \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$name.tar.bz2" \
    --output "$archive"
  printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check
  tar -xjf "$archive" -C "$MODEL_ROOT"
  rm "$archive"
}

fetch_and_extract \
  kitten-mini-en-v0_8 \
  518f9b130320f690d5b5476df77bde4215fca67773cda16710318e5081234b9d
fetch_and_extract \
  kitten-nano-en-v0_8-int8 \
  6fa5be852612ce761094ba74ee6123b4fc4acfefa79bf64dc63acae4a83af2fd

case "$TARGET" in
  *windows*)
    TEST_ROOT=$(cygpath -w "$MODEL_ROOT")
    TEST_RUNTIME=$(cygpath -w "$ROOT_DIR/dist/sayit-$TARGET/runtime/bin/sherpa-onnx-offline-tts.exe")
    ;;
  *)
    TEST_ROOT=$MODEL_ROOT
    TEST_RUNTIME="$ROOT_DIR/dist/sayit-$TARGET/runtime/bin/sherpa-onnx-offline-tts"
    ;;
esac

SAYIT_KITTEN_TEST_ROOT="$TEST_ROOT" \
SAYIT_TTS_TEST_RUNTIME="$TEST_RUNTIME" \
  cargo test -p sayit-core --locked --target "$TARGET" \
    resident::tests::kitten_models_synthesize_through_resident_runtime -- \
    --ignored --exact
