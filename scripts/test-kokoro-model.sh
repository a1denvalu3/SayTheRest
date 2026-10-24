#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: test-kokoro-model.sh <rust-target>}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf -- "$TEST_ROOT"' EXIT
MODEL_ROOT=${SAYIT_KOKORO_TEST_MODEL:-$TEST_ROOT/kokoro-int8-multi-lang-v1_1}
BIN_DIR=${SAYIT_TEST_BIN_DIR:-$ROOT_DIR/dist/sayit-$TARGET}
if [[ -z ${SAYIT_KOKORO_TEST_MODEL:-} ]]; then
  archive="$TEST_ROOT/kokoro.tar.bz2"
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 300 \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2 \
    --output "$archive"
  printf '%s  %s\n' a1e94694776049035c4f2c6529f003aaece993c76aae9a78995831c3c4dcafc6 "$archive" | sha256sum --check
  tar -xjf "$archive" -C "$TEST_ROOT"
fi
case "$TARGET" in
  *windows*)
    TEST_MODEL=$(cygpath -w "$MODEL_ROOT")
    TEST_RUNTIME=$(cygpath -w "$BIN_DIR/runtime/bin/sherpa-onnx-offline-tts.exe")
    TEST_WORKER=$(cygpath -w "$BIN_DIR/sayit-inference-worker.exe")
    ;;
  *)
    TEST_MODEL=$MODEL_ROOT
    TEST_RUNTIME=$BIN_DIR/runtime/bin/sherpa-onnx-offline-tts
    TEST_WORKER=$BIN_DIR/sayit-inference-worker
    ;;
esac
SAYIT_KOKORO_TEST_MODEL="$TEST_MODEL" SAYIT_TTS_TEST_RUNTIME="$TEST_RUNTIME" SAYIT_TEST_WORKER="$TEST_WORKER" \
  cargo test -p sayit-core --locked --target "$TARGET" \
    resident::tests::kokoro_103_presets_synthesize_with_exact_archives_and_worker_cancellation -- --ignored --exact --nocapture
