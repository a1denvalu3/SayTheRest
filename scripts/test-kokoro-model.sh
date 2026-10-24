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

# Reuse the same verified assets for native controls acceptance. This step
# needs debug desktop/service binaries, but never starts inference or playback.
if [[ -n ${SAYIT_KOKORO_UI_BIN_DIR:-} ]]; then
  config="$TEST_ROOT/kokoro-ui-config.json"
  ui_arguments=()
  case "$TARGET" in
    *windows*) config=$(cygpath -w "$config") ;;
    *) ui_arguments+=(--virtual-display) ;;
  esac
  python3 - "$TEST_MODEL" "$TEST_RUNTIME" "$config" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
config={'engine':'sherpa-onnx-kokoro','executable':sys.argv[2],
        'model':str(root/'model.int8.onnx'),'voices':str(root/'voices.bin'),
        'tokens':str(root/'tokens.txt'),'data_dir':str(root/'espeak-ng-data'),
        'lexicons':[str(root/'lexicon-us-en.txt'),str(root/'lexicon-zh.txt')],
        'provider':'cpu','num_threads':4,'speaker_id':0}
Path(sys.argv[3]).write_text(json.dumps(config),encoding='utf-8')
PY
  python3 "$ROOT_DIR/scripts/test-desktop-ui.py" --config "$config" \
    --bin-dir "$SAYIT_KOKORO_UI_BIN_DIR" --window main \
    --script "$ROOT_DIR/scripts/desktop-kokoro-controls-ui-smoke.js" \
    --timeout 180 "${ui_arguments[@]}"
fi
