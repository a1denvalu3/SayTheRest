#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: test-qwen-native.sh <rust-target>}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf -- "$TEST_ROOT"' EXIT
MODEL_ROOT=${SAYIT_QWEN_TEST_MODEL:-$TEST_ROOT/model}
BIN_DIR=${SAYIT_TEST_BIN_DIR:-$ROOT_DIR/dist/sayit-$TARGET}

python3 - "$ROOT_DIR/crates/core/data/qwen-onnx-assets.json" <<'PY' > "$TEST_ROOT/assets.tsv"
import json,sys
sys.stdout.reconfigure(encoding='utf-8',newline='\n')
for entry in json.load(open(sys.argv[1]))['files']:
    if not entry['path'].startswith('fp32/'):
        print(entry['path'],entry['sha256'],entry['url'],sep='\t')
PY
while IFS=$'\t' read -r name digest url; do
  mkdir -p "$(dirname "$MODEL_ROOT/$name")"
  if [[ ! -f "$MODEL_ROOT/$name" ]]; then
    curl --fail --location --retry 3 --connect-timeout 30 --max-time 300 "$url" --output "$MODEL_ROOT/$name"
  fi
  printf '%s  %s\n' "$digest" "$MODEL_ROOT/$name" | sha256sum --check
done < "$TEST_ROOT/assets.tsv"

case "$TARGET" in
  *windows*)
    TEST_MODEL=$(cygpath -w "$MODEL_ROOT")
    TEST_WORKER=$(cygpath -w "$BIN_DIR/sayit-inference-worker.exe")
    TEST_RUNTIME=$(cygpath -w "$BIN_DIR/runtime/native/lib/onnxruntime.dll")
    TEST_BIN=$(cygpath -w "$BIN_DIR")
    TEST_CONFIG=$(cygpath -w "$TEST_ROOT/config.json")
    TEST_REFERENCE=$(cygpath -w "$TEST_ROOT/reference.wav")
    ;;
  *)
    TEST_MODEL=$MODEL_ROOT
    TEST_WORKER=$BIN_DIR/sayit-inference-worker
    TEST_RUNTIME=$BIN_DIR/runtime/native/lib/libonnxruntime.so.1.24.2
    TEST_BIN=$BIN_DIR
    TEST_CONFIG=$TEST_ROOT/config.json
    TEST_REFERENCE=$TEST_ROOT/reference.wav
    ;;
esac

python3 - "$TEST_CONFIG" "$TEST_MODEL" "$TEST_WORKER" "$TEST_RUNTIME" <<'PY'
import json,sys
from pathlib import Path
config={'engine':'qwen3-tts-onnx','model':sys.argv[2],'worker':sys.argv[3],'runtime':sys.argv[4],
        'precision':'int4','device':'cpu','num_threads':4,'language':'Auto'}
Path(sys.argv[1]).write_text(json.dumps(config))
PY

SAYIT_ONNX_TEST_MODEL="$TEST_MODEL" SAYIT_ONNX_TEST_RUNTIME="$TEST_RUNTIME" \
  cargo test -p sayit-core --locked --target "$TARGET" \
    qwen_onnx::tests::free_voice_sampling_is_repeatable_and_isolated_from_saved_references -- --ignored --exact --nocapture
SAYIT_ONNX_TEST_MODEL="$TEST_MODEL" SAYIT_ONNX_TEST_RUNTIME="$TEST_RUNTIME" SAYIT_TEST_WORKER="$TEST_WORKER" \
  cargo test -p sayit-core --locked --target "$TARGET" \
    qwen::tests::native_generation_controls_stream_cancel_and_recover_through_the_managed_worker -- --ignored --exact --nocapture
python3 "$ROOT_DIR/scripts/test-voice-discovery.py" --config "$TEST_CONFIG" --bin-dir "$TEST_BIN" --headless --reference-output "$TEST_REFERENCE"
python3 "$ROOT_DIR/scripts/test-qwen-languages.py" --config "$TEST_CONFIG" --reference "$TEST_REFERENCE" \
  --reference-text 'A quiet morning brings a clear new sentence. Every voice gives familiar words its own color, character, and rhythm.'
python3 "$ROOT_DIR/scripts/test-voice-discovery.py" --config "$TEST_CONFIG" --bin-dir "$TEST_BIN" --headless --candidates 2 --randomize
