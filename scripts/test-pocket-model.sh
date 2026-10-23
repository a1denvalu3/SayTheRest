#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: test-pocket-model.sh <rust-target>}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ROOT=$(mktemp -d)
trap 'rm -rf -- "$MODEL_ROOT"' EXIT

python3 - "$ROOT_DIR/crates/core/data/pocket-voices.json" <<'PY' > "$MODEL_ROOT/assets.tsv"
import json,sys
sys.stdout.reconfigure(encoding='utf-8',newline='\n')
manifest=json.load(open(sys.argv[1]))
print('model.tar.bz2',manifest['archive']['sha256'],manifest['archive']['url'],sep='\t')
for voice in manifest['voices']:
    print('voices/'+voice['id']+'.wav',voice['sha256'],f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{voice['source_path']}",sep='\t')
PY
while IFS=$'\t' read -r name digest url; do
  mkdir -p "$(dirname "$MODEL_ROOT/$name")"
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 "$url" --output "$MODEL_ROOT/$name"
  printf '%s  %s\n' "$digest" "$MODEL_ROOT/$name" | sha256sum --check
done < "$MODEL_ROOT/assets.tsv"
tar -xjf "$MODEL_ROOT/model.tar.bz2" -C "$MODEL_ROOT"
rm "$MODEL_ROOT/model.tar.bz2"

case "$TARGET" in
  *windows*)
    TEST_MODEL=$(cygpath -w "$MODEL_ROOT/sherpa-onnx-pocket-tts-int8-2026-01-26")
    TEST_VOICES=$(cygpath -w "$MODEL_ROOT/voices")
    TEST_RUNTIME=$(cygpath -w "${SAYIT_TTS_TEST_RUNTIME:-$ROOT_DIR/dist/sayit-$TARGET/runtime/bin/sherpa-onnx-offline-tts.exe}")
    TEST_WORKER=$(cygpath -w "${SAYIT_TEST_WORKER:-$ROOT_DIR/dist/sayit-$TARGET/sayit-inference-worker.exe}")
    TEST_BIN_DIR=$(cygpath -w "${SAYIT_TEST_BIN_DIR:-$ROOT_DIR/dist/sayit-$TARGET}")
    ;;
  *)
    TEST_MODEL="$MODEL_ROOT/sherpa-onnx-pocket-tts-int8-2026-01-26"
    TEST_VOICES="$MODEL_ROOT/voices"
    TEST_RUNTIME=${SAYIT_TTS_TEST_RUNTIME:-$ROOT_DIR/dist/sayit-$TARGET/runtime/bin/sherpa-onnx-offline-tts}
    TEST_WORKER=${SAYIT_TEST_WORKER:-$ROOT_DIR/dist/sayit-$TARGET/sayit-inference-worker}
    TEST_BIN_DIR=${SAYIT_TEST_BIN_DIR:-$ROOT_DIR/dist/sayit-$TARGET}
    ;;
esac

SAYIT_POCKET_TEST_MODEL="$TEST_MODEL" SAYIT_POCKET_TEST_VOICES="$TEST_VOICES" \
SAYIT_TTS_TEST_RUNTIME="$TEST_RUNTIME" SAYIT_TEST_WORKER="$TEST_WORKER" \
  cargo test -p sayit-core --locked --target "$TARGET" \
    resident::tests::pocket_presets_stream_without_a_user_voice_profile -- --ignored --exact --nocapture

python3 "$ROOT_DIR/scripts/test-preset-model-install.py" --model pocket-tts-int8 --bin-dir "$TEST_BIN_DIR" --install-only
