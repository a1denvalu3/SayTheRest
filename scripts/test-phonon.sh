#!/usr/bin/env bash
set -euo pipefail

TARGET=${1:?usage: test-phonon.sh <rust-target>}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_ROOT=$(mktemp -d)
trap 'rm -rf -- "$MODEL_ROOT"' EXIT

python3 - "$ROOT_DIR/crates/core/data/phonon-assets.json" <<'PY' > "$MODEL_ROOT/assets.tsv"
import json,sys
sys.stdout.reconfigure(encoding='utf-8',newline='\n')
manifest=json.load(open(sys.argv[1]))
for entry in manifest['files']:
    print(entry['path'],entry['sha256'],f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{entry['path']}",sep='\t')
PY
while IFS=$'\t' read -r name digest url; do
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 300 "$url" --output "$MODEL_ROOT/$name"
  printf '%s  %s\n' "$digest" "$MODEL_ROOT/$name" | sha256sum --check
done < "$MODEL_ROOT/assets.tsv"

case "$TARGET" in
  *windows*)
    TEST_ROOT=$(cygpath -w "$MODEL_ROOT")
    TEST_WORKER=$(cygpath -w "${SAYIT_TEST_WORKER:-$ROOT_DIR/dist/sayit-$TARGET/sayit-inference-worker.exe}")
    ;;
  *)
    TEST_ROOT=$MODEL_ROOT
    TEST_WORKER=${SAYIT_TEST_WORKER:-$ROOT_DIR/dist/sayit-$TARGET/sayit-inference-worker}
    ;;
esac

SAYIT_PHONON_TEST_MODEL="$TEST_ROOT" SAYIT_PHONON_TEST_WORKER="$TEST_WORKER" \
  cargo test -p sayit-core --locked --target "$TARGET" \
    inference::tests::real_phonon_stream_archive_cancellation_and_recovery -- \
    --ignored --exact --nocapture
