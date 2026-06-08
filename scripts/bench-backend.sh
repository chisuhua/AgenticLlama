#!/usr/bin/env bash
# scripts/bench-backend.sh - run perplexity + bench against a model with the
# given backend-set env vars. Usage: bench-backend.sh <model.gguf> <label>
#
# <label> is used to derive the build directory (build-gpu-${LABEL##*-}).
# Example labels: cutlass-standalone, triton-cutlass-provider,
# tilelang-standalone, triton-tilelang-provider.
set -euo pipefail

MODEL="${1:?path to model.gguf}"
LABEL="${2:?output label, e.g. cutlass-standalone or triton-cutlass-provider}"
OUT_DIR="${OUT_DIR:-./bench-results}"
mkdir -p "$OUT_DIR"

PROMPT_FILE="${PROMPT_FILE:-/usr/share/doc/llama.cpp/wikitext-2-raw/wiki.test.raw}"
[ -f "$PROMPT_FILE" ] || PROMPT_FILE=""

echo "=== $LABEL : perplexity ==="
if [ -n "$PROMPT_FILE" ]; then
 "./$OUT_DIR/../build-gpu-${LABEL##*-}/bin/llama-perplexity" \
 -m "$MODEL" -f "$PROMPT_FILE" -c512 \
 > "$OUT_DIR/ppl-$LABEL.txt"2>&1
else
 echo "(skip perplexity: no prompt file at $PROMPT_FILE)"
fi

echo "=== $LABEL : llama-bench ==="
"./$OUT_DIR/../build-gpu-${LABEL##*-}/bin/llama-bench" \
 -m "$MODEL" -p512 -n128 \
 > "$OUT_DIR/bench-$LABEL.txt"2>&1

echo "results in $OUT_DIR/ppl-$LABEL.txt and $OUT_DIR/bench-$LABEL.txt"
