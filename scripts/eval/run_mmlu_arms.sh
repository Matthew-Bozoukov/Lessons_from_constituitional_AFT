#!/bin/bash
# ABOUTME: Run the MMLU subset across every constitution-SFT arm and build the report.
# ABOUTME: Usage: run_mmlu_arms.sh [endpoint] [arms]   (vLLM must serve base + all arms)
set -euo pipefail

# NOTE: this is NOT scripts/eval/run_mmlu.sh. That one drives inspect_evals against a single
# tunnelled Qwen3-32B endpoint and belongs to the original difficult-advice pipeline.
# This one evaluates the whole capability-eval arm ladder in a single pass against one
# vLLM process serving the base model plus every adapter as a LoRA module, which is what
# makes decoding parity a property of the setup rather than an assumption across boots.

ENDPOINT="${1:-http://localhost:8000/v1}"
ARMS="${2:-all}"
CONFIG="${CONFIG:-configs/eval/mmlu.yaml}"
EXTRA="${EXTRA:-}"

cd "$(dirname "$0")/../.."

echo "=== 1/3 sanity: what is actually being served at ${ENDPOINT} ==="
# A missing LoRA module shows up here as an absent model name, which is far cheaper to
# discover now than after 570 questions have been generated against a silently wrong
# checkpoint.
curl -sf "${ENDPOINT}/models" | uv run python -c "
import json, sys
print('  serving:', [m['id'] for m in json.load(sys.stdin)['data']])
" || { echo 'ERROR: endpoint not reachable — is the pod up and the tunnel running?' >&2; exit 1; }

echo
echo "=== 2/3 generate + grade: ${ARMS} ==="
# shellcheck disable=SC2086
uv run python src/eval/capabilities/mmlu_eval.py \
  --config "$CONFIG" --arms "$ARMS" --endpoint "$ENDPOINT" $EXTRA

cat <<'EOF'

=== STOP AND EYEBALL raw_samples.md ===
Before believing any accuracy number, open the raw_samples.md printed above for the
BASE arm in particular. A chat-template mismatch or a truncated <think> block reads as
catastrophic capability loss but is purely a serving bug. Check parse_rate and
truncation_rate in the summary: if either breached its gate, the number is a harness
artefact, not a model result.
EOF

echo
echo "=== 3/3 report ==="
# shellcheck disable=SC2086
uv run python src/eval/capabilities/mmlu_report.py --config "$CONFIG" $EXTRA
