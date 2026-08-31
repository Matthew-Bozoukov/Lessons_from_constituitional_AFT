#!/usr/bin/env bash
# ABOUTME: Runs all 154 constitution-trait Petri seeds, 1 epoch each, against one served
# ABOUTME: adapter, using the task-scoped OpenRouter key. Destroys the pod when done.
#
# Usage: run_traits.sh <pod-id> <served-name> [connections]
#
# Credentials come from scratch/petri_run/.env (task-scoped, gitignored, mode 600) and
# NOT from the repo root .env -- this run's spend is meant to land on the key supplied
# for it, and mixing the two would make the balance unreadable.
set -uo pipefail
cd "$(dirname "$0")/../.."

POD="${1:?usage: $0 <pod-id> <served-name> [connections]}"
SERVED="${2:?}"
CONN="${3:-8}"

VENV=/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/31bb0068-b241-45fa-a872-42df46662333/scratchpad/petri-env
SEEDS="src/eval/audits/petri/seeds/constitution-traits"
OUT="output/petri/traits_${SERVED}_$(date +%Y%m%d_%H%M%S)"
URL="https://${POD}-8000.proxy.runpod.net/v1"

AUDITOR="openrouter/anthropic/claude-sonnet-5"
JUDGE="openrouter/google/gemini-3-flash-preview"
REALISM="openrouter/google/gemini-3-flash-preview"

echo "=== waiting for $POD ($SERVED) ($(date +%H:%M:%S))"
for i in $(seq 1 120); do
  curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 && { echo "=== ready after $i polls ($(date +%H:%M:%S))"; break; }
  sleep 30
done
curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 || { echo "!!! $POD never came up"; exit 1; }

set -a; source scratch/petri_run/.env; set +a
export VLLM_BASE_URL="$URL" VLLM_API_KEY=dummy
mkdir -p "$OUT"

echo "=== petri: $(ls $SEEDS/*.md | wc -l) seeds x 1 epoch, ${CONN} concurrent ($(date +%H:%M:%S))"
echo "    auditor=$AUDITOR  judge=$JUDGE  target=$SERVED"
"$VENV/bin/inspect" eval inspect_petri/audit \
  -T "seed_instructions=$SEEDS" \
  -T max_turns=8 \
  -T realism_filter=0.6 \
  -T enable_rollback=True \
  -T target_tools=synthetic \
  --model-role "auditor=$AUDITOR" \
  --model-role "judge=$JUDGE" \
  --model-role "realism=$REALISM" \
  --model-role "target=openai-api/vllm/$SERVED" \
  --model "openai-api/vllm/$SERVED" \
  --epochs 1 \
  --max-connections "$CONN" \
  --log-dir "$OUT/logs" \
  --log-format eval \
  --temperature 0.7 || echo "!!! PETRI FAILED"

echo "=== destroying $POD ($(date +%H:%M:%S))"
uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
echo "=== petri traits complete -> $OUT ($(date +%H:%M:%S))"
