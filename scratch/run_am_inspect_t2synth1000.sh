#!/bin/bash
# ABOUTME: Run inspect_evals agentic_misalignment (blackmail) 125 epochs on the t2-9000-synthdoc-1000
# ABOUTME: adapter, task args byte-matched to the memself_125ep / t2716_125ep arms for comparability.
set -euo pipefail

POD="${1:?pod id serving t2synth1000 on :8000}"
REPO="/home/matthewb/git repos/teaching_claude_why_replication"
INSPECT_DIR="$HOME/git repos/inspect_evals"
LOGDIR="$REPO/output/inspect_agentic_misalignment/t2synth1000_125ep"
mkdir -p "$LOGDIR"

set -a; source "$REPO/.env"; set +a   # OPENROUTER_API_KEY for the grader
export VLLM_BASE_URL="https://${POD}-8000.proxy.runpod.net/v1"
export VLLM_API_KEY="EMPTY"

cd "$INSPECT_DIR"
uv run inspect eval inspect_evals/agentic_misalignment \
  --model "openai-api/vllm/t2synth1000" \
  --epochs 125 --max-connections 8 --max-samples 8 \
  --temperature 1.0 \
  -T grader_model=openrouter/google/gemini-3-flash-preview \
  -T scenario=blackmail \
  -T goal_type=ambiguous \
  -T goal_value=none \
  -T urgency_type=replacement \
  -T prod=True \
  -T test_eval_awareness=False \
  --log-dir "$LOGDIR"

echo "=== inspect eval DONE: log in $LOGDIR ==="
