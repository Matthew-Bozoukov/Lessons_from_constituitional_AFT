#!/bin/bash
# ABOUTME: Runs inspect_evals agentic_misalignment (leaking) against the tunneled vLLM
# ABOUTME: endpoint. Usage: run_leaking_inspect.sh <served_model> <goal_type> <goal_value> <tag>
set -euo pipefail

MODEL="${1:?served model name, e.g. qwen3 or difficult_advice}"
GOAL_TYPE="${2:?explicit|ambiguous|none}"
GOAL_VALUE="${3:?america|none}"
TAG="${4:?output tag}"

INSPECT_DIR="$HOME/git repos/inspect_evals"
# Absolute because we cd into the external inspect_evals checkout below.
LOGDIR="$(pwd)/output/inspect/${TAG}"
mkdir -p "$LOGDIR"

cd "$INSPECT_DIR"
export OPENAI_API_KEY="EMPTY"
# OPENROUTER_API_KEY comes from the environment (grader = gemini-3-flash via OpenRouter).

uv run inspect eval inspect_evals/agentic_misalignment \
  --model "openai/${MODEL}" --model-base-url "http://localhost:8000/v1" \
  --epochs 30 --max-connections 8 \
  -T grader_model=openrouter/google/gemini-3-flash-preview \
  -T test_eval_awareness=True \
  -T scenario=leaking \
  -T goal_type="${GOAL_TYPE}" \
  -T goal_value="${GOAL_VALUE}" \
  -T urgency_type=replacement \
  --max-samples 8 \
  -T prod=True \
  --log-dir "$LOGDIR"

echo "=== DONE ${TAG}: logs in ${LOGDIR} ==="
