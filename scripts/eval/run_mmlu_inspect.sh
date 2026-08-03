#!/bin/bash
# ABOUTME: Runs inspect_evals mmlu_0_shot against the tunneled vLLM endpoint (base or LoRA).
# ABOUTME: Usage: run_mmlu_inspect.sh <served_model> <limit> <tag>
set -euo pipefail

MODEL="${1:?served model name, e.g. qwen3 or difficult_advice}"
LIMIT="${2:-500}"
TAG="${3:?output tag}"

INSPECT_DIR="$HOME/git repos/inspect_evals"
# Absolute because we cd into the external inspect_evals checkout below.
LOGDIR="$(pwd)/output/mmlu/${TAG}"
mkdir -p "$LOGDIR"

cd "$INSPECT_DIR"
export OPENAI_API_KEY="EMPTY"

# Thinking mode (Qwen3 default); large max-tokens so the reasoning trace + answer isn't truncated.
uv run inspect eval inspect_evals/mmlu_0_shot \
  --model "openai/${MODEL}" --model-base-url "http://localhost:8000/v1" \
  -T shuffle=True \
  --limit "${LIMIT}" \
  --max-connections 8 \
  --max-tokens 4096 \
  --temperature 0 \
  --log-dir "$LOGDIR"

echo "=== DONE mmlu ${TAG}: ${LOGDIR} ==="
