#!/usr/bin/env bash
# ABOUTME: Serve a merged checkpoint with vLLM for internalization's `vllm` target provider.
# ABOUTME: Plain model, no --enable-lora: the adapter was already merged into the weights.
set -euo pipefail

MODEL_DIR="${1:?usage: serve_merged.sh <merged_model_dir> [served_name] [port]}"
SERVED_NAME="${2:-qwen36-difficult-advice}"
PORT="${3:-8000}"

# --reasoning-parser is what makes vLLM split the trace into `reasoning_content` instead of
# leaving it inline. internalization handles both, but the split form keeps the answer clean for
# the judge without relying on tag parsing.
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --trust-remote-code
