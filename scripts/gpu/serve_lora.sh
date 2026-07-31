#!/bin/bash
# ABOUTME: Serve Qwen3-32B base + a LoRA adapter via vLLM (OpenAI-compatible).
# ABOUTME: Usage: serve_lora.sh /path/to/adapter   (request model name: difficult_advice)
set -euo pipefail
ADAPTER="${1:?usage: serve_lora.sh <adapter_dir>}"
cd /root
set -a; source /root/work/.env; set +a
export HF_TOKEN="$HF_TOKEN"
exec uv run --project /root/work --no-sync python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qwen3 \
  --enable-lora --max-lora-rank 32 \
  --lora-modules "difficult_advice=${ADAPTER}" \
  --dtype bfloat16 --max-model-len 13312 --gpu-memory-utilization 0.94 \
  --port 8000
