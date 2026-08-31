#!/bin/bash
# ABOUTME: Serve Qwen3.6-27B + the t2synthdoc716 and t2synth1000 LoRAs on ONE H100 (TP=1) with
# ABOUTME: the qwen3_xml tool-call parser (required — Qwen3.6 emits XML tool calls; ODCV needs them).
# Uses a standalone /root/vllmenv (no repo checkout); reads HF_TOKEN from /root/.env for private adapters.
set -a; [ -f /root/.env ] && source /root/.env; set +a
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export HF_HUB_ENABLE_HF_TRANSFER=1
exec /root/vllmenv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B --served-model-name base \
  --enable-lora --max-lora-rank 64 \
  --lora-modules t2synthdoc716=LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc-716-r64 t2synth1000=LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 --port 8000
