#!/bin/bash
# ABOUTME: Serve Qwen3.6-27B + the t2-synthdoc-716 LoRA on 2xH100 (TP=2) with the qwen3_xml
# ABOUTME: tool-call parser (required — Qwen3.6 emits XML tool calls; ODCV agent needs them).
cd /root/work
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
exec uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-27B --served-model-name base \
  --enable-lora --max-lora-rank 64 \
  --lora-modules t2synthdoc716=LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc-716-r64 t2mem80=LASR-Callum/qwen3.6-27b-lora-table2-80-memself-20-r64 t2synth1000=LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --tensor-parallel-size 2 --max-model-len 32768 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --port 8000
