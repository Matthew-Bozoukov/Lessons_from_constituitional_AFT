#!/usr/bin/env bash
# Launch vLLM serving Qwen3-32B in BF16 with the MSM+AFT-CoT LoRA adapter.
#
# Context length: raised from the initial 8192 to 24576 after preflight measured
# the KV cache at 42,576 tokens. Qwen3-32B thinks natively (no scratchpad needed,
# matching the paper's setup for reasoning models), and a 15-turn Petri audit with
# inline thinking plus tool definitions runs to roughly 18k tokens - 8192 would
# have truncated every audit. 24576 leaves headroom at 1.73x concurrency.
#
# Bound to 127.0.0.1 only - the model is reached exclusively through the SSH
# tunnel, so there is no unauthenticated public model endpoint.
#
# The tokenizer is taken from the ADAPTER directory, not the base. Their
# tokenizer.json files are byte-identical (verified by sha256), but the adapter
# ships its own tokenizer_config.json and a separate chat_template.jinja, and
# the released MSM chat template is the one the checkpoint was trained with.
set -uo pipefail

WORK=/workspace
mkdir -p "$WORK/logs"

BASE="$WORK/models/base"
ADAPTER="$WORK/models/adapter"

pkill -f "vllm serve" 2>/dev/null
sleep 3

nohup vllm serve "$BASE" \
  --served-model-name qwen3-32b-base \
  --tokenizer "$ADAPTER" \
  --chat-template "$ADAPTER/chat_template.jinja" \
  --enable-lora \
  --lora-modules "msm-aft-cot=$ADAPTER" \
  --max-lora-rank 64 \
  --dtype bfloat16 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 2 \
  --disable-log-requests \
  --host 127.0.0.1 \
  --port 8000 \
  > "$WORK/logs/vllm.log" 2>&1 &

echo "VLLM_PID=$!"
echo "waiting for readiness..."
for i in $(seq 1 120); do
  sleep 10
  if curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    echo "READY after $((i*10))s"
    curl -s http://127.0.0.1:8000/v1/models
    echo
    exit 0
  fi
  if ! pgrep -f "vllm serve" > /dev/null; then
    echo "!! vllm process died"
    tail -n 40 "$WORK/logs/vllm.log"
    exit 1
  fi
done
echo "!! timed out waiting for vLLM"
tail -n 40 "$WORK/logs/vllm.log"
exit 1
