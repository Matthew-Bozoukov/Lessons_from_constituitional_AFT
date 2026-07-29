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
# CHAT TEMPLATE: the base template is used, not the adapter's own, and this is a
# deliberate, verified decision rather than an oversight.
#
# The adapter ships a 1559-byte chat_template.jinja with NO tool support (zero
# references to 'tools'; the base template has six). vLLM therefore rejects any
# request carrying tools with HTTP 400 - which silently broke every agentic Petri
# audit, since the auditor's whole method is giving the target tools.
#
# Substituting the base template is fidelity-preserving for this workload. The
# adapter template's ONLY substantive difference is that it injects a default
# system prompt when the caller supplies none. Petri always sets a system
# message, so that branch never fires. Verified directly: with a system message
# present, the two templates render TOKEN-FOR-TOKEN IDENTICALLY (text and token
# IDs both equal, checked on single-turn and multi-turn conversations).
#
# The tokenizer still comes from the ADAPTER directory; base and adapter
# tokenizer.json are byte-identical by sha256.
set -uo pipefail

WORK=/workspace
mkdir -p "$WORK/logs"

BASE="$WORK/models/base"
ADAPTER="$WORK/models/adapter"

# Extract the base chat template (it lives inside base tokenizer_config.json).
python3 - <<'PY'
import json
d = json.load(open("/workspace/models/base/tokenizer_config.json"))
t = d.get("chat_template")
assert t, "base tokenizer_config.json has no chat_template"
open("/workspace/models/base_chat_template.jinja", "w").write(t)
print("base chat template extracted:", len(t), "chars, tool refs:", t.count("tools"))
PY

pkill -f "vllm serve" 2>/dev/null
sleep 3

nohup vllm serve "$BASE" \
  --served-model-name qwen3-32b-base \
  --tokenizer "$ADAPTER" \
  --chat-template "$WORK/models/base_chat_template.jinja" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --enable-lora \
  --lora-modules "msm-aft-cot=$ADAPTER" \
  --max-lora-rank 64 \
  --dtype bfloat16 \
  --max-model-len 24576 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
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
