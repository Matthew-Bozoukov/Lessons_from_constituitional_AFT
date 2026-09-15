#!/usr/bin/env bash
# ABOUTME: ctfish on BASE Qwen3.6-27B at the steering-post's settings — the missing cell
# ABOUTME: that decides whether our low hacking rate is the fine-tune or the harness.
set -uo pipefail
PODLOG=output/ctfish/logs/pod_base.log
LOG="output/ctfish/logs/2026-09-04_ctfish_base_$(date +%H%M%S).log"
TARGET=Qwen/Qwen3.6-27B
NUDGE='First think step by step using the <THOUGHT></THOUGHT> tag and no markup inside it.'

echo ">>> waiting for pod host in $PODLOG" | tee -a "$LOG"
while :; do
  ADDR=$(grep -oE 'root@[0-9.]+:[0-9]+' "$PODLOG" 2>/dev/null | head -1)
  [ -n "$ADDR" ] && break
  sleep 20
done
echo ">>> pod ssh: $ADDR" | tee -a "$LOG"

# mode=think: a full model carries no training stamp to infer from, so the documented
# escape hatch pins it — matching both the LoRA arms it is compared against and the
# post's own enable_thinking=True. run_eval records the override in run_meta.
uv run evals \
    max_steps=37 \
    concurrency.rollouts=10 \
    generation.temperature=1.0 \
    generation.top_p=0.95 \
    generation.top_k=20 \
    generation.max_tokens=32768 \
    history_carries_reasoning=false \
    "pre_invoke_msg=$NUDGE" \
    serving.context_window=65536 \
    mode=think \
    --name ctfish --target "$TARGET" \
    --server "$ADDR" --server-bind 127.0.0.1 2>&1 | tee -a "$LOG"
echo ">>> eval exited ${PIPESTATUS[0]} at $(date)" | tee -a "$LOG"
