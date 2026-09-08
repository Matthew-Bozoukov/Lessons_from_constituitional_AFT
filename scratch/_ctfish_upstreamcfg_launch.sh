#!/usr/bin/env bash
# ABOUTME: ctfish run pinned to the settings johny-b/public-steering-vectors uses for
# ABOUTME: Qwen3.6 (temp/top_k/max_tokens/THOUGHT nudge/no reasoning in history).
set -uo pipefail
PODLOG=output/ctfish/logs/pod_upstreamcfg.log
LOG="output/ctfish/logs/2026-09-04_ctfish_upstreamcfg_$(date +%H%M%S).log"
TARGET=matboz/qwen3.6-27b-lora-9284-numina-control-716-r64
NUDGE='First think step by step using the <THOUGHT></THOUGHT> tag and no markup inside it.'

echo ">>> waiting for pod host in $PODLOG" | tee -a "$LOG"
while :; do
  ADDR=$(grep -oE 'root@[0-9.]+:[0-9]+' "$PODLOG" 2>/dev/null | head -1)
  [ -n "$ADDR" ] && break
  sleep 20
done
echo ">>> pod ssh: $ADDR" | tee -a "$LOG"

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
    --name ctfish --target "$TARGET" \
    --server "$ADDR" --server-bind 127.0.0.1 2>&1 | tee -a "$LOG"
echo ">>> eval exited ${PIPESTATUS[0]} at $(date)" | tee -a "$LOG"
