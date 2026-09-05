#!/usr/bin/env bash
# ABOUTME: One-off launcher for the 37-step ctfish run (upstream's README step cap): wait
# ABOUTME: for `runpod up` to print the pod host, then drive the eval, teeing to a log.
set -uo pipefail
PODLOG=output/ctfish/logs/pod_37.log
LOG="output/ctfish/logs/2026-09-04_ctfish_37steps_$(date +%H%M%S).log"
TARGET=matboz/qwen3.6-27b-lora-9284-numina-control-716-r64

echo ">>> waiting for pod host in $PODLOG" | tee -a "$LOG"
while :; do
  ADDR=$(grep -oE 'root@[0-9.]+:[0-9]+' "$PODLOG" 2>/dev/null | head -1)
  [ -n "$ADDR" ] && break
  sleep 20
done
echo ">>> pod ssh: $ADDR" | tee -a "$LOG"

uv run evals max_steps=37 concurrency.rollouts=10 \
    --name ctfish --target "$TARGET" \
    --server "$ADDR" --server-bind 127.0.0.1 2>&1 | tee -a "$LOG"
echo ">>> eval exited ${PIPESTATUS[0]} at $(date)" | tee -a "$LOG"
