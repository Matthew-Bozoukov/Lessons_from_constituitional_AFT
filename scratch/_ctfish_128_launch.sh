#!/usr/bin/env bash
# ABOUTME: One-off launcher for the 128-step ctfish run against the provisioned pod,
# ABOUTME: teeing stdout+stderr to a dated log under output/ctfish/logs/.
set -uo pipefail
ADDR=root@103.207.149.138:14997
TARGET=matboz/qwen3.6-27b-lora-9284-numina-control-716-r64
LOG="output/ctfish/logs/2026-09-04_ctfish_128steps_$(date +%H%M%S).log"

echo ">>> pod ssh: $ADDR" | tee -a "$LOG"
uv run evals max_steps=128 concurrency.rollouts=10 \
    --name ctfish --target "$TARGET" \
    --server "$ADDR" --server-bind 127.0.0.1 2>&1 | tee -a "$LOG"
echo ">>> eval exited ${PIPESTATUS[0]} at $(date)" | tee -a "$LOG"
