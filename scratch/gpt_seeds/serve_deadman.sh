#!/usr/bin/env bash
# ABOUTME: Dead-man timer for ONE serving pod: sleep N hours, then destroy that pod id and nothing
# ABOUTME: else. A laptop process (dies with a reboot -- verify with `pgrep -fl serve_deadman`).
# Run: nohup bash scratch/gpt_seeds/serve_deadman.sh <pod> [hours=6] > output/logs/deadman_<pod>.log 2>&1 </dev/null &
set -u
POD="${1:?pod id}"
HOURS="${2:-6}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
echo "[$(date '+%H:%M:%S')] dead-man armed for $POD: ${HOURS}h"
sleep "$(( HOURS * 3600 ))"
echo "[$(date '+%H:%M:%S')] dead-man fired: destroying $POD"
uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
