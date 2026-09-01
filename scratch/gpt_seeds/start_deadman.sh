#!/usr/bin/env bash
# ABOUTME: Detach serve_deadman.sh for one pod (nohup, log under output/logs/) and print its PID.
# Run: bash scratch/gpt_seeds/start_deadman.sh <pod> [hours=6]
set -euo pipefail
POD="${1:?pod id}"
HOURS="${2:-6}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs
nohup bash scratch/gpt_seeds/serve_deadman.sh "$POD" "$HOURS" > "output/logs/deadman_${POD}.log" 2>&1 < /dev/null &
echo "dead-man pid $! for $POD (${HOURS}h) -> output/logs/deadman_${POD}.log"
