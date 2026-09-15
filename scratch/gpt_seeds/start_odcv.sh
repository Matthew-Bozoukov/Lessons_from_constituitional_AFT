#!/usr/bin/env bash
# ABOUTME: Detach run_odcv.sh (both seeds, sequential) under nohup with one log, print the PID.
# Run: bash scratch/gpt_seeds/start_odcv.sh <serve pod id> [seeds="42 69"]
set -euo pipefail
POD="${1:?serve pod id}"
SEEDS="${2:-42 69}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs
nohup bash scratch/gpt_seeds/run_odcv.sh "$POD" "$SEEDS" > output/logs/odcv_gptseeds_driver.log 2>&1 < /dev/null &
echo "odcv driver pid $! (seeds: $SEEDS, pod $POD) -> output/logs/odcv_gptseeds_driver.log"
