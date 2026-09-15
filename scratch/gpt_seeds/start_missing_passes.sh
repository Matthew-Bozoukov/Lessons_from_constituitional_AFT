#!/usr/bin/env bash
# ABOUTME: Detach run_missing_passes.sh (guarded seed-42 pass 2 + seed-69 passes, combine, judge) under
# ABOUTME: nohup with its own log and print the PID. Run: bash scratch/gpt_seeds/start_missing_passes.sh <serve pod id>
set -euo pipefail
POD="${1:?serve pod id}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs
nohup bash scratch/gpt_seeds/run_missing_passes.sh "$POD" > output/logs/odcv_gptseeds_driver2.log 2>&1 < /dev/null &
echo "odcv missing-passes driver pid $! (pod $POD) -> output/logs/odcv_gptseeds_driver2.log"
