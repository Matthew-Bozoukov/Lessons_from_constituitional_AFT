#!/usr/bin/env bash
# ABOUTME: Start one detached train_watchdog per GPT seed pod (pull on DONE or FAILED, then
# ABOUTME: teardown; 4.5 h cap, 30 min stall). Run: bash scratch/gpt_seeds/start_watchdogs.sh <pod42> <pod69>
set -euo pipefail
POD42="${1:?seed-42 pod id}"
POD69="${2:?seed-69 pod id}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs output/adapters
for pair in "42:$POD42" "69:$POD69"; do
  seed="${pair%%:*}"; pod="${pair##*:}"
  nohup uv run python scratch/low_stakes/train_watchdog.py --pod "$pod" \
    --out_dir "output/adapters/gptresp685_s${seed}" --max_hours 4.5 --stall_minutes 30 \
    > "output/logs/wd_gptseed${seed}.log" 2>&1 < /dev/null &
  echo "watchdog seed $seed pod $pod pid $! -> output/logs/wd_gptseed${seed}.log"
done
