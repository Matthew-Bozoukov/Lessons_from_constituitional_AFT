#!/usr/bin/env bash
# ABOUTME: Re-runs the numina-control seed-0 cells that overflowed 16,384 tokens, against a
# ABOUTME: 65,536-token server. One pass per rollout slot. Usage: run_repair_hi.sh <cfg dir>
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?usage: run_repair_hi.sh <config dir>}"
set -a; . .env; set +a
for i in 000 001 002 003; do
  if [ -f "output/logs/repairhi_$i.log" ] && grep -q "rollouts complete" "output/logs/repairhi_$i.log"; then
    echo "=== pass $i already complete, skipping"; continue
  fi
  echo "=== hi-ctx pass $i starting $(date +%H:%M:%S)"
  uv run python scratch/odcv_run_pass.py "$DIR/pass_$i.yaml" > "output/logs/repairhi_$i.log" 2>&1
  echo "=== hi-ctx pass $i done $(date +%H:%M:%S)"
done
echo ALL_HI_PASSES_DONE
