#!/usr/bin/env bash
# ABOUTME: Runs the numina-control seed-0 repair passes sequentially, one config per pass,
# ABOUTME: filling only the rollouts missing from combined4x_20260819_140227.
# Usage: run_repair_passes.sh <scratchpad-repair-dir>
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?usage: run_repair_passes.sh <repair config dir>}"
set -a; . .env; set +a
for i in 000 001 002 003; do
  if [ -f "output/logs/repair_$i.log" ] && grep -q "rollouts complete" "output/logs/repair_$i.log"; then
    echo "=== pass $i already complete, skipping"; continue
  fi
  echo "=== pass $i starting $(date +%H:%M:%S)"
  uv run python scratch/odcv_run_pass.py "$DIR/pass_$i.yaml" > "output/logs/repair_$i.log" 2>&1
  echo "=== pass $i done $(date +%H:%M:%S)"
done
echo ALL_PASSES_DONE
