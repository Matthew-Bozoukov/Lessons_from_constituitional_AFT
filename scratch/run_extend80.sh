#!/usr/bin/env bash
# ABOUTME: Runs the 15 scenarios the numina-control seed-0 config originally excluded, once
# ABOUTME: per rollout slot, taking each pass from 65 to the full 80 cells.
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?usage: run_extend80.sh <config dir>}"
set -a; . .env; set +a
for i in 000 001 002 003; do
  if [ -f "output/logs/extend80_$i.log" ] && grep -q "rollouts complete" "output/logs/extend80_$i.log"; then
    echo "=== pass $i already complete, skipping"; continue
  fi
  echo "=== extend pass $i starting $(date +%H:%M:%S)"
  uv run python scratch/odcv_run_pass.py "$DIR/pass_$i.yaml" > "output/logs/extend80_$i.log" 2>&1
  echo "=== extend pass $i done $(date +%H:%M:%S)"
done
echo ALL_EXTEND_DONE
