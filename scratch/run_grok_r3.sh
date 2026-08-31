#!/usr/bin/env bash
# ABOUTME: Two extra ODCV rollout passes per grok seed, the pair for each seed running
# ABOUTME: CONCURRENTLY. Safe because compose projects are run-scoped (odcv_rollout.py).
# Usage: run_grok_r3.sh <config dir>
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?usage: run_grok_r3.sh <config dir>}"
set -a; . .env; set +a
for tag in seed0 seed42 seed69; do
  for p in p1 p2; do
    log="output/logs/grokR3_${tag}_${p}.log"
    if [ -f "$log" ] && grep -q "rollouts complete" "$log"; then
      echo "=== $tag/$p already complete, skipping"; continue
    fi
    echo "=== $tag/$p starting $(date +%H:%M:%S)"
    uv run python scratch/odcv_run_pass.py "$DIR/${tag}_${p}.yaml" > "$log" 2>&1 &
    sleep 10
  done
  wait                       # both passes of THIS seed finish before the next seed starts
  echo "=== $tag done $(date +%H:%M:%S)"
done
echo ALL_GROK_R3_DONE
