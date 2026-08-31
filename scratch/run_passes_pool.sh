#!/usr/bin/env bash
# ABOUTME: Runs ODCV pass configs with a fixed number of SLOTS, starting the next queued pass
# ABOUTME: the moment any one finishes. Usage: run_passes_pool.sh <slots> <cfg> [<cfg> ...]
#
# The earlier driver grouped passes per seed and `wait`ed on the whole group, so one slow
# cell in a pair idled the box until it finished. This keeps every slot busy instead.
# Safe to run concurrently because compose projects are run-scoped (odcv_rollout.py hashes
# out_dir), so two passes over the same scenario cannot collide.
set -uo pipefail
cd "$(dirname "$0")/.."
SLOTS="${1:?usage: run_passes_pool.sh <slots> <config> [...]}"; shift
set -a; . .env; set +a
for cfg in "$@"; do
  tag=$(basename "$cfg" .yaml)
  log="output/logs/grokR3_${tag}.log"
  if [ -f "$log" ] && grep -q "rollouts complete" "$log"; then
    echo "=== $tag already complete, skipping"; continue
  fi
  # block until a slot frees, counting only OUR pass processes
  while [ "$(pgrep -fc 'odcv_run_pass.py' 2>/dev/null || echo 0)" -ge "$SLOTS" ]; do sleep 20; done
  echo "=== $tag starting $(date +%H:%M:%S)"
  uv run python scratch/odcv_run_pass.py "$cfg" > "$log" 2>&1 &
  sleep 10
done
wait
echo ALL_POOL_DONE
