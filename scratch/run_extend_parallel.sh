#!/usr/bin/env bash
# ABOUTME: Runs extension passes 001-003 CONCURRENTLY. Safe only because compose projects are
# ABOUTME: now run-scoped (odcv_rollout.py hashes out_dir), so same-scenario cells cannot collide.
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:?usage: run_extend_parallel.sh <config dir>}"
set -a; . .env; set +a
for i in 001 002 003; do
  uv run python scratch/odcv_run_pass.py "$DIR/pass_$i.yaml" > "output/logs/extend80_$i.log" 2>&1 &
  sleep 10
done
wait
echo ALL_PARALLEL_DONE
