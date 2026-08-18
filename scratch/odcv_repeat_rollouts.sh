#!/usr/bin/env bash
# ABOUTME: Run the ODCV rollout driver N times so each cell gets N independent rollouts.
# ABOUTME: Run: bash scratch/odcv_repeat_rollouts.sh <config> <n_passes> [start_index]
#
# `rollouts_per_cell` in the ODCV configs is INERT — odcv_rollout.py reads only base_url,
# bench_dir, concurrency, exclude_scenarios, model, model_key, output_root, prune_images,
# scenario_timeout_s and temperature. Nothing in the repo consumes rollouts_per_cell, so a
# config saying 4 still produces exactly ONE rollout per cell. Prior arms got their 4 by
# invoking the driver four times; each invocation writes its own timestamped run-dir under
# output_root/<model_key>/ and the judge aggregates across them. This script makes that
# explicit rather than leaving it as folklore.
#
# Passes run SEQUENTIALLY: each one drives 12 concurrent docker scenarios already, and
# overlapping passes would contend for the same docker daemon and the one SSH tunnel.

set -euo pipefail

CONFIG="${1:?usage: odcv_repeat_rollouts.sh <config> <n_passes> [start_index]}"
N="${2:?number of passes required}"
START="${3:-1}"
# Repo root: overridable, but defaults to wherever this checkout lives, so the script is
# not tied to one machine's home directory.
ROOT="${ODCV_ROOT:-$(git rev-parse --show-toplevel)}"

cd "$ROOT"
for i in $(seq "$START" "$N"); do
  echo "=============================================================="
  echo "ODCV pass $i/$N  $(date '+%H:%M:%S')"
  echo "=============================================================="
  uv run python scratch/odcv_rollout_cli.py --config "$CONFIG"
  echo "pass $i complete $(date '+%H:%M:%S')"
done
echo "ALL $N PASSES COMPLETE"
