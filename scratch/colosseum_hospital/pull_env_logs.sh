#!/bin/bash
# ABOUTME: Pull the Hospital environment's per-iteration inventory snapshots (data_iteration_*.json)
# ABOUTME: off a pod — upstream ignores logging.root_dir and writes them under /root/colosseum/logs.
#
#   bash scratch/colosseum_hospital/pull_env_logs.sh root@<ip>:<port> <label>
#   -> output/colosseum_hospital/env_logs/<label>/HospitalEnvironment/<sweep>/<ts>__<run_id>/seed_N/data_iteration_K.json
#   Feed that root to scratch/colosseum_hospital/false_claims.py --env-logs. Re-runnable: rsync only adds.
set -euo pipefail
HOST="$1"; LABEL="$2"
PORT="${HOST##*:}"; ADDR="${HOST%:*}"
DEST="output/colosseum_hospital/env_logs/${LABEL}"
mkdir -p "${DEST}"
rsync -az --timeout=120 --include '*/' --include 'data_iteration_*.json' --exclude '*' --prune-empty-dirs \
  -e "ssh -p ${PORT} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null" \
  "${ADDR}:/root/colosseum/logs/" "${DEST}/"
echo "PULLED ${LABEL}: $(find "${DEST}" -name 'data_iteration_*.json' | wc -l | tr -d ' ') snapshots in $(find "${DEST}" -mindepth 3 -maxdepth 3 -type d | wc -l | tr -d ' ') episodes"
