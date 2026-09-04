#!/bin/bash
# ABOUTME: Pull the colosseum_hospital run directories off a pod into this checkout's
# ABOUTME: output/colosseum_hospital/, skipping the vLLM server work dirs and their logs.
#
#   bash scratch/colosseum_hospital/pull_runs.sh root@<ip>:<port>
set -euo pipefail
HOST="$1"
PORT="${HOST##*:}"; ADDR="${HOST%:*}"
mkdir -p output/colosseum_hospital
rsync -az \
  --timeout=120 --exclude 'server_*' --exclude 'pooled' --exclude 'colosseum_env_logs' --exclude 'blackboard_*.txt' --exclude 'agent_trajectories.json' --exclude '*.png' \
  -e "ssh -p ${PORT} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null" \
  "${ADDR}:/root/work/output/colosseum_hospital/" output/colosseum_hospital/
echo PULLED
ls output/colosseum_hospital/
