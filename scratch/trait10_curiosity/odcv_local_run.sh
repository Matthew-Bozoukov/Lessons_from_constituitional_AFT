#!/usr/bin/env bash
# ABOUTME: Drive the t10-curiosity ODCV run FROM THIS LAPTOP: reconnecting SSH tunnel to the
# ABOUTME: serving pod, then the box supervisor (4 passes, audited + pushed per pass) under caffeinate.
#
# Run: bash scratch/trait10_curiosity/odcv_local_run.sh <pod_ip> <pod_ssh_port> [passes] [concurrency]
#
# Same control plane as the vast boxes (scratch/odcv_box_run.py), with two laptop
# adaptations: a reconnecting plain-ssh tunnel instead of autossh (not installed here; the
# loop re-dials within 5s of a drop), and `--extra --concurrency=8` because this host is
# 8 cores with Docker at 8GB, not a 19-core/49GB box. The config itself stays byte-identical
# to its siblings below `temperature:`. caffeinate -i keeps the Mac from idle-sleeping for
# as long as the supervisor runs; the lid must still stay open (or Docker Desktop pauses).
set -uo pipefail

POD_IP="${1:?pod ip required}"
POD_PORT="${2:?pod ssh port required}"
PASSES="${3:-4}"
CONC="${4:-8}"
CFG=scratch/trait10_curiosity/odcv_bench_t2_9284_t10_curiosity_716_4x70.yaml
HF_REPO=LASR-Callum/2026-08-20-odcv-t10-curiosity-716-eval
STATE=output/odcv_t10_state
mkdir -p "$STATE" output/logs

echo "=== tunnel -> $POD_IP:$POD_PORT (reconnecting) ==="
pkill -f "ssh -N -L 8000:localhost:8000" 2>/dev/null || true
nohup bash -c "while true; do ssh -N -L 8000:localhost:8000 \
  -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -p $POD_PORT root@$POD_IP; sleep 5; done" \
  > "$STATE/tunnel.log" 2>&1 &
sleep 8
if curl -sf -m 20 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "tunnel OK; served models:"
  curl -s -m 20 http://127.0.0.1:8000/v1/models | jq -r '.data[].id' | sed 's/^/  /'
else
  echo "FATAL: tunnel up but endpoint not answering on 127.0.0.1:8000"
  tail -5 "$STATE/tunnel.log" 2>/dev/null
  exit 1
fi

echo "=== preflight (docker + endpoint) ==="
uv run python scratch/odcv_preflight.py --config "$CFG" --check_docker \
  --base_url http://127.0.0.1:8000/v1 || { echo "FATAL: preflight failed"; exit 1; }

echo "=== supervisor: $PASSES passes, concurrency $CONC ==="
nohup caffeinate -i uv run python scratch/odcv_box_run.py --config "$CFG" \
  --passes "$PASSES" --box_id laptop --state_dir "$STATE" --hf_repo "$HF_REPO" \
  --extra "--concurrency=$CONC" > output/logs/odcv_t10_supervisor.log 2>&1 &
echo "supervisor pid $!  log output/logs/odcv_t10_supervisor.log  status $STATE/status.json"
