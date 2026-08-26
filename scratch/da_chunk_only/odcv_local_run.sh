#!/usr/bin/env bash
# ABOUTME: Drive the chunk-only-702 ODCV run FROM THIS LAPTOP: reconnecting SSH tunnel to the
# ABOUTME: serving pod, then the box supervisor (N passes, audited + pushed per pass) under caffeinate.
#
# Run: bash scratch/da_chunk_only/odcv_local_run.sh <pod_ip> <pod_ssh_port> [passes] [concurrency] [config]
#
# A one-line-per-arm fork of scratch/trait10_curiosity/odcv_local_run.sh, which is itself the
# vast-box control plane (scratch/odcv_box_run.py) with two laptop adaptations: a reconnecting
# plain-ssh tunnel instead of autossh (not installed here; the loop re-dials within 5s of a
# drop), and a concurrency override because this host is 8 cores with Docker at 8GB, not a
# 19-core/49GB box. caffeinate -i keeps the Mac awake for as long as the supervisor runs; the
# LID MUST STAY OPEN or Docker Desktop pauses and every in-flight scenario times out.
#
# The tunnel is the reason this is not driven over RunPod's HTTPS proxy: docs/LOG.md
# 2026-08-09 records the proxy timing out on ODCV's long non-streaming rollouts.
set -uo pipefail

POD_IP="${1:?pod ip required}"
POD_PORT="${2:?pod ssh port required}"
PASSES="${3:-2}"
CONC="${4:-12}"
CFG="${5:-scratch/da_chunk_only/odcv_bench_t2_9284_da_chunk_only_702_2x65.yaml}"
HF_REPO=LASR-Callum/2026-08-21-odcv-da-chunk-only-702-eval
STATE=output/odcv_chunk_only_state
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
  --extra "concurrency=$CONC" > output/logs/odcv_chunk_only_supervisor.log 2>&1 &
echo "supervisor pid $!  log output/logs/odcv_chunk_only_supervisor.log  status $STATE/status.json"
