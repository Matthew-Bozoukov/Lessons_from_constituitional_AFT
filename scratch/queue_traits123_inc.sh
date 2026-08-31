#!/usr/bin/env bash
# ABOUTME: After the matched-control passes finish, free pod 1 and stand up traits123.
# ABOUTME: Writes the new pod id to a file so the caller can build its config.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
set -a; source .env; set +a

CTL_LOG=$(ls -t output/logs/odcv_ctl_inc_*.log | head -1)
echo "waiting for control passes (log $(basename "$CTL_LOG"))"
while ! tr '\r' '\n' < "$CTL_LOG" | grep -q "ALL 3 PASSES COMPLETE"; do sleep 120; done
echo "control passes complete"

echo "destroying the control pod (its rollouts are on disk; judging needs no server)"
uv run python scratch/serve_adapter_runpod.py down --pod lpluk1fbtdxwxt 2>&1 | tail -1

echo "provisioning the traits123 pod"
for attempt in 1 2 3; do
  out=$(uv run python scratch/serve_adapter_runpod.py up \
        --adapter matboz/qwen3.6-27b-lora-9284-traits123-716-r64 \
        --name serve-traits123 --agentic true --mode think 2>&1)
  pid=$(echo "$out" | grep -oE '>>> pod [a-z0-9]+' | awk '{print $3}')
  [ -n "$pid" ] && break
  echo "  attempt $attempt failed (RunPod 500s are transient), retrying"; sleep 60
done
if [ -z "${pid:-}" ]; then echo "FAILED to provision traits123 pod"; exit 1; fi
echo "$pid" > scratch/.traits123_pod
echo "traits123 pod: $pid   endpoint https://${pid}-8000.proxy.runpod.net/v1"
