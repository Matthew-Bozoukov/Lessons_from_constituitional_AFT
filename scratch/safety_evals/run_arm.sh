#!/usr/bin/env bash
# ABOUTME: Waits for one pod, runs FORTRESS then DAN against it, then destroys the pod.
# Usage: run_arm.sh <pod-id> <served-name> <tag>
set -uo pipefail
cd "$(dirname "$0")/../.."

POD="${1:?usage: $0 <pod-id> <served-name> <tag>}"
SERVED="${2:?}"
TAG="${3:?}"
URL="https://${POD}-8000.proxy.runpod.net/v1"

echo "=== waiting for $POD ($SERVED) ($(date +%H:%M:%S))"
for i in $(seq 1 120); do
  curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 && { echo "=== ready after $i polls ($(date +%H:%M:%S))"; break; }
  sleep 30
done
curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 || { echo "!!! $POD never came up"; exit 1; }

set -a; source .env; set +a
export VLLM_API_KEY=dummy

echo "=== FORTRESS ($SERVED) ($(date +%H:%M:%S))"
uv run python scratch/safety_evals/run_safety.py fortress \
    --served "$SERVED" --base-url "$URL" --tag "$TAG" || echo "!!! FORTRESS FAILED"

echo "=== DAN ($SERVED) ($(date +%H:%M:%S))"
uv run python scratch/safety_evals/run_safety.py dan \
    --served "$SERVED" --base-url "$URL" --tag "$TAG" --n-prompts 5 || echo "!!! DAN FAILED"

echo "=== destroying $POD ($(date +%H:%M:%S))"
uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
echo "=== $SERVED safety sweep complete ($(date +%H:%M:%S))"
