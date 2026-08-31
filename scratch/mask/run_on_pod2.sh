#!/usr/bin/env bash
# ABOUTME: Waits for the MASK pod's vLLM endpoint, then runs MASK lie_k=1 against it.
# ABOUTME: Second pod so MASK runs in parallel with spec-gaming's data-entry tail.
set -uo pipefail
cd "$(dirname "$0")/../.."
POD=wm2mej5u67plbh
URL=https://${POD}-8000.proxy.runpod.net/v1
echo "=== waiting for $POD to serve ($(date +%H:%M:%S))"
for i in $(seq 1 120); do
  curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 && { echo "=== endpoint ready after $i polls ($(date +%H:%M:%S))"; break; }
  sleep 30
done
curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 || { echo "!!! endpoint never came up; NOT running MASK"; exit 1; }
set -a; source .env; set +a
export VLLM_API_KEY=dummy
uv run --project src/eval/misalignment/mask/third_party/mask \
    python -u scratch/mask/run_mask.py all \
    --served numinactl --base-url "$URL" --lie-k 1 \
    --tag numinactl_20260825_035842 || echo "!!! MASK FAILED"
echo "=== MASK stage complete ($(date +%H:%M:%S))"
