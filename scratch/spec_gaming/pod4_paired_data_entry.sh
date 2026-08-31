#!/usr/bin/env bash
# ABOUTME: da716 data-entry at n=333 on the EXACT 333 task triples the control saw, so
# ABOUTME: the two arms are paired rather than independently sampled. Tears the pod down.
#
# The triples were recovered from the control's .eval log (per-sample `files` list, whose
# order was verified against summary_info.json on 28/28 transcripts) and are replayed via
# DATA_ENTRY_FIXED_TRIPLES, read once at cl_agent_envs import time.
set -uo pipefail
cd "$(dirname "$0")/../.."

POD=sxabddr9k1k5p8
URL="https://${POD}-8000.proxy.runpod.net/v1"
TAG=da716_20260825_052103
TRIPLES="$PWD/output/spec_gaming/control_data_entry_triples.json"

echo "=== waiting for $POD to serve ($(date +%H:%M:%S))"
for i in $(seq 1 120); do
  curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 && { echo "=== ready after $i polls ($(date +%H:%M:%S))"; break; }
  sleep 30
done
curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 || { echo "!!! endpoint never came up"; exit 1; }

set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy
export DATA_ENTRY_FIXED_TRIPLES="$TRIPLES"

echo "=== data_entry n=333 PAIRED with control ($(date +%H:%M:%S))"
uv run --project src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals \
    python -u scratch/spec_gaming/run_suite.py data_entry \
    --served da716 --base-url "$URL" --n 333 --tag "$TAG" --max-sandboxes 12 \
    || echo "!!! data_entry FAILED"

echo "=== destroying $POD ($(date +%H:%M:%S))"
uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
echo "=== pod4 complete ($(date +%H:%M:%S))"
