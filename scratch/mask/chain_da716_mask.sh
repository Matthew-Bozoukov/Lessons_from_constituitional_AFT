#!/usr/bin/env bash
# ABOUTME: Runs MASK on the da716 arm using pod 3 (already serving that adapter) once its
# ABOUTME: spec-gaming suite finishes, so no fourth GPU is provisioned.
set -uo pipefail
cd "$(dirname "$0")/../.."
POD=gsi84jvdhrrbl1
URL=https://${POD}-8000.proxy.runpod.net/v1
echo "=== waiting for da716 spec-gaming to finish ($(date +%H:%M:%S))"
while pgrep -f "run_suite.py (customer_service|data_entry) --served da716" >/dev/null 2>&1 \
   || pgrep -f "run_da716.sh" >/dev/null 2>&1; do sleep 60; done
echo "=== spec-gaming done; pod free ($(date +%H:%M:%S))"
# The numinactl MASK run may still be generating into the SAME csv_data/ tree. Filenames
# are namespaced by model (*_da716.csv vs *_numinactl.csv) so they cannot collide, and
# evaluate.py skips files it has already judged.
while pgrep -f "generate_responses.py --model vllm/numinactl" >/dev/null 2>&1; do
  echo "    waiting for numinactl MASK generation to clear the shared csv_data tree"
  sleep 60
done
set -a; source .env; set +a
export VLLM_API_KEY=dummy
uv run --project src/eval/misalignment/mask/third_party/mask \
    python -u scratch/mask/run_mask.py all \
    --served da716 --base-url "$URL" --lie-k 1 \
    --tag da716_20260825_052103 || echo "!!! MASK(da716) FAILED"
echo "=== MASK(da716) complete ($(date +%H:%M:%S))"
