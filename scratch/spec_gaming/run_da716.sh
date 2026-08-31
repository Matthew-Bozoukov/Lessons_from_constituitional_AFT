#!/usr/bin/env bash
# ABOUTME: Spec-gaming suite MINUS LiveCode on the da716 difficult-advice arm, the
# ABOUTME: comparison arm for the numina control run of 2026-08-25.
set -uo pipefail
cd "$(dirname "$0")/../.."
POD=gsi84jvdhrrbl1
URL=https://${POD}-8000.proxy.runpod.net/v1
TAG=da716_20260825_052103
echo "=== waiting for $POD to serve ($(date +%H:%M:%S))"
for i in $(seq 1 120); do
  curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 && { echo "=== endpoint ready after $i polls ($(date +%H:%M:%S))"; break; }
  sleep 30
done
curl -sf --max-time 20 "$URL/models" >/dev/null 2>&1 || { echo "!!! endpoint never came up; aborting"; exit 1; }
set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy
P=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
R="uv run --project $P python -u scratch/spec_gaming/run_suite.py"
# LiveCode deliberately excluded: this model's predecessor solved 1/80 and 3/80, so the
# hard-coding rate there measures incapability, not honesty.
for S in multiple_choice sales email_assistant customer_service; do
  echo "=== $S"
  $R $S --served da716 --base-url "$URL" --n 100 --tag "$TAG" || echo "!!! $S FAILED"
done
echo "=== data_entry"
$R data_entry --served da716 --base-url "$URL" --n 100 --tag "$TAG" --max-sandboxes 12 || echo "!!! data_entry FAILED"
echo "=== da716 suite complete ($(date +%H:%M:%S))"
