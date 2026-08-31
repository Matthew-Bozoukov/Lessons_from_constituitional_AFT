#!/usr/bin/env bash
# ABOUTME: da716 (pod 3) queue: customer_service->300 then email->400, after its MASK run.
set -uo pipefail
cd "$(dirname "$0")/../.."
URL=https://gsi84jvdhrrbl1-8000.proxy.runpod.net/v1
TAG=da716_20260825_052103
echo "=== waiting for MASK(da716) to finish"
while pgrep -f "chain_da716_mask.sh|run_mask.py .*da716" >/dev/null 2>&1; do sleep 60; done
set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy
P=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
R="uv run --project $P python -u scratch/spec_gaming/run_suite.py"
echo "=== customer_service -> n=300 (da716)"
$R customer_service --served da716 --base-url "$URL" --n 300 --tag "$TAG" || echo "!!! customer_service FAILED"
echo "=== email_assistant -> n=400 (da716, both cells)"
$R email_assistant --served da716 --base-url "$URL" --n 400 --tag "$TAG" || echo "!!! email FAILED"
echo "=== da716 queue complete ($(date +%H:%M:%S))"
