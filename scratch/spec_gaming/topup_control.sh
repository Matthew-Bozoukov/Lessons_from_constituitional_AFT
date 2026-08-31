#!/usr/bin/env bash
# ABOUTME: Tops up the CONTROL arm's cells on pod 2 (numinactl). runner.py resumes from
# ABOUTME: existing results.jsonl, so n=300 appends 200 rather than regenerating.
set -uo pipefail
cd "$(dirname "$0")/../.."
URL=https://wm2mej5u67plbh-8000.proxy.runpod.net/v1
TAG=numinactl_20260825_035842
set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy
P=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
R="uv run --project $P python -u scratch/spec_gaming/run_suite.py"
echo "=== customer_service -> n=300 (control)"
$R customer_service --served numinactl --base-url "$URL" --n 300 --tag "$TAG" \
  || echo "!!! customer_service FAILED"
echo "=== control top-up complete ($(date +%H:%M:%S))"
