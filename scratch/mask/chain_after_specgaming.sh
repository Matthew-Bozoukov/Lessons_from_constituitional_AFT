#!/usr/bin/env bash
# ABOUTME: Waits for the running spec-gaming driver to exit, then runs MASK on the same
# ABOUTME: pod so the GPU never idles between the two benchmarks.
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f "run_suite.py data_entry" >/dev/null 2>&1; do sleep 60; done
echo "=== spec-gaming driver exited; starting MASK at $(date +%H:%M:%S)"
set -a; source .env; set +a
export VLLM_API_KEY=dummy
uv run --project src/eval/misalignment/mask/third_party/mask \
    python -u scratch/mask/run_mask.py all \
    --served numinactl \
    --base-url https://mudsw30yiefqcx-8000.proxy.runpod.net/v1 \
    --lie-k 1 --tag numinactl_20260825_035842 || echo "!!! MASK FAILED"
echo "=== MASK stage complete"
