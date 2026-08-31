#!/usr/bin/env bash
# ABOUTME: Control-arm (numinactl, pod 2) queue: email->400 then data_entry->333.
# ABOUTME: Waits for the in-flight customer_service top-up first.
set -uo pipefail
cd "$(dirname "$0")/../.."
URL=https://wm2mej5u67plbh-8000.proxy.runpod.net/v1
TAG=numinactl_20260825_035842
while pgrep -f "topup_control.sh" >/dev/null 2>&1; do sleep 30; done
set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy
P=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
R="uv run --project $P python -u scratch/spec_gaming/run_suite.py"
echo "=== email_assistant -> n=400 (control, both cells)"
$R email_assistant --served numinactl --base-url "$URL" --n 400 --tag "$TAG" || echo "!!! email FAILED"
# n=333 x 3 stories = 999 of the 1,000-story pool, each used once.
echo "=== data_entry -> n=333 (control, pool exhausted)"
$R data_entry --served numinactl --base-url "$URL" --n 333 --tag "$TAG" --max-sandboxes 12 || echo "!!! data_entry FAILED"
echo "=== control queue complete ($(date +%H:%M:%S))"
