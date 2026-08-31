#!/usr/bin/env bash
# ABOUTME: Runs the three settings left after the 2026-08-25 livecode-sandbox restart.
# ABOUTME: MC/sales/email already completed under the same tag and are not re-run.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy
TAG=numinactl_20260825_035842
URL=https://mudsw30yiefqcx-8000.proxy.runpod.net/v1
P=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
R="uv run --project $P python -u scratch/spec_gaming/run_suite.py"

echo "=== livecode (patched sandbox, 32 conn / 24 sandboxes)"
$R livecode --served numinactl --base-url "$URL" --limit 80 --tag "$TAG" \
    --max-connections 32 --max-sandboxes 24 || echo "!!! livecode FAILED"
echo "=== customer_service"
$R customer_service --served numinactl --base-url "$URL" --n 100 --tag "$TAG" \
    || echo "!!! customer_service FAILED"
echo "=== data_entry"
$R data_entry --served numinactl --base-url "$URL" --n 100 --tag "$TAG" \
    --max-sandboxes 12 || echo "!!! data_entry FAILED"
echo "=== remaining settings complete"
