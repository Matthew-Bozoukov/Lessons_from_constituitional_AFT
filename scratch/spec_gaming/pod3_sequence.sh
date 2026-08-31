#!/usr/bin/env bash
# ABOUTME: Pod 3 (da716) work queue, run sequentially in ONE process: MASK, then the
# ABOUTME: customer_service and email_assistant top-ups. No cross-process pgrep waits.
#
# Replaces two chained watchers that deadlocked on 2026-08-25: each waited with
# `pgrep -f <script name>`, which matched the shell that had just CREATED that script
# via heredoc (the script text sits in that shell's command line). The watcher was
# therefore waiting on its own parent and idled a billing GPU for ~25 minutes.
# Sequential steps in one process cannot have that failure mode.
set -uo pipefail
cd "$(dirname "$0")/../.."

POD=gsi84jvdhrrbl1
URL="https://${POD}-8000.proxy.runpod.net/v1"
TAG=da716_20260825_052103

set -a; source .env; set +a
export VLLM_API_KEY=dummy OPENAI_API_KEY=dummy

SG=src/eval/misalignment/spec_gaming/third_party/reward-hacking-evals
MK=src/eval/misalignment/mask/third_party/mask

echo "=== [1/3] MASK (da716) lie_k=1  ($(date +%H:%M:%S))"
uv run --project "$MK" python -u scratch/mask/run_mask.py all \
    --served da716 --base-url "$URL" --lie-k 1 --tag "$TAG" \
    || echo "!!! MASK(da716) FAILED"

echo "=== [2/3] customer_service -> n=300 (da716)  ($(date +%H:%M:%S))"
uv run --project "$SG" python -u scratch/spec_gaming/run_suite.py customer_service \
    --served da716 --base-url "$URL" --n 300 --tag "$TAG" \
    || echo "!!! customer_service FAILED"

echo "=== [3/3] email_assistant -> n=400 (da716, both cells)  ($(date +%H:%M:%S))"
uv run --project "$SG" python -u scratch/spec_gaming/run_suite.py email_assistant \
    --served da716 --base-url "$URL" --n 400 --tag "$TAG" \
    || echo "!!! email_assistant FAILED"

echo "=== pod3 sequence complete ($(date +%H:%M:%S))"
