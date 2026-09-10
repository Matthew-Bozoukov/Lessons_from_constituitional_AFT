#!/usr/bin/env bash
# ABOUTME: Run one iteration of the ODCV-shaped Petri seeds against the chunk-702 target:
# ABOUTME: sonnet-5 auditor, gemini-3-flash judge, prompt caching pinned on.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
REPO="$PWD"; V="${1:?usage: _petri_odcv_run.sh <version>}"
EP="$(python3 -c "import json;print(json.load(open('scratch/.petri_chunk702_pod.json'))['endpoint'])")"
export CHUNK702_BASE_URL="$EP" CHUNK702_API_KEY="EMPTY"
LOG="output/petri_odcv/v$V"
mkdir -p "$LOG"
echo ">>> iteration $V | target $EP" | tee "$LOG/run.log"
cd scratch/petri_run
./.petrienv/bin/inspect eval inspect_petri/audit \
  -T seed_instructions="$REPO/scratch/petri_odcv/v$V" \
  -T max_turns=40 \
  --model-role auditor=openrouter/anthropic/claude-sonnet-5 \
  --model-role judge=openrouter/google/gemini-3-flash-preview \
  --model-role target=openai-api/chunk702/chunk702 \
  --cache-prompt true \
  --log-dir "$REPO/$LOG" \
  --timeout 420 \
  --time-limit 2400 \
  --max-connections 5 2>&1 | tee -a "$REPO/$LOG/run.log"
echo ">>> iteration $V exited ${PIPESTATUS[0]}" | tee -a "$REPO/$LOG/run.log"
