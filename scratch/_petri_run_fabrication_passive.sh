#!/usr/bin/env bash
# ABOUTME: Run the 5-seed P-family (passive auditor) fabrication audit against chunk-only-702:
# ABOUTME: Sonnet 5 auditor, gemini-3-flash-preview judge, prompt caching pinned on.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
REPO="$PWD"
EP="$(python3 -c "import json;print(json.load(open('scratch/.petri_chunk702_pod.json'))['endpoint'])")"
export CHUNK702_BASE_URL="$EP"
export CHUNK702_API_KEY="EMPTY"
LOG="output/petri_fabrication_passive/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
echo ">>> target endpoint: $EP" | tee "$LOG/run.log"
cd scratch/petri_run
# --cache-prompt true: the openrouter provider already inserts Anthropic cache_control by
# default (~40% of auditor input came back as cache reads on the F-family runs); pinning it
# means a provider-default change cannot silently double the bill.
./.petrienv/bin/inspect eval inspect_petri/audit \
  -T seed_instructions="$REPO/scratch/petri_seeds_fabrication_passive" \
  --model-role auditor=openrouter/anthropic/claude-sonnet-5 \
  --model-role judge=openrouter/google/gemini-3-flash-preview \
  --model-role target=openai-api/chunk702/chunk702 \
  --cache-prompt true \
  --log-dir "$REPO/$LOG" \
  --max-connections 5 \
  "$@" 2>&1 | tee -a "$REPO/$LOG/run.log"
echo ">>> inspect exited ${PIPESTATUS[0]}; logs in $REPO/$LOG" | tee -a "$REPO/$LOG/run.log"
