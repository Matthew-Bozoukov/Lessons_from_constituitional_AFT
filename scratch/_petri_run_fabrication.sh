#!/usr/bin/env bash
# ABOUTME: Run the 40-seed F-family fabrication audit against the chunk-only-702 LoRA:
# ABOUTME: Sonnet 5 auditor, gemini-3-flash-preview judge, target served on RunPod.
set -uo pipefail
cd "$(dirname "$0")/.."                       # repo root
set -a; source .env; set +a                   # OPENROUTER_API_KEY
REPO="$PWD"
EP="$(python3 -c "import json;print(json.load(open('scratch/.petri_chunk702_pod.json'))['endpoint'])")"
export CHUNK702_BASE_URL="$EP"
export CHUNK702_API_KEY="EMPTY"
LIMIT="${1:-}"                                # e.g. "--limit 2" for a smoke
LOG="output/petri_fabrication/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
echo ">>> target endpoint: $EP" | tee "$LOG/run.log"
cd scratch/petri_run
./.petrienv/bin/inspect eval inspect_petri/audit \
  -T seed_instructions="$REPO/scratch/petri_seeds_fabrication" \
  --model-role auditor=openrouter/anthropic/claude-sonnet-5 \
  --model-role judge=openrouter/google/gemini-3-flash-preview \
  --model-role target=openai-api/chunk702/chunk702 \
  --log-dir "$REPO/$LOG" \
  --max-connections 8 \
  $LIMIT 2>&1 | tee -a "$REPO/$LOG/run.log"
echo ">>> inspect exited ${PIPESTATUS[0]}; logs in $REPO/$LOG" | tee -a "$REPO/$LOG/run.log"
