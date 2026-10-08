#!/usr/bin/env bash
# ABOUTME: Wait for the matboz-petri-da15-1002 pod, run the 90-seed constitution audit against
# ABOUTME: the 2026-10-02 da-15 adapter on the nosynth settings, then take the pod down.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=u8wcixmtpmyk1s
SERVER=root@152.236.142.242:19697
TARGET=dougalldeepmind/2026-10-02-qwen36-0-da-15
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_petri_da15_1002.log

for i in $(seq 1 80); do
  if curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY; then
    echo ">>> READY after $((i*30))s"; break
  fi
  [ "$i" = 80 ] && { echo "!!! never READY after 40min; taking the pod down"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done

# Defaults are deliberately not overridden: auditor claude-sonnet-5, judge
# gemini-3-flash-preview, max_turns 25, connections 8, the full 90-seed set, 32k context --
# the same settings the 2026-09-22-nosynth audit ran on, so the two are comparable.
echo ">>> auditing $TARGET (90 seeds, sonnet-5 auditor, flash judge, 25 turns)" | tee "$LOG"
./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server "$SERVER" --target "$TARGET" 2>&1 | tee -a "$LOG" | tail -30
echo ">>> audit finished rc=${PIPESTATUS[0]}" | tee -a "$LOG"

# The driver deliberately does not terminate the pod; teardown is the caller's, and this is
# the caller. An audit that died part-way leaves its Inspect log dir behind either way, so
# nothing is lost by releasing the card.
echo ">>> taking down $POD" | tee -a "$LOG"
./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2 | tee -a "$LOG"
