#!/usr/bin/env bash
# ABOUTME: Run the 90-seed constitution audit on the 2026-10-02 da-5 adapter, on its own port
# ABOUTME: so it cannot collide with a tunnel another audit has not finished releasing.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=b7rxibggujk7pr
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_petri_da5.log
exec > >(tee -a "$LOG") 2>&1
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done
# --port 8001: the previous attempt died because the da-15 retry's tunnel still held 8000
# when this started. A second audit gets its own port rather than waiting on the first's
# teardown.
./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server root@152.236.142.241:15665 --port 8001 \
  --target dougalldeepmind/2026-10-02-qwen36-0-da-5 2>&1 | tail -25
echo ">>> da-5 audit rc=${PIPESTATUS[0]}"
echo ">>> taking down $POD"
./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
