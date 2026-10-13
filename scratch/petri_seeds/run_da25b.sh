#!/usr/bin/env bash
# ABOUTME: Resume the da-25 constitution audit (7 of 90 scored before its tunnel died), under the
# ABOUTME: liveness watchdog that aborts instead of paying the auditor to talk to a dead endpoint.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=3g30jhzqcws6sg
HOST=root@212.247.220.136:11397
RETRY=output/petri/2026-10-03_155634_2026-10-02-qwen36-0-da-25/logs/2026-10-03T15-00-44-00-00_audit_52wq8vi4ngVMbfrUaBwZnR.eval
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_petri_da25_resume.log
exec > >(tee -a "$LOG") 2>&1

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done

./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server "$HOST" --port 8001 \
  --target dougalldeepmind/2026-10-02-qwen36-0-da-25 \
  --retry "$RETRY" 2>&1 | tail -25
echo ">>> da-25 resume rc=${PIPESTATUS[0]}"

echo ">>> taking down $POD"
./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
