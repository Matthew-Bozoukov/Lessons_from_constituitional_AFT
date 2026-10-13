#!/usr/bin/env bash
# ABOUTME: The 90-seed constitution audit on the delib-sonnet 15% arm, on the same settings every
# ABOUTME: other arm was audited with, then release the pod.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=tkclz92mmdq4z9
ADDR=root@152.236.142.242:15414
TARGET=dougalldeepmind/2026-10-04-qwen36-0-delib-sonnet-15
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_petri_delib_sonnet.log
exec > >(tee -a "$LOG") 2>&1

bal () { ./.venv/bin/python -c "
from dotenv import dotenv_values
import json,urllib.request
k=dotenv_values('.env')['OPENROUTER_API_KEY']
d=json.load(urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+k})))['data']
print(f\"\$ {d['total_credits']-d['total_usage']:.2f}\")"; }

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done
echo ">>> budget before: $(bal)"

# Every default untouched -- auditor claude-sonnet-5, judge gemini-3-flash-preview, max_turns 25,
# 8 in flight, the full 90-seed set, 32k context -- so this is comparable to nosynth, da-5,
# da-15 and da-25. The driver's liveness watchdog aborts if the target stops answering, which is
# what a dead tunnel cost on 2026-10-03 (~$52 of auditor spend for 7 scored samples).
./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server "$ADDR" --port 8004 --target "$TARGET" 2>&1 | tail -25
echo ">>> audit rc=${PIPESTATUS[0]} | budget now: $(bal)"

echo ">>> taking down $POD"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
echo ">>> surviving matboz pods:"; ./.venv/bin/runpod pods 2>&1 | grep -i matboz || echo "   none"
