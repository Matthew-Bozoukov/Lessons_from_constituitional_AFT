#!/usr/bin/env bash
# ABOUTME: Three-pass ODCV then MASK against the stated-harm adapter, both served from one pod.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=l0nymtps5xgkmn
ADDR=root@152.236.142.242:16738
TARGET=dougalldeepmind/2026-10-04-qwen36-0-ablation-stated-harm-15
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_sh_odcv3_mask.log
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
echo ">>> budget before ODCV: $(bal)"

# Three passes in ONE run, so the lite protocol's CI comes from spread across passes as it is
# meant to, rather than from me pooling separate run directories by hand.
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$TARGET" --server "$ADDR" --ssh-key ~/.ssh/id_ed25519 \
  --port 8002 passes=3 concurrency=32 2>&1 | tail -30
echo ">>> odcv rc=${PIPESTATUS[0]} | budget now: $(bal)"

# MASK on the same pod: 1,000 rows, flash-judged, `mode: think` pinned by the config because a
# MASK ladder is only readable within one serving mode (~12 points between modes).
# --terminate-pod here, on the LAST eval, so the card goes away when everything is done.
./.venv/bin/evals --name mask --target "$TARGET" --server "$ADDR" \
  --ssh-key ~/.ssh/id_ed25519 --port 8003 --terminate-pod 2>&1 | tail -30
echo ">>> mask rc=${PIPESTATUS[0]} | budget now: $(bal)"
echo ">>> surviving matboz pods:"; ./.venv/bin/runpod pods 2>&1 | grep -i matboz || echo "   none"
