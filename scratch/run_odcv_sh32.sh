#!/usr/bin/env bash
# ABOUTME: ODCV-lite, one pass at concurrency 32, against the stated-harm adapter.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=ddvgxwzsqvn1zw
ADDR=root@152.236.142.242:15942
TARGET=dougalldeepmind/2026-10-04-qwen36-0-ablation-stated-harm-15
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_odcv_statedharm_c32.log
exec > >(tee -a "$LOG") 2>&1

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done

# concurrency 32 as asked. The scenario containers run on THIS machine, which had 0GB of 15
# available at launch; at 32 the earlier run peaked at 63 containers and load 27 on 12 cores.
# If cells are OOM-killed they drop from the scored set, so the pass audit below is the check
# that matters -- a short pass means the concurrency, not the model.
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$TARGET" --server "$ADDR" --ssh-key ~/.ssh/id_ed25519 \
  --port 8002 --terminate-pod passes=1 concurrency=32 2>&1 | tail -35
echo ">>> evals rc=${PIPESTATUS[0]}"
echo ">>> surviving matboz pods:"; ./.venv/bin/runpod pods 2>&1 | grep -i matboz || echo "   none"
