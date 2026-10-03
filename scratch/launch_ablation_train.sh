#!/usr/bin/env bash
# ABOUTME: Wait for the matboz-ablation pod to report READY, then launch the ablation-10 SFT
# ABOUTME: run on it under nohup, and take the pod down once the adapter is safely on the Hub.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=xzrilpv9wq6dka
SSH="ssh -p 12365 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@213.181.104.55"
MIX=dougalldeepmind/2026-10-02-ablation-10-mix
ADAPTER=dougalldeepmind/2026-10-02-qwen36-0-ablation-10

for i in $(seq 1 80); do
  if curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY; then
    echo ">>> READY after $((i*30))s"; break
  fi
  [ "$i" = 80 ] && { echo "!!! never READY after 40min — NOT launching"; exit 1; }
  sleep 30
done

$SSH 'cd /root/work && mkdir -p output/logs && \
  LOG=output/logs/$(date +%Y%m%d_%H%M%S)_train_ablation_10.log && \
  echo "$LOG" > output/logs/.latest_ablation && \
  setsid nohup uv run train --config configs/train/sft.yaml model=qwen36 \
    data_repo='"$MIX"' seed=0 wandb=true > "$LOG" 2>&1 < /dev/null & \
  sleep 6; echo "launched: $(cat output/logs/.latest_ablation)"'

# The bracket keeps the pattern from matching the ssh shell that CARRIES it -- without it
# pgrep finds its own parent's command line and the trainer looks alive forever (cost ~1h20m
# of idle H200 on 2026-10-01).
for i in $(seq 1 120); do
  sleep 60
  if ! $SSH 'pgrep -f "[u]v run train" >/dev/null' 2>/dev/null; then
    echo ">>> trainer exited after ${i}m"; break
  fi
  [ "$i" = 120 ] && { echo "!!! still training after 2h — leaving the pod UP"; exit 1; }
done

echo "=== training log tail ==="
$SSH 'tr "\r" "\n" < "$(cat /root/work/output/logs/.latest_ablation)" | grep -viE "[0-9]+%\|" | tail -12'

# The artifact must be safe BEFORE the card goes away.
if ./.venv/bin/python - <<PY
import sys
from dotenv import dotenv_values
from huggingface_hub import HfApi
api = HfApi(token=dotenv_values(".env")["HF_TOKEN_MATBOZ"])
try:
    info = api.model_info("$ADAPTER")
except Exception as e:
    print(f"!!! adapter NOT on the Hub: {type(e).__name__}"); sys.exit(1)
files = {s.rfilename for s in info.siblings}
missing = {"adapter_model.safetensors", "adapter_config.json"} - files
print(f">>> adapter {info.sha[:12]} — {len(files)} files" + (f" MISSING {missing}" if missing else " (complete)"))
sys.exit(1 if missing else 0)
PY
then
  echo ">>> taking down $POD"
  ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
else
  echo "!!! adapter absent or incomplete — leaving $POD UP for inspection"
fi
