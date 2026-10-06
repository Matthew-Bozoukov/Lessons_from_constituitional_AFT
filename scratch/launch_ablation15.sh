#!/usr/bin/env bash
# ABOUTME: Train the ablation-15 arm on 2xH200 (DDP), then run ODCV-lite at ONE pass against the
# ABOUTME: resulting adapter served from the same pod, and release the card at the end.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=lrc6w4mzlwuw5h
HOST=root@103.196.86.23:15072
SSH="ssh -p 15072 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@103.196.86.23"
MIX=dougalldeepmind/2026-10-04-ablation-15-mix
ADAPTER=dougalldeepmind/2026-10-04-qwen36-0-ablation-15
RLOG=/root/work/output/logs/train_ablation_15.log
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_ablation15_train_then_odcv.log
exec > >(tee -a "$LOG") 2>&1

down() { echo ">>> taking down $POD"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2; }

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; down; exit 1; }
  sleep 30
done

# --- 1. train, 2 GPUs via torchrun -----------------------------------------------------
$SSH "mkdir -p /root/work/output/logs && cd /root/work && \
  setsid nohup uv run torchrun --nproc_per_node=2 scripts/train/train_lora.py \
    --config configs/train/sft.yaml model=qwen36 data_repo=$MIX seed=0 wandb=true \
    > $RLOG 2>&1 < /dev/null & \
  sleep 8; echo 'launched'; tail -3 $RLOG 2>/dev/null"

# Bracketed so the pattern cannot match the ssh shell carrying it. `|| true` on the ssh so a
# transient connection failure is not read as "the trainer exited" (that false negative ended a
# watch early on 2026-10-03).
for i in $(seq 1 180); do
  sleep 60
  if $SSH 'pgrep -f "[t]orchrun" >/dev/null; echo "ALIVE:$?"' 2>/dev/null | grep -q "ALIVE:1"; then
    echo ">>> trainer exited after ${i}m"; break
  fi
  [ "$i" = 180 ] && { echo "!!! still training after 3h — leaving $POD UP"; exit 1; }
done
echo "=== training log tail ==="
$SSH "tr '\r' '\n' < $RLOG | grep -viE '[0-9]+%\|' | tail -14"

# --- 2. the adapter must be on the Hub before the card is reused or released -------------
if ! ./.venv/bin/python - <<PY
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
then echo "!!! adapter absent or incomplete — leaving $POD UP for inspection"; exit 1; fi

# --- 3. ODCV, ONE pass, serving from this same pod ---------------------------------------
# passes=1 as asked. concurrency 16 rather than the config's 32 because the scenario CONTAINERS
# run on THIS machine, which has been sitting near 2GB free; at 32 the failure is OOM-killed
# cells dropping out of the scored set, not slowness.
echo ">>> ODCV-lite, 1 pass, against $ADAPTER"
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$ADAPTER" --server "$HOST" --ssh-key ~/.ssh/id_ed25519 \
  --port 8002 passes=1 concurrency=16 2>&1 | tail -40
echo ">>> evals rc=${PIPESTATUS[0]}"

down
