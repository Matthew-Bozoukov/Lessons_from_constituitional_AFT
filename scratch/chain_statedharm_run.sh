#!/usr/bin/env bash
# ABOUTME: Train the stated-harm 15% arm on 2xH200, then ODCV-lite at one pass on an --eval pod.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
POD=fgdgc91jnwtwje
SSH="ssh -p 17483 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@212.247.220.136"
MIX=dougalldeepmind/2026-10-04-ablation-stated-harm-15-mix
ADAPTER=dougalldeepmind/2026-10-04-qwen36-0-ablation-stated-harm-15
RLOG=/root/work/output/logs/train_stated_harm.log
LOG=output/logs/$(date +%Y%m%d_%H%M%S)_statedharm_train_odcv.log
exec > >(tee -a "$LOG") 2>&1

for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD"; exit 1; }
  sleep 30
done

$SSH "mkdir -p /root/work/output/logs && cd /root/work && \
  setsid nohup uv run torchrun --nproc_per_node=2 scripts/train/train_lora.py \
    --config configs/train/sft.yaml model=qwen36 data_repo=$MIX seed=0 wandb=true \
    > $RLOG 2>&1 < /dev/null & sleep 8; echo launched"

# A SUCCESSFUL ssh reporting no torchrun ends the wait; a failed ssh does not, because a dropped
# connection read as "finished" ended a watch early on 2026-10-03.
for i in $(seq 1 180); do
  sleep 60
  if $SSH 'pgrep -f "[t]orchrun" >/dev/null; echo "ALIVE:$?"' 2>/dev/null | grep -q "ALIVE:1"; then
    echo ">>> trainer exited after ${i}m"; break
  fi
  [ "$i" = 180 ] && { echo "!!! still training after 3h — leaving $POD UP"; exit 1; }
done
$SSH "tr '\r' '\n' < $RLOG | grep -viE '[0-9]+%\|' | tail -12"

./.venv/bin/python - <<PY || { echo "!!! adapter incomplete — leaving $POD UP"; exit 1; }
import sys
from dotenv import dotenv_values
from huggingface_hub import HfApi
api = HfApi(token=dotenv_values(".env")["HF_TOKEN_MATBOZ"])
try: info = api.model_info("$ADAPTER")
except Exception as e: print(f"!!! adapter NOT on the Hub: {type(e).__name__}"); sys.exit(1)
f = {s.rfilename for s in info.siblings}
m = {"adapter_model.safetensors", "adapter_config.json"} - f
print(f">>> adapter {info.sha[:12]} — {len(f)} files" + (f" MISSING {m}" if m else " (complete)"))
sys.exit(1 if m else 0)
PY
echo ">>> taking down the train pod $POD"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2

# ODCV needs an --eval pod: a --train pod has no vLLM and the preflight refuses it. --eval mask
# for the H200 card, because two H100s failed to publish SSH within 420s today and billed until
# terminated.
UP2=$(./.venv/bin/runpod up matboz-statedharm-odcv --eval mask --target "$ADAPTER" --push_env 2>&1)
echo "$UP2" | grep -E "^pod:|^host:|NVIDIA|BILLING|no SSH"
POD2=$(echo "$UP2" | grep -oE "^pod: +[a-z0-9]+" | awk '{print $2}')
ADDR2=$(echo "$UP2" | grep -oE "host: +root@[0-9.]+:[0-9]+" | sed 's/host: *//')
[ -z "$POD2" ] && { echo "!!! no eval pod parsed — check for a stray pod"; exit 1; }
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD2}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> eval pod READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! eval pod never READY"; ./.venv/bin/runpod down --pod "$POD2"; exit 1; }
  sleep 30
done
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$ADAPTER" --server "$ADDR2" --ssh-key ~/.ssh/id_ed25519 \
  --port 8002 --terminate-pod passes=1 concurrency=16 2>&1 | tail -35
echo ">>> evals rc=${PIPESTATUS[0]}"
echo ">>> surviving matboz pods:"; ./.venv/bin/runpod pods 2>&1 | grep -i matboz || echo "   none"
