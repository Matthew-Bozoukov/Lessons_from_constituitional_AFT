#!/usr/bin/env bash
# ABOUTME: After the da-t6-note SFT run finishes and its adapter is on the Hub, take the idle
# ABOUTME: train pod down, serve the adapter on an eval pod and drive ODCV-lite (3 passes) here.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY

TRAIN_POD=obpykf45ttoso1
TRAIN_SSH="ssh -p 14491 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o ConnectTimeout=20 root@157.157.221.201"
LOG=/root/work/output/logs/20261001_154717_train_da_t6_note_15.log
ADAPTER=dougalldeepmind/2026-10-01-qwen36-0-da-t6-note-15

# ---- 1. wait for the trainer to exit -----------------------------------------
for i in $(seq 1 90); do
  # The bracket keeps the pattern from matching the ssh shell that CARRIES it: without it
  # pgrep finds its own parent's command line and the trainer looks alive forever.
  if ! $TRAIN_SSH 'pgrep -f "[u]v run train" >/dev/null' 2>/dev/null; then
    echo ">>> trainer exited after $((i))m"; break
  fi
  [ "$i" = 90 ] && { echo "!!! trainer still running after 90m — stopping, no eval started"; exit 1; }
  sleep 60
done
echo "=== tail of the training log ==="
$TRAIN_SSH "tr '\r' '\n' < $LOG | grep -viE '[0-9]+%\|' | tail -15"

# ---- 2. the adapter must exist on the Hub BEFORE anything is torn down --------
if ! ./.venv/bin/python - <<'PY'
import sys
from dotenv import dotenv_values
from huggingface_hub import HfApi
env = dotenv_values(".env")
api = HfApi(token=env["HF_TOKEN_MATBOZ"])
try:
    info = api.model_info("dougalldeepmind/2026-10-01-qwen36-0-da-t6-note-15")
except Exception as e:
    print(f"!!! adapter NOT on the Hub: {type(e).__name__}"); sys.exit(1)
files = {s.rfilename for s in info.siblings}
need = {"adapter_model.safetensors", "adapter_config.json"}
missing = need - files
print(f">>> adapter {info.sha[:12]} — {len(files)} files" + (f" MISSING {missing}" if missing else " (complete)"))
sys.exit(1 if missing else 0)
PY
then echo "!!! adapter absent or incomplete — leaving the train pod UP for inspection, no eval"; exit 1; fi

# ---- 3. the train pod has done its job; it is ours and it is idle -------------
echo ">>> taking down the idle train pod $TRAIN_POD"
./.venv/bin/runpod down --pod "$TRAIN_POD" 2>&1 | tail -3

# ---- 4. serve the adapter on this eval's own inference card -------------------
echo ">>> provisioning the odcv eval pod"
UP=$(./.venv/bin/runpod up matboz-t6note-odcv --eval odcv --target "$ADAPTER" --push_env 2>&1)
echo "$UP" | tail -14
HOST=$(echo "$UP" | grep -oE "host: +root@[0-9.]+:[0-9]+" | head -1 | sed 's/host: *//')
[ -z "$HOST" ] && { echo "!!! could not parse the eval pod host — NOT starting odcv; check for a stray pod"; exit 1; }
echo ">>> eval pod at $HOST"

# ---- 5. drive ODCV-lite here; --terminate-pod owns the eval pod's cleanup -----
echo ">>> starting odcv-lite (passes: 3) against $ADAPTER"
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$ADAPTER" --server "$HOST" --ssh-key ~/.ssh/id_ed25519 \
  --terminate-pod 2>&1 | tail -40
