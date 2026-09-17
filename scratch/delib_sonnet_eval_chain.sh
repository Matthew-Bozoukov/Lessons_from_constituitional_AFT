#!/usr/bin/env bash
# ABOUTME: Waits for the delib-sonnet-7 adapter to land on the Hub, tears down the training pod, then rents one
# ABOUTME: eval pod per eval and runs MASK (batched judge) and ODCV lite in parallel, each tearing its pod down.
set -uo pipefail
cd /Users/jamie/Projects/lasr
export _ZO_DOCTOR=0
set -a; source .env; set +a
TRAIN_POD=7ih27tiu6tawod
LOG=output/eval_chain/delib_sonnet_7.log
say() { echo ">>> $(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

find_adapter() {
  uv run python - <<'PY' 2>/dev/null | grep -v zoxide | tail -1
from src.infra.huggingface import hf_api, hf_org
api = hf_api()
for m in api.list_models(author=hf_org(), search="delib-sonnet-7"):
    if m.id.endswith("-qwen36-0-delib-sonnet-7"):
        files = {s.rfilename for s in api.model_info(m.id).siblings}
        if "training_meta.json" in files and "adapter_model.safetensors" in files:
            print(m.id)
PY
}

say "waiting for the delib-sonnet-7 adapter on the Hub"
ADAPTER=""
while [ -z "$ADAPTER" ]; do
  sleep 300
  ADAPTER=$(find_adapter || true)
done
say "adapter: $ADAPTER"
sleep 120   # let the training process finish its epilogue before the pod goes
say "tearing down the training pod $TRAIN_POD"
uv run runpod down --pod "$TRAIN_POD" 2>&1 | grep -v zoxide | tee -a "$LOG"

say "renting one eval pod per eval"
uv run runpod up --name jamie-mask-delib-sonnet-7 --eval "$ADAPTER" --push_env --max_hours 6 > output/eval_chain/mask_pod.log 2>&1 &
P1=$!
uv run runpod up --name jamie-odcv-delib-sonnet-7 --eval "$ADAPTER" --push_env --max_hours 6 > output/eval_chain/odcv_pod.log 2>&1 &
P2=$!
wait $P1 $P2
MASK_HOST=$(grep -o "root@[0-9.]*:[0-9]*" output/eval_chain/mask_pod.log | head -1)
ODCV_HOST=$(grep -o "root@[0-9.]*:[0-9]*" output/eval_chain/odcv_pod.log | head -1)
say "mask pod: $MASK_HOST | odcv pod: $ODCV_HOST"
if [ -z "$MASK_HOST" ] || [ -z "$ODCV_HOST" ]; then say "POD RENTAL FAILED; see output/eval_chain/*_pod.log"; exit 1; fi

say "launching MASK (batched judge) and ODCV lite"
uv run evals --name mask --target "$ADAPTER" --server "$MASK_HOST" --ssh-key ~/.ssh/id_ed25519 --port 8021 --terminate-pod gen_concurrency=32 empty_content=reasoning judge_batch=true > output/eval_chain/mask.log 2>&1 &
M=$!
uv run evals --name odcv --config configs/eval/odcv/lite.yaml --target "$ADAPTER" --server "$ODCV_HOST" --ssh-key ~/.ssh/id_ed25519 --port 8022 --terminate-pod > output/eval_chain/odcv.log 2>&1 &
O=$!
wait $M; say "MASK exited $? : $(grep -o 'pushed https://[^ ]*' output/eval_chain/mask.log | tail -1)"
wait $O; say "ODCV exited $? : $(grep -o 'pushed https://[^ ]*' output/eval_chain/odcv.log | tail -1)"
uv run runpod pods 2>&1 | grep -v zoxide | tee -a "$LOG"
say "chain complete"
