#!/bin/bash
# ABOUTME: One arm, unattended: rent a 1-GPU train pod, train on a pinned mix, verify the pushed adapter against
# ABOUTME: the pod's copy, tear the pod down, then hand over to scratch/run_mask_odcv.sh for MASK and ODCV-lite.
# Usage: scratch/train_then_evals.sh <tag> <mix repo> <mix revision> <seed> <eval port base>
# A pod is torn down ONLY after its adapter is verified; on any failure it is left up and the reason is printed.
set -u
TAG=$1; MIX=$2; REV=$3; SEED=$4; PORT=$5
cd /Users/jamie/Projects/lasr
L=output/autoresearch/logs; mkdir -p $L
BRANCH=$(git branch --show-current)
SUBJ=${MIX#*/}; SUBJ=${SUBJ#20??-??-??-}; SUBJ=${SUBJ%-mix}
LOCAL=qwen36_${SEED}_$(echo "$SUBJ" | tr '-' '_'); TODAY=$(date -u +%F)
say() { echo "[$(date -u +%H:%M)] $TAG: $*"; }
RL=$L/rent_train_$TAG.log
say "renting: uv run runpod up --name jamie-train-$TAG --train configs/train/sft.yaml --model qwen36 --count 1 --push_env --branch $BRANCH"
uv run runpod up --name jamie-train-$TAG --train configs/train/sft.yaml --model qwen36 --count 1 --push_env --branch "$BRANCH" > "$RL" 2>&1
POD=$(grep -o "runpod down --pod [a-z0-9]*" "$RL" | tail -1 | awk '{print $4}')
SPORT=$(grep -o "ssh -p [0-9]* root@[0-9.]*" "$RL" | tail -1 | awk '{print $3}')
IP=$(grep -o "ssh -p [0-9]* root@[0-9.]*" "$RL" | tail -1 | awk '{print $4}'); IP=${IP#root@}
grep -q "The boot has finished (READY" "$RL" && [ -n "$SPORT" ] && [ -n "$IP" ] || { say "train pod not READY (pod '${POD:-?}' left as it is); see $RL"; tail -4 "$RL" | cut -c1-200; exit 3; }
say "train pod $POD READY at $IP:$SPORT"
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$SPORT" root@"$IP" "cd /root/work && setsid nohup uv run train --config configs/train/sft.yaml model=qwen36 data_repo=$MIX data_revision=$REV seed=$SEED > /root/work/train.log 2>&1 < /dev/null & disown" &
say "running on the pod: uv run train --config configs/train/sft.yaml model=qwen36 data_repo=$MIX data_revision=$REV seed=$SEED"
sleep 120
find_adapter() { uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from huggingface_hub import HfApi
a=HfApi()
for m in a.list_models(author='dougalldeepmind', search='qwen36-$SEED-$SUBJ'):
    if m.id.endswith('-qwen36-$SEED-$SUBJ') and m.id.split('/')[1][:10] >= '$TODAY':
        f=[s.rfilename for s in a.model_info(m.id).siblings]
        if 'adapter_model.safetensors' in f and 'training_meta.json' in f: print(m.id)
" 2>/dev/null | tail -1; }
pod_ssh() { ssh -o ConnectTimeout=20 -p "$SPORT" root@"$IP" "$@" 2>/dev/null; }
ADAPTER=""
for i in $(seq 1 180); do
  ADAPTER=$(find_adapter); [ -n "$ADAPTER" ] && break
  if [ "$(pod_ssh 'pgrep -f "train_lor[a]|bin/trai[n] " >/dev/null && echo ALIVE || echo DEAD')" = "DEAD" ]; then
    sleep 120; ADAPTER=$(find_adapter); [ -n "$ADAPTER" ] && break
    say "training exited without an adapter (pod $POD left up):"; pod_ssh 'grep -v -i warn /root/work/train.log | tail -8' | cut -c1-220; exit 6; fi
  sleep 60
done
[ -n "$ADAPTER" ] || { say "no adapter after 180 min (pod $POD left up)"; exit 7; }
HUB=$(uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from huggingface_hub import HfApi
i=HfApi().model_info('$ADAPTER', files_metadata=True); print(next(s.lfs.sha256 for s in i.siblings if s.rfilename=='adapter_model.safetensors'))" 2>/dev/null | tail -1)
PODSHA=$(pod_ssh "sha256sum /root/work/output/train/*_${LOCAL}/adapter/adapter_model.safetensors" | awk '{print $1}' | head -1)
mkdir -p output/train_logs; scp -o ConnectTimeout=20 -P "$SPORT" root@"$IP":/root/work/train.log output/train_logs/${ADAPTER#*/}.train.log >/dev/null 2>&1
if [ -n "$HUB" ] && [ "$HUB" = "$PODSHA" ]; then
  say "adapter $ADAPTER verified (sha256 ${HUB:0:12}); tearing down $POD"; uv run runpod down --pod "$POD" 2>&1 | tail -1
else say "adapter sha mismatch (hub ${HUB:0:12} vs pod ${PODSHA:0:12}) -- pod $POD left up"; exit 8; fi
say "TRAIN DONE adapter=$ADAPTER; starting MASK (port $PORT) and ODCV-lite (port $((PORT + 1)))"
exec bash scratch/run_mask_odcv.sh "$TAG" "$ADAPTER" "$PORT"
