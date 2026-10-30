#!/bin/bash
# ABOUTME: Take over an arm whose training is ALREADY RUNNING on a pod: wait for its adapter on the Hub, verify the
# ABOUTME: sha256 against the pod's copy, save the train log, tear the pod down, then hand over to run_arm.sh for evals.
# Usage: scratch/autoresearch/finish_train.sh <tag> <pod id> <ip:port> <mix repo> <mix revision> <seed> <port base>
# Exists because editing run_arm.sh while instances of it were running broke those instances (bash reads a script
# incrementally): never edit a running script; add a new file instead.
set -u
TAG=$1; POD=$2; HOSTPORT=$3; MIX=$4; REV=$5; SEED=$6; PORT=$7
cd /Users/jamie/Projects/lasr
IP=${HOSTPORT%:*}; SPORT=${HOSTPORT#*:}
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $SPORT root@$IP"
SUBJ=${MIX#*/}; SUBJ=${SUBJ#20??-??-??-}; SUBJ=${SUBJ%-mix}
LOCAL=qwen36_${SEED}_$(echo "$SUBJ" | tr '-' '_'); TODAY=$(date -u +%F)
say() { echo "[$(date -u +%H:%M)] $TAG: $*"; }
find_adapter() { uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from huggingface_hub import HfApi
a=HfApi()
for m in a.list_models(author='dougalldeepmind', search='qwen36-$SEED-$SUBJ'):
    if m.id.endswith('-qwen36-$SEED-$SUBJ') and m.id.split('/')[1][:10] >= '$TODAY':
        f=[s.rfilename for s in a.model_info(m.id).siblings]
        if 'adapter_model.safetensors' in f and 'training_meta.json' in f: print(m.id)
" 2>/dev/null | tail -1; }
say "taking over the wait for training on $POD ($HOSTPORT)"
ADAPTER=""
for i in $(seq 1 150); do
  ADAPTER=$(find_adapter); [ -n "$ADAPTER" ] && break
  if [ "$($SSH 'pgrep -f "uv run trai[n]" >/dev/null && echo ALIVE || echo DEAD' 2>/dev/null)" = "DEAD" ]; then
    sleep 90; ADAPTER=$(find_adapter); [ -n "$ADAPTER" ] && break
    say "training exited without an adapter (pod left up):"; $SSH 'tail -8 /root/work/train.log'; exit 6; fi
  sleep 60
done
[ -n "$ADAPTER" ] || { say "no adapter after 150 min (pod left up)"; exit 7; }
HUB=$(uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from huggingface_hub import HfApi
i=HfApi().model_info('$ADAPTER', files_metadata=True); print(next(s.lfs.sha256 for s in i.siblings if s.rfilename=='adapter_model.safetensors'))" 2>/dev/null)
PODSHA=$($SSH "sha256sum /root/work/output/train/*_${LOCAL}/adapter/adapter_model.safetensors" 2>/dev/null | awk '{print $1}' | head -1)
mkdir -p output/train_logs; scp -o ConnectTimeout=20 -P "$SPORT" root@"$IP":/root/work/train.log output/train_logs/${ADAPTER#*/}.train.log >/dev/null 2>&1
if [ -n "$HUB" ] && [ "$HUB" = "$PODSHA" ]; then
  say "adapter $ADAPTER verified (sha256 ${HUB:0:12}); tearing down $POD"; uv run runpod down --pod "$POD" 2>&1 | tail -1
else say "adapter sha mismatch (hub ${HUB:0:12} vs pod ${PODSHA:0:12}) -- pod left up"; exit 8; fi
exec bash scratch/autoresearch/run_arm.sh "$TAG" "$MIX" "$REV" "$SEED" "$PORT"
