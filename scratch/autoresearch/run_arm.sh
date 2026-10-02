#!/bin/bash
# ABOUTME: One autoresearch arm end to end, unattended: budget check, rent a train pod, train on a pinned mix,
# ABOUTME: verify the adapter, tear down, then MASK and ODCV-lite on their own pods in parallel, pushed to the Hub.
# Usage: scratch/autoresearch/run_arm.sh <tag> <mix repo> <mix revision> <seed> <port base>
#   tag: short label for pod names and logs (pods are jamie-ar-{train,mask,odcv}-<tag>, which is what the
#   budget guard counts). Ports <base> and <base>+1 are the MASK and ODCV tunnels; give every arm its own pair.
# A failed boot or dead training is reported and the pod left for the budget watchdog (it saves the boot log and
# terminates pods older than 3 h); nothing here tears down a pod that has not finished its work.
set -u
TAG=$1; MIX=$2; REV=$3; SEED=$4; PORT=$5
REPO=/Users/jamie/Projects/lasr; cd "$REPO"
L=output/autoresearch/logs; mkdir -p $L
BUD="uv run python scratch/autoresearch/budget.py"
say() { echo "[$(date -u +%H:%M)] $TAG: $*"; }
SUBJ=${MIX#*/}; SUBJ=${SUBJ#20??-??-??-}; SUBJ=${SUBJ%-mix}
LOCAL=qwen36_${SEED}_$(echo "$SUBJ" | tr '-' '_')
TODAY=$(date -u +%F)
BRANCH=$(git branch --show-current)

find_adapter() { uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from huggingface_hub import HfApi
a=HfApi()
for m in a.list_models(author='dougalldeepmind', search='qwen36-$SEED-$SUBJ'):
    if m.id.endswith('-qwen36-$SEED-$SUBJ') and m.id.split('/')[1][:10] >= '$TODAY':
        f=[s.rfilename for s in a.model_info(m.id).siblings]
        if 'adapter_model.safetensors' in f and 'training_meta.json' in f: print(m.id)
" 2>/dev/null | tail -1; }

ADAPTER=${ADAPTER:-$(find_adapter)}
if [ -z "$ADAPTER" ]; then
  $BUD check --reserve 20 || { say "budget refuses this arm"; exit 9; }
  RL=$L/rent_train_$TAG.log
  # TRAIN_POD + TRAIN_HOSTPORT adopt a pod that is already up and READY (e.g. one whose `runpod up` timed out
  # waiting for a slow boot that later finished): no rent, straight to training.
  for i in $([ -n "${TRAIN_POD:-}" ] && echo "" || seq 1 40); do
    if uv run runpod pods 2>&1 | grep -q " jamie-ar-train-$TAG "; then say "a pod named jamie-ar-train-$TAG already exists -- stopping"; exit 2; fi
    say "train rent attempt $i: uv run runpod up jamie-ar-train-$TAG --train configs/train/sft.yaml --model qwen36 --count 1 --push_env --branch $BRANCH"
    if uv run runpod up jamie-ar-train-$TAG --train configs/train/sft.yaml --model qwen36 --count 1 --push_env --branch "$BRANCH" > "$RL" 2>&1; then break; fi
    if grep -q "BILLING NOW" "$RL"; then say "pod created but boot failed -- left up for the watchdog:"; grep -v "still bootstrapping" "$RL" | tail -8; exit 3; fi
    say "$(grep -o 'HTTPError.*\|RuntimeError.*\|AssertionError.*' "$RL" | tail -1 | cut -c1-140)"
    $BUD check --reserve 20 >/dev/null || { say "budget refuses"; exit 9; }
    sleep 150
  done
  if [ -n "${TRAIN_POD:-}" ]; then POD=$TRAIN_POD; HOSTPORT=$TRAIN_HOSTPORT; say "adopting ready pod $POD"
  else
    grep -q "BILLING NOW" "$RL" || { say "gave up renting a train pod"; exit 4; }
    POD=$(grep -o "pod [a-z0-9]* — BILLING NOW" "$RL" | head -1 | awk '{print $2}')
    HOSTPORT=$(grep -o "root@[0-9.]*:[0-9]*" "$RL" | head -1 | sed 's/root@//')
  fi
  IP=${HOSTPORT%:*}; SPORT=${HOSTPORT#*:}
  SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $SPORT root@$IP"
  say "pod $POD ($IP:$SPORT) ready; launching: uv run train --config configs/train/sft.yaml model=qwen36 data_repo=$MIX data_revision=$REV seed=$SEED"
  $SSH "cd /root/work && setsid nohup uv run train --config configs/train/sft.yaml model=qwen36 data_repo=$MIX data_revision=$REV seed=$SEED > /root/work/train.log 2>&1 < /dev/null & disown" &
  sleep 420
  ST=$($SSH 'pgrep -f "uv run trai[n]" >/dev/null && echo ALIVE || echo DEAD; tail -c 400 /root/work/train.log | tr "\r" "\n" | grep -o "[0-9]*/[0-9]* \[[^]]*\]" | tail -1' 2>/dev/null | tr '\n' ' ')
  say "7 min after launch: $ST"
  case "$ST" in DEAD*) say "training died (pod left up):"; $SSH 'tail -8 /root/work/train.log'; exit 5;; esac
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
else
  say "adapter already on the Hub: $ADAPTER"
fi

run_eval() {  # <eval> <port>
  local EV=$1 P=$2 RL=$L/rent_${EV}_$TAG.log POD="" SERVER="" GPU=()
  [ "$EV" = odcv ] && GPU=(--gpu "NVIDIA H200")
  $BUD check --reserve 7 || { say "$EV: budget refuses"; return 9; }
  for i in $(seq 1 30); do
    uv run runpod up jamie-ar-$EV-$TAG --eval "$EV" --target "$ADAPTER" ${GPU[@]+"${GPU[@]}"} > "$RL" 2>&1
    POD=$(grep -o "pod [a-z0-9]* — BILLING NOW" "$RL" | head -1 | awk '{print $2}')
    SERVER=$(grep -o "\-\-server root@[0-9.]*:[0-9]*" "$RL" | head -1 | awk '{print $2}')
    [ -n "$SERVER" ] && break
    say "$EV rent attempt $i failed: $(grep -o 'HTTPError.*\|RuntimeError.*' "$RL" | tail -1 | cut -c1-100)"
    [ -n "$POD" ] && { say "$EV: pod $POD has no address -- left for the watchdog"; return 3; }
    sleep 120
  done
  [ -n "$SERVER" ] || { say "$EV: gave up renting"; return 4; }
  for i in $(seq 1 90); do curl -s --max-time 15 "https://$POD-8080.proxy.runpod.net/boot.log" | grep -q "^READY" && break; sleep 30; done
  curl -s --max-time 15 "https://$POD-8080.proxy.runpod.net/boot.log" | grep -q "^READY" || { say "$EV: boot never READY -- $POD left for the watchdog"; return 5; }
  if lsof -nP -iTCP:$P -sTCP:LISTEN >/dev/null 2>&1; then say "$EV: local port $P busy -- refusing"; return 6; fi
  say "running: uv run evals --name $EV --target $ADAPTER --server $SERVER --port $P --terminate-pod"
  uv run evals --name "$EV" --target "$ADAPTER" --server "$SERVER" --port "$P" --terminate-pod > "$L/eval_${EV}_$TAG.log" 2>&1
  local RC=$?
  say "$EV evals exited $RC; $(grep -ao 'pushed https://huggingface.co/datasets/[^ ]*' "$L/eval_${EV}_$TAG.log" | tail -1)"
  return $RC
}
run_eval mask "$PORT" & M=$!
sleep 20
run_eval odcv "$((PORT + 1))" & O=$!
wait $M; MR=$?; wait $O; OR=$?
say "ARM DONE adapter=$ADAPTER mask_rc=$MR odcv_rc=$OR | $($BUD status 2>/dev/null | tail -1)"
