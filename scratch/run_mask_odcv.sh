#!/bin/bash
# ABOUTME: MASK and ODCV-lite for one trained adapter, each on its own pod and local port.
# ABOUTME: scratch/autoresearch/run_evals.sh without the overnight budget guard; pods are named jamie-<eval>-<tag>.
# Usage: scratch/run_mask_odcv.sh <tag> <adapter repo> <port base> [--after-arm-done]
#   --after-arm-done: first wait for "ARM DONE" in output/autoresearch/logs/arm_<tag>.log (the train pipeline's last
#   line, printed after the adapter is verified and the train pod torn down).
set -u
TAG=$1; ADAPTER=$2; PORT=$3; WAIT=${4:-}
cd /Users/jamie/Projects/lasr
L=output/autoresearch/logs; mkdir -p $L
say() { echo "[$(date -u +%H:%M)] $TAG: $*"; }
if [ "$WAIT" = "--after-arm-done" ]; then until grep -q "ARM DONE" "$L/arm_$TAG.log" 2>/dev/null; do sleep 30; done; say "train pipeline finished; starting evals"; fi

run_eval() {  # <eval> <port>
  local EV=$1
  local P=$2
  local RL=$L/rent_${EV}_$TAG.log
  local POD="" SERVER="" RC=0
  local GPU=()
  [ "$EV" = odcv ] && GPU=(--gpu "NVIDIA H200")
  for i in $(seq 1 30); do
    uv run runpod up jamie-$EV-$TAG --eval "$EV" --target "$ADAPTER" ${GPU[@]+"${GPU[@]}"} > "$RL" 2>&1
    POD=$(grep -o "pod [a-z0-9]* — BILLING NOW" "$RL" | head -1 | awk '{print $2}')
    SERVER=$(grep -o "\-\-server root@[0-9.]*:[0-9]*" "$RL" | head -1 | awk '{print $2}')
    [ -n "$SERVER" ] && break
    say "$EV rent attempt $i failed: $(grep -o 'HTTPError.*\|RuntimeError.*' "$RL" | tail -1 | cut -c1-100)"
    [ -n "$POD" ] && { say "$EV: pod $POD has no address -- left up for diagnosis"; return 3; }
    sleep 120
  done
  [ -n "$SERVER" ] || { say "$EV: gave up renting"; return 4; }
  say "$EV: pod $POD at $SERVER; waiting for boot READY"
  for i in $(seq 1 90); do curl -s --max-time 15 "https://$POD-8080.proxy.runpod.net/boot.log" | grep -q "^READY" && break; sleep 30; done
  curl -s --max-time 15 "https://$POD-8080.proxy.runpod.net/boot.log" | grep -q "^READY" || { say "$EV: boot never READY -- $POD left up for diagnosis"; return 5; }
  if lsof -nP -iTCP:$P -sTCP:LISTEN >/dev/null 2>&1; then say "$EV: local port $P busy -- refusing"; return 6; fi
  say "running: uv run evals --name $EV --target $ADAPTER --server $SERVER --port $P --terminate-pod"
  uv run evals --name "$EV" --target "$ADAPTER" --server "$SERVER" --port "$P" --terminate-pod > "$L/eval_${EV}_$TAG.log" 2>&1
  RC=$?
  say "$EV evals exited $RC; $(grep -ao 'pushed https://huggingface.co/datasets/[^ ]*' "$L/eval_${EV}_$TAG.log" | tail -1)"
  return $RC
}
run_eval mask "$PORT" & M=$!
sleep 20
run_eval odcv "$((PORT + 1))" & O=$!
wait $M; MR=$?; wait $O; OR=$?
say "EVALS DONE adapter=$ADAPTER mask_rc=$MR odcv_rc=$OR"
