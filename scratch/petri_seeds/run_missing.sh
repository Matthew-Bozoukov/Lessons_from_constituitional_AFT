#!/usr/bin/env bash
# ABOUTME: Rent a pod, rerun only the constitution seeds that have no valid score in an arm's existing
# ABOUTME: audit runs (stalled, cancelled or dead-target), take the pod down, then push the merged audit.
# Run: scratch/petri_seeds/run_missing.sh <hf target> <tunnel port> [<existing run dir>...]
#   (no run dirs = an arm never audited: all 90 seeds)
#   e.g. run_missing.sh dougalldeepmind/2026-10-05-qwen36-0-da-15 8065 output/petri/2026-10-08_183339_2026-10-05-qwen36-0-da-15
# A fresh run on a subset (--seeds) rather than --retry: a retry reuses the target port the old log
# recorded, which is dead. Use a port no still-running driver points at, or it reconnects.
# The body is one { ...; exit; } block so bash parses all of it before running any: bash reads a
# script as it executes, and an edit mid-run (2026-10-09) made two copies skip their push.
{
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
# RUNPOD_API_KEY too: the shell profile's key is another account's, so `runpod down` would miss the pod.
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY RUNPOD_API_KEY
TARGET=$1
PORT=$2
shift 2
OLD=("$@")
ARM=$(basename "$TARGET" | cut -d- -f4-)   # 2026-10-05-qwen36-0-da-15 -> qwen36-0-da-15
# REPO overrides the derived name (a base model like Qwen/Qwen3.6-27B derives an empty ARM).
REPO=${REPO:-$(date +%Y-%m-%d)-petri-$ARM}
ARM=${ARM:-$(basename "$TARGET")}
STAMP=$(date +%Y%m%d_%H%M%S)
SUB=output/petri/seeds_rerun/${STAMP}_${ARM}
POD_JSON=output/logs/petri_${ARM}_pod.json
LOG=output/logs/${STAMP}_petri_${ARM}_missing.log
# H100, not the H200 the MASK profile entry names: the H200 secure pool returned 500 on every
# create on 2026-10-09. The H100 is the profile's default inference card and holds Petri's
# 8 x 32k-context load (~263k KV tokens, configs/models/qwen36.yaml).
GPU=${GPU:-"NVIDIA H100 80GB HBM3"}
# CTX: the target context window. Seeds that overflow 32k (the base model, 2026-10-16) need 65536.
CTX=${CTX:-32768}
CONN=${CONN:-8}
exec > >(tee -a "$LOG") 2>&1
rm -f "$POD_JSON"   # a stale record would point teardown at an earlier pod

bal () { ./.venv/bin/python -c "
from dotenv import dotenv_values
import json,urllib.request
k=dotenv_values('.env')['OPENROUTER_API_KEY']
d=json.load(urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/credits',headers={'Authorization':'Bearer '+k})))['data']
print(f\"\$ {d['total_credits']-d['total_usage']:.2f}\")"; }
down () {
  POD=$(./.venv/bin/python -c "import json;print(json.load(open('$POD_JSON'))['pod_id'])")
  echo ">>> taking down $POD"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
  echo ">>> surviving matboz pods:"; ./.venv/bin/runpod pods 2>&1 | grep -i matboz || echo "   none"
}

echo ">>> petri rerun of missing seeds -> $TARGET, push to $REPO, budget before: $(bal)"
./.venv/bin/python scratch/petri_seeds/missing_seeds.py "$SUB" "${OLD[@]}" || exit 1
./.venv/bin/python -u scratch/petri_seeds/provision_pod.py \
  --name "matboz-petri-$ARM" --target "$TARGET" --out "$POD_JSON" --max_hours 4 --gpu "$GPU" \
  || { echo "!!! provisioning failed"; [ -f "$POD_JSON" ] && down; exit 1; }
ADDR=$(./.venv/bin/python -c "import json;d=json.load(open('$POD_JSON'));print(f\"root@{d['ip']}:{d['port']}\")")
# "POD READY" is the rental, not the boot: vLLM's venv lands later, and the preflight refuses a host
# without it (2026-10-16). Wait for the boot log's READY, as the other drivers do.
POD=$(./.venv/bin/python -c "import json;print(json.load(open('$POD_JSON'))['pod_id'])")
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY \
    && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; down; exit 1; }
  sleep 30
done

# run_audit's defaults are the plain audit's: sonnet-5 auditor, flash judge, 25 turns, 8 in flight.
./.venv/bin/python -u scratch/petri_seeds/run_audit.py \
  --server "$ADDR" --port "$PORT" --target "$TARGET" --seeds "$SUB" \
  --context-window "$CTX" --connections "$CONN"
RC=$?
NEW=$(ls -td output/petri/*_"$(basename "$TARGET")" | head -1)
echo ">>> rerun landed in $NEW"
echo ">>> audit rc=$RC | budget now: $(bal)"
down
[ "$RC" = 0 ] || exit "$RC"

./.venv/bin/python -u scratch/petri_seeds/push_results.py \
  --run_dirs "$(printf '%s\n' "${OLD[@]}" "$NEW" | ./.venv/bin/python -c 'import json,sys;print(json.dumps(sys.stdin.read().split()))')" \
  --target "$TARGET" --repo "$REPO"
echo ">>> done"
exit
}
