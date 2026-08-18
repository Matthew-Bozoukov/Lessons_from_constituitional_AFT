#!/usr/bin/env bash
# ABOUTME: Copies the prompt set + generator onto a booted pod and starts it detached, so the
# ABOUTME: sweep keeps running after the operator's laptop goes away.
# Run: bash scratch/deploy_fabgen.sh <pod-id> <arm> [samples]

set -euo pipefail

POD="${1:?usage: deploy_fabgen.sh <pod-id> <arm> [samples]}"
ARM="${2:?arm required}"
SAMPLES="${3:-32}"
REPO="/home/matthewb/git repos/teaching_claude_why_replication"

# RunPod exposes SSH on a per-pod host:port; read it from the API rather than guessing.
read -r IP PORT < <(cd "$REPO" && uv run python -c "
import sys; sys.path.insert(0,'.')
from src.eval.misalignment.internalization.scripts.runpod import call
p=call('GET','/pods/$POD')
print(p['publicIp'], p['portMappings']['22'])
")
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT"

echo "pod $POD  ssh $IP:$PORT  arm=$ARM samples=$SAMPLES"
$SSH 'mkdir -p /root/fabgen'
$SCP "$REPO/scratch/fabrication_prompts.json" "root@$IP:/root/fabgen/prompts.json"
$SCP "$REPO/scratch/pod_generate.py" "root@$IP:/root/pod_generate.py"

# setsid detaches from the SSH session's process group: without it the generator dies the
# moment this connection closes, which is the whole failure this deployment exists to avoid.
# PID 1 is the pod bootstrap and must never be signalled.
$SSH "python3 -m pip install -q openai >/dev/null 2>&1; \
      cd /root && setsid nohup python3 /root/pod_generate.py \
        --arm $ARM --samples $SAMPLES --prompts /root/fabgen/prompts.json \
        --out /root/fabgen </dev/null > /root/fabgen/run.log 2>&1 & \
      sleep 5; echo '--- first log lines ---'; head -3 /root/fabgen/run.log"

echo "started. poll with:"
echo "  ssh -p $PORT root@$IP 'tail -5 /root/fabgen/run.log'"
