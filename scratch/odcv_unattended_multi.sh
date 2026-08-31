#!/usr/bin/env bash
# ABOUTME: Run several ODCV configs back to back on one serving pod, then destroy the pod.
# ABOUTME: Usage: odcv_unattended_multi.sh <pod_id> <passes> <config.yaml> [config.yaml ...]
set -uo pipefail
POD="$1"; PASSES="$2"; shift 2
REPO="/home/matthewb/git repos/teaching_claude_why_replication"
cd "$REPO"
set -a; source .env; set +a

echo "=== ODCV unattended: $# configs x $PASSES passes, pod $POD ==="
echo "start $(date -Is)"

# Every config must parse and name a live served model BEFORE any pass runs: a config that
# dies on load would otherwise fall straight through to teardown and waste the whole pod.
for CFG in "$@"; do
  uv run python -c "
import sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1])
for k in ('model', 'model_key', 'base_url', 'concurrency'):
    assert k in c, f'{sys.argv[1]}: missing {k}'
print(f'  config OK {sys.argv[1]} model={c.model}')
" "$CFG" || { echo "ABORT: $CFG failed validation; pod $POD left running"; exit 2; }
done

for CFG in "$@"; do
  echo ">>> CONFIG $CFG  $(date -Is)"
  bash scratch/odcv_repeat_rollouts.sh "$CFG" "$PASSES"
  echo ">>> CONFIG $CFG exited rc=$? $(date -Is)"
done

echo "=== destroying pod $POD ==="
uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
echo "done $(date -Is)"
