#!/usr/bin/env bash
# ABOUTME: Run ODCV passes unattended: blocks laptop suspend, then destroys the serving pod.
# ABOUTME: Usage: odcv_unattended.sh <config.yaml> <passes> <pod_id>
set -euo pipefail
CFG="$1"; PASSES="$2"; POD="$3"
REPO="/home/matthewb/git repos/teaching_claude_why_replication"
cd "$REPO"
set -a; source .env; set +a

echo "=== ODCV unattended: $PASSES passes, pod $POD ==="
echo "start $(date -Is)"

# The passes may fail; the pod must be destroyed either way.
set +e
bash scratch/odcv_repeat_rollouts.sh "$CFG" "$PASSES"
RC=$?
set -e
echo "passes exited rc=$RC at $(date -Is)"

echo "=== destroying pod $POD ==="
uv run python scratch/serve_adapter_runpod.py down --pod "$POD"
uv run python scratch/list_runpod_pods.py 2>/dev/null | tail -5
echo "done $(date -Is)"
