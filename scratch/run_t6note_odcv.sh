#!/usr/bin/env bash
# ABOUTME: Serve the da-t6-note adapter on an ODCV inference pod and drive ODCV-lite (3 passes)
# ABOUTME: from this machine, on a pinned port so a stale tunnel socket cannot collide again.
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
ADAPTER=dougalldeepmind/2026-10-01-qwen36-0-da-t6-note-15
PORT=8010

echo ">>> provisioning the odcv eval pod"
UP=$(./.venv/bin/runpod up matboz-t6note-odcv --eval odcv --target "$ADAPTER" --push_env 2>&1)
echo "$UP" | tail -12
HOST=$(echo "$UP" | grep -oE "host: +root@[0-9.]+:[0-9]+" | head -1 | sed 's/host: *//')
POD=$(echo "$UP" | grep -oE "^pod: +[a-z0-9]+" | head -1 | awk '{print $2}')
if [ -z "$HOST" ]; then
  echo "!!! could not parse the eval pod host. If a pod was created it is NOT being driven:"
  echo "$UP" | grep -iE "pod|BILLING" | head -5
  exit 1
fi
echo ">>> eval pod $POD at $HOST — odcv-lite, passes 3, port $PORT"

# --terminate-pod makes the eval process the pod's owner: it goes away on publication OR
# failure, so a crash past this point does not leave the card billing.
./.venv/bin/evals --name odcv --config configs/eval/odcv/lite.yaml \
  --target "$ADAPTER" --server "$HOST" --ssh-key ~/.ssh/id_ed25519 \
  --port "$PORT" --terminate-pod 2>&1 | tail -45
RC=$?
echo ">>> evals exited rc=$RC"
echo ">>> surviving pods (a matboz-* row here needs taking down):"
./.venv/bin/runpod pods 2>&1 | tail -6
