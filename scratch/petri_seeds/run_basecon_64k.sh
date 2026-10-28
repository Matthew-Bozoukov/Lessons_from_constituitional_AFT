#!/usr/bin/env bash
# ABOUTME: Fresh 90-seed base+constitution Petri audit on an already-rented pod, target served at a
# ABOUTME: 64k context so the constitution preamble + long thinking never hit the window.
# Run: bash scratch/petri_seeds/run_basecon_64k.sh <pod_id> root@<ip>:<port> [connections]
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
export PETRI_SYSTEM_PREAMBLE_FILE="$PWD/constitutions/claude_distilled_09_principles/constitution.md"
TARGET="Qwen/Qwen3.6-27B"
POD="$1"
SERVER="$2"
CONN="${3:-32}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="output/logs/${TS}_petri_basecon_64k.log"
mkdir -p output/logs
exec > >(tee -a "$LOG") 2>&1
echo ">>> fresh 90 on $POD | context 65536 | conn=$CONN | preamble $PETRI_SYSTEM_PREAMBLE_FILE"
grep -c "_patched_stage_system" .venv/lib/python3.12/site-packages/inspect_petri/target/_controller.py \
  | { read n; [ "$n" -ge 2 ] || { echo "!!! petri injection patch missing"; exit 1; }; }
grep -q "VENDORED PATCH (2026-10-14)" .venv/lib/python3.12/site-packages/inspect_ai/model/_providers/openrouter.py \
  || { echo "!!! openrouter cache patch missing"; exit 1; }
AOUT="${LOG%.log}_audit.txt"
./.venv/bin/python -u scratch/petri_seeds/run_audit.py --server "$SERVER" --target "$TARGET" \
  --connections "$CONN" --context-window 65536 > "$AOUT" 2>&1
rc=$?
tail -25 "$AOUT"
if grep -qiE "no samples completed|Task interrupted|APIStatusError" "$AOUT"; then
  echo "!!! interrupted; pod $POD KEPT"
  grep -oiE "Error code: [0-9]+|in_flight_budget_exhausted|Task interrupted" "$AOUT" | sort -u | sed 's/^/    /'
  rc=1
fi
echo ">>> petri basecon 64k rc=$rc at $(date -u +%FT%TZ)"
if [ "$rc" -eq 0 ]; then ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
else echo ">>> pod $POD kept: uv run runpod down --pod $POD"; fi
