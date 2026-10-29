#!/usr/bin/env bash
# ABOUTME: Resumes the interrupted base+constitution Petri audit on a fresh pod: re-runs only the
# ABOUTME: unscored seeds of the given .eval log, constitution first in every target system message.
# Run: bash scratch/petri_seeds/run_basecon_resume.sh <retry.eval> [connections]
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY
export PETRI_SYSTEM_PREAMBLE_FILE="$PWD/constitutions/claude_distilled_09_principles/constitution.md"
TARGET="Qwen/Qwen3.6-27B"
NAME="matboz-petri-basecon-resume"
RETRY="$1"
CONN="${2:-24}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="output/logs/${TS}_petri_basecon_resume2.log"
POD_JSON="/tmp/petri_pod_basecon_resume.json"
mkdir -p output/logs
exec > >(tee -a "$LOG") 2>&1
echo ">>> resume $(basename "$RETRY") | preamble $PETRI_SYSTEM_PREAMBLE_FILE | conn=$CONN"
grep -c "_patched_stage_system" .venv/lib/python3.12/site-packages/inspect_petri/target/_controller.py \
  | { read n; [ "$n" -ge 2 ] || { echo "!!! petri injection patch missing"; exit 1; }; }
grep -q "VENDORED PATCH (2026-10-14)" .venv/lib/python3.12/site-packages/inspect_ai/model/_providers/openrouter.py \
  || { echo "!!! openrouter cache patch missing"; exit 1; }
./.venv/bin/python -u scratch/petri_seeds/provision_pod.py --name "$NAME" --target "$TARGET" \
  --out "$POD_JSON" --max_hours 6 || { echo "!!! provisioning failed"; exit 1; }
POD=$(./.venv/bin/python -c "import json;print(json.load(open('$POD_JSON'))['pod_id'])")
SERVER=$(./.venv/bin/python -c "import json;d=json.load(open('$POD_JSON'));print(f\"root@{d['ip']}:{d['port']}\")")
echo ">>> pod=$POD server=$SERVER"
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY; pod $POD kept"; exit 1; }
  sleep 30
done
AOUT="${LOG%.log}_audit.txt"
./.venv/bin/python -u scripts/eval/audits/petri/run_audit.py --server "$SERVER" --target "$TARGET" \
  --connections "$CONN" --retry "$RETRY" > "$AOUT" 2>&1
rc=$?
tail -25 "$AOUT"
if grep -qiE "no samples completed|Task interrupted|APIStatusError" "$AOUT"; then
  echo "!!! interrupted; pod $POD KEPT"
  grep -oiE "Error code: [0-9]+|in_flight_budget_exhausted|Task interrupted" "$AOUT" | sort -u | sed 's/^/    /'
  rc=1
fi
echo ">>> petri basecon resume2 rc=$rc at $(date -u +%FT%TZ)"
if [ "$rc" -eq 0 ]; then ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
else echo ">>> pod $POD kept: uv run runpod down --pod $POD"; fi
