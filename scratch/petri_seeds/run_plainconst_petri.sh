#!/usr/bin/env bash
# ABOUTME: Petri audit of the plain control with the whole constitution first in every target system
# ABOUTME: message: verify injection on 2 seeds, run all 90 driven from here, then tear the pod down.
# Run: bash scratch/petri_seeds/run_plainconst_petri.sh [connections]
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY RUNPOD_API_KEY
export PETRI_SYSTEM_PREAMBLE_FILE="$PWD/constitutions/claude_distilled_09_principles/constitution.md"
TARGET="dougalldeepmind/2026-10-05-qwen36-0-plain"
NAME="matboz-petri-plainconst"
CONN="${1:-24}"
CTX=65536       # the constitution adds ~4.5k tokens to every target turn; same window as the overnight run
PORT=8001       # 8000 is the base resume's, so both audits can run at once
MIN_CREDIT=40   # the 2-seed check cost ~$0.6, so 90 seeds is ~$30; margin for long audits
TS=$(date +%Y%m%d_%H%M%S)
LOG="output/logs/${TS}_petri_plainconst.log"
POD_JSON="/tmp/claude-1000/petri_pod_plainconst.json"
mkdir -p output/logs
exec > >(tee -a "$LOG") 2>&1
echo ">>> plain+constitution Petri | target $TARGET | preamble $PETRI_SYSTEM_PREAMBLE_FILE | conn=$CONN ctx=$CTX"
grep -c "_patched_stage_system" .venv/lib/python3.12/site-packages/inspect_petri/target/_controller.py \
  | { read n; [ "$n" -ge 2 ] || { echo "!!! petri injection patch missing"; exit 1; }; } || exit 1
LEFT=$(./.venv/bin/python -c "
from dotenv import dotenv_values; import json, urllib.request
k = dotenv_values('.env')['OPENROUTER_API_KEY']
d = json.load(urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/credits', headers={'Authorization': 'Bearer ' + k})))['data']
print(round(d['total_credits'] - d['total_usage'], 2))")
echo ">>> OpenRouter credit left: \$$LEFT (need >= \$$MIN_CREDIT)"
./.venv/bin/python -c "import sys; sys.exit(0 if $LEFT >= $MIN_CREDIT else 1)" \
  || { echo "!!! not enough OpenRouter credit; no pod rented"; exit 1; }
./.venv/bin/python -u scratch/petri_seeds/provision_pod.py --name "$NAME" --target "$TARGET" \
  --out "$POD_JSON" --max_hours 6 || { echo "!!! provisioning failed"; exit 1; }
POD=$(./.venv/bin/python -c "import json;print(json.load(open('$POD_JSON'))['pod_id'])")
SERVER=$(./.venv/bin/python -c "import json;d=json.load(open('$POD_JSON'));print(f\"root@{d['ip']}:{d['port']}\")")
echo ">>> pod=$POD server=$SERVER"
down() { echo ">>> taking down $POD: $1"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2; }
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { down "never READY"; exit 1; }
  sleep 30
done
audit() { ./.venv/bin/python -u scripts/eval/audits/petri/run_audit.py --server "$SERVER" --target "$TARGET" \
  --context-window "$CTX" "$@"; }

echo ">>> VERIFY: 2 seeds"
# Its own port: the stopped tunnel is terminated, not waited on, and still held 8001 when the
# full run started (2026-10-15).
audit --port $((PORT + 1)) --connections 2 --limit 2 > "${LOG%.log}_verify.txt" 2>&1; tail -5 "${LOG%.log}_verify.txt"
VLOG=$(ls -t output/petri/*_2026-10-05-qwen36-0-plain/logs/*.eval | head -1)
./.venv/bin/python scripts/eval/audits/petri/verify_target_preamble.py "$VLOG" "$PETRI_SYSTEM_PREAMBLE_FILE" \
  || { down "constitution NOT first in every target system message ($VLOG)"; exit 1; }
echo ">>> injection verified in $VLOG"

echo ">>> FULL: 90 seeds"
AOUT="${LOG%.log}_audit.txt"
audit --port "$PORT" --connections "$CONN" > "$AOUT" 2>&1
rc=$?
tail -25 "$AOUT"
grep -oiE "Error code: [0-9]+|in_flight_budget_exhausted|Task interrupted" "$AOUT" | sort -u | sed 's/^/    /'
RUN_LOG=$(ls -t output/petri/*_2026-10-05-qwen36-0-plain/logs/*.eval | head -1)
./.venv/bin/python scripts/eval/audits/petri/verify_target_preamble.py "$RUN_LOG" "$PETRI_SYSTEM_PREAMBLE_FILE" \
  || echo "!!! full-run injection check FAILED ($RUN_LOG)"
echo ">>> petri plain+constitution rc=$rc at $(date -u +%FT%TZ) | log $RUN_LOG"
down "run finished (scored samples are in the local .eval; resume with run_audit.py --retry)"
