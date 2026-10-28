#!/usr/bin/env bash
# ABOUTME: Resumes the no-constitution base Qwen3.6-27B Petri audit (latest retry log, 38/90 scored) on a
# ABOUTME: fresh pod: re-runs only the unscored seeds, then always tears the pod down (logs stay local).
# Run: bash scratch/petri_seeds/run_base_resume.sh [connections]
set -uo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY HF_TOKEN HUGGINGFACE_API_KEY RUNPOD_API_KEY PETRI_SYSTEM_PREAMBLE_FILE
TARGET="Qwen/Qwen3.6-27B"
NAME="matboz-petri-base-resume"
RETRY="output/petri/2026-10-13_102629_Qwen3.6-27B/logs/2026-10-15T16-18-06-00-00_audit_LAjMYVnde7t94K6T7zkzg6.eval"  # 38/90 scored after the 2026-10-15 pause
CONN="${1:-24}"
MIN_CREDIT=10   # ~71 seeds at ~$0.45 each; a shortfall is handled by watch_credit_push.py (stop, push partial)
TS=$(date +%Y%m%d_%H%M%S)
LOG="output/logs/${TS}_petri_base_resume.log"
POD_JSON="/tmp/claude-1000/petri_pod_base_resume.json"
mkdir -p output/logs
exec > >(tee -a "$LOG") 2>&1
echo ">>> resume $(basename "$RETRY") | no preamble | conn=$CONN"
[ -f "$RETRY" ] || { echo "!!! no log $RETRY"; exit 1; }
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
for i in $(seq 1 80); do
  curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY && { echo ">>> READY after $((i*30))s"; break; }
  [ "$i" = 80 ] && { echo "!!! never READY"; ./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2; exit 1; }
  sleep 30
done
AOUT="${LOG%.log}_audit.txt"
# --port 8000: eval-retry reuses the base_url the log recorded (http://127.0.0.1:8000/v1).
./.venv/bin/python -u scripts/eval/audits/petri/run_audit.py --server "$SERVER" --target "$TARGET" \
  --connections "$CONN" --port 8000 --retry "$RETRY" > "$AOUT" 2>&1
rc=$?
tail -25 "$AOUT"
grep -oiE "Error code: [0-9]+|in_flight_budget_exhausted|Task interrupted" "$AOUT" | sort -u | sed 's/^/    /'
echo ">>> petri base resume rc=$rc at $(date -u +%FT%TZ); taking down $POD (scored samples are in the local .eval)"
./.venv/bin/runpod down --pod "$POD" 2>&1 | tail -2
