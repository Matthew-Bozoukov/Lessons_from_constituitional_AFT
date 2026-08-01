#!/usr/bin/env bash
# ABOUTME: One-shot health check of pod, vLLM, supervisor processes and epoch progress.
# ABOUTME: Prints ONLY anomalies and state changes, so it can drive a quiet monitor.
#
# Emits nothing when everything is as expected. That is deliberate: a check that
# prints on every run trains you to ignore it, and this project has already lost
# a pod to a stale heartbeat and leaked monitors to a shared state file - both
# conditions this would have surfaced.
#
# Exit code is always 0; problems are reported on stdout as ALERT lines.

set -uo pipefail
cd "$(dirname "$0")/../../../vulnerabilities" 2>/dev/null || exit 0

STATE="${TMPDIR:-/tmp}/petri_sanity_state"
EP=runtime/provider-monitor/ssh-endpoint.json
KEYFILE="$HOME/.config/msm-audit/infra.env"

emit() { echo "$(date '+%H:%M') $*"; }

# ---- provider: exactly one pod of mine, and the balance is not draining ------
KEY=$(grep -oP '(?<=^RUNPOD_API_KEY=).*' "$KEYFILE" 2>/dev/null | tr -d '\r')
if [ -n "$KEY" ]; then
  RESP=$(curl -s -m 25 -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { clientBalance pods { name desiredStatus } } }"}' 2>/dev/null || true)
  MINE=$(printf '%s' "$RESP" | grep -o '"name":"nika-[^"]*"' | wc -l)
  BAL=$(printf '%s' "$RESP" | python -c "import sys,json;print(round(json.load(sys.stdin)['data']['myself']['clientBalance'],2))" 2>/dev/null || echo "?")
  [ "$MINE" -eq 0 ] && emit "ALERT pod: none of mine running (expected 1) - epochs cannot proceed"
  [ "$MINE" -gt 1 ] && emit "ALERT pod: $MINE of mine running (expected 1) - duplicate provisioning"
  case "$BAL" in ?|"") ;; *) awk -v b="$BAL" 'BEGIN{if(b+0<40) exit 0; exit 1}' && emit "ALERT balance: \$$BAL low";; esac
fi

# ---- supervisors: exactly one monitor, one watchdog, at least one keeper -----
count_proc() {
  powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='powershell.exe'\" -EA SilentlyContinue | Where-Object { \$_.CommandLine -match '$1' }).Count" 2>/dev/null | tr -d '\r\n '
}
MON=$(count_proc 'Monitor-Loop'); WD=$(count_proc 'Watchdog-Loop'); KP=$(count_proc 'HeartbeatKeeper')
[ "${MON:-0}" != "1" ] && emit "ALERT monitor: ${MON:-?} running (expected 1)"
[ "${WD:-0}" != "1" ]  && emit "ALERT watchdog: ${WD:-?} running (expected 1)"
[ "${KP:-0}" -lt 1 ] 2>/dev/null && emit "ALERT keeper: none running - watchdog will reap the pod on idle timeout"

# ---- heartbeat freshness: a stale lease is how the last pod died -------------
HB=runtime/watchdog/heartbeat.json
if [ -f "$HB" ]; then
  AGE=$(python -c "
import json,datetime,sys
d=json.load(open('$HB',encoding='utf-8-sig'))
t=datetime.datetime.fromisoformat(d['last_heartbeat_utc'].replace('Z','+00:00'))
print(int((datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()/60))
" 2>/dev/null || echo 0)
  [ "${AGE:-0}" -gt 20 ] 2>/dev/null && emit "ALERT heartbeat: ${AGE}min stale - pod at risk of idle reap"
fi

# ---- vLLM: reachable and serving all four arms ------------------------------
if [ -f "$EP" ] && [ -n "$KEY" ]; then
  SSHKEY=$(grep -oP '(?<=^MSM_SSH_PRIVATE_KEY=).*' "$KEYFILE" 2>/dev/null | tr -d '\r')
  HOST=$(python -c "import json;print(json.load(open('$EP',encoding='utf-8-sig'))['ssh_host'])" 2>/dev/null || echo "")
  PORT=$(python -c "import json;print(json.load(open('$EP',encoding='utf-8-sig'))['ssh_port'])" 2>/dev/null || echo "")
  if [ -n "$HOST" ] && [ "${MINE:-0}" -ge 1 ]; then
    ARMS=$(ssh -i "$SSHKEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" "root@$HOST" \
      'curl -s -m 10 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -o "\"id\":" | wc -l' 2>/dev/null || echo 0)
    if [ "${ARMS:-0}" -eq 0 ]; then
      SERVING=$(ssh -i "$SSHKEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" "root@$HOST" \
        'pgrep -f "vll[m] serve" >/dev/null && echo loading || echo dead' 2>/dev/null || echo unknown)
      [ "$SERVING" = "dead" ] && emit "ALERT vllm: process not running on the pod"
    elif [ "${ARMS:-0}" -ne 4 ]; then
      emit "ALERT vllm: serving ${ARMS} arms (expected 4)"
    fi
  fi
fi

# ---- progress: report only when the epoch count changes ---------------------
PETRI=../teaching-claude-why/petri
DONE=$(ls -d "$PETRI"/logs/v2-e* 2>/dev/null | wc -l)
PREV=$(cat "$STATE" 2>/dev/null || echo "")
if [ "$DONE" != "$PREV" ]; then
  emit "PROGRESS epoch batches on disk: $DONE (was ${PREV:-none})"
  echo "$DONE" > "$STATE"
fi
exit 0
