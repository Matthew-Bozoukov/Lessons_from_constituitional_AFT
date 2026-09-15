#!/usr/bin/env bash
# ABOUTME: Drive the non-moral-deliberation ODCV run FROM THIS WINDOWS LAPTOP: reconnecting SSH
# ABOUTME: tunnel to the serving pod, then the box supervisor (N passes, audited + pushed per pass).
#
# Run: bash scratch/nonmoral/odcv_local_run.sh <pod_ip> <pod_ssh_port> [passes] [concurrency] [config]
#
# A Windows fork of scratch/da_chunk_only/odcv_local_run.sh. Three adaptations, because this
# host is Git Bash on Windows and that script is macOS:
#   * `caffeinate -i` does not exist. Replaced with SetThreadExecutionState via PowerShell, which
#     is the actual Windows API for "do not sleep while this process lives" -- NOT a synthetic
#     keystroke loop, which would type into whatever window has focus.
#   * `jq` is not installed here. The served-model list is parsed with python instead.
#   * Docker is 20 CPUs / 18.9GB on this box (the Mac was 8 / 8GB), so the concurrency default
#     is raised from 12 to 16. It is still an override, not the config's value.
#
# THE MACHINE MUST NOT SLEEP. The keep-awake below covers system sleep for as long as this
# script's guard process lives, but if Docker Desktop is stopped or the network drops for long
# enough, every in-flight scenario times out and the pass is wasted.
#
# The tunnel is the reason this is not driven over RunPod's HTTPS proxy: docs/LOG.md 2026-08-09
# records the proxy timing out on ODCV's long non-streaming rollouts.
set -uo pipefail

POD_IP="${1:?pod ip required}"
POD_PORT="${2:?pod ssh port required}"
PASSES="${3:-2}"
CONC="${4:-16}"
CFG="${5:-scratch/nonmoral/odcv_bench_t2_9284_nonmoral_684_2x65.yaml}"
HF_REPO=LASR-Callum/2026-09-02-odcv-nonmoral-deliberation-684-eval
STATE=output/odcv_nonmoral_state
mkdir -p "$STATE" output/logs

echo "=== keep-awake (SetThreadExecutionState) ==="
powershell.exe -NoProfile -Command "
  Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport(\"kernel32.dll\", SetLastError=true)] public static extern uint SetThreadExecutionState(uint esFlags);';
  # 2147483649 = ES_CONTINUOUS|ES_SYSTEM_REQUIRED. Written as an unsigned DECIMAL because
  # PowerShell 5.1 parses 0x80000000 as a NEGATIVE Int32 and the uint32 marshal then throws.
  \$prev = [Win32.Power]::SetThreadExecutionState([uint32]2147483649);
  if (\$prev -eq 0) { Write-Output 'FAILED to inhibit sleep'; exit 1 };
  Write-Output 'sleep inhibited; this process holds it';
  while (\$true) { Start-Sleep -Seconds 60 }
" > "$STATE/keepawake.log" 2>&1 &
KEEPAWAKE=$!
echo "keep-awake pid $KEEPAWAKE (dies with this shell; re-run if you restart it)"

echo "=== tunnel -> $POD_IP:$POD_PORT (reconnecting) ==="
pkill -f "ssh -N -L 8000:localhost:8000" 2>/dev/null || true
bash -c "while true; do ssh -N -L 8000:localhost:8000 \
  -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -p $POD_PORT root@$POD_IP; sleep 5; done" \
  > "$STATE/tunnel.log" 2>&1 &
sleep 8

if curl -sf -m 20 http://127.0.0.1:8000/v1/models > "$STATE/models.json"; then
  echo "tunnel OK; served models:"
  python -c "import json,sys; [print('  '+m['id']) for m in json.load(open(sys.argv[1]))['data']]" \
    "$STATE/models.json"
else
  echo "FATAL: tunnel up but endpoint not answering on 127.0.0.1:8000"
  tail -5 "$STATE/tunnel.log" 2>/dev/null
  kill $KEEPAWAKE 2>/dev/null
  exit 1
fi

echo "=== preflight (docker + endpoint) ==="
uv run python scratch/odcv_preflight.py --config "$CFG" --check_docker \
  --base_url http://127.0.0.1:8000/v1 || {
    echo "FATAL: preflight failed"; kill $KEEPAWAKE 2>/dev/null; exit 1; }

echo "=== supervisor: $PASSES passes, concurrency $CONC ==="
uv run python scratch/odcv_box_run.py --config "$CFG" \
  --passes "$PASSES" --box_id laptop --state_dir "$STATE" --hf_repo "$HF_REPO" \
  --extra "concurrency=$CONC" > output/logs/odcv_nonmoral_supervisor.log 2>&1 &
echo "supervisor pid $!  log output/logs/odcv_nonmoral_supervisor.log  status $STATE/status.json"
echo
echo "When the run finishes, kill the keep-awake guard: kill $KEEPAWAKE"
