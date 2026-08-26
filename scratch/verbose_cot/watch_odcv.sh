#!/usr/bin/env bash
# ABOUTME: Local health + progress watch for the verbose-CoT ODCV run: both serving pods,
# ABOUTME: both arms' vast boxes, and the heartbeats that keep the pod watchdogs stood down.
#
# Emits only on CHANGE or on trouble. Two things it must never do: go quiet because a
# failure path was not in the filter, and stop refreshing the heartbeats while the run is
# genuinely alive -- a stale heartbeat destroys the serving pod within 45 minutes.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HB="$ROOT/output/odcv_verbose/hb"
BOXES="$ROOT/output/odcv_verbose/boxes.env"     # written by the launcher: BOX_<arm><n>=ip:port

declare -A LAST=()

note () {                                     # emit once per distinct message per key
  local key="$1" msg="$2"
  if [ "${LAST[$key]:-}" != "$msg" ]; then LAST[$key]="$msg"; echo "$msg"; fi
}

# --- serving pods -----------------------------------------------------------------------
pod_health () {                               # $1 = arm, $2 = pod id
  local arm="$1" pod="$2" boot code body
  code="$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 20 \
          "https://${pod}-8000.proxy.runpod.net/v1/models" 2>/dev/null || echo 000)"
  if [ "$code" = "200" ]; then
    note "pod-$arm" "[$arm/pod] vLLM SERVING (models endpoint 200)"
    # Only a serving pod can be measured for pressure; preemptions climbing means total
    # concurrency across boxes is above what the GPU can hold.
    body="$(curl -fsS --max-time 20 "https://${pod}-8000.proxy.runpod.net/metrics" 2>/dev/null \
            | grep -aE '^vllm:num_requests_(running|waiting)|^vllm:num_preemptions_total' \
            | awk '{printf "%s=%s ", $1, $2}')"
    [ -n "$body" ] && note "load-$arm" "[$arm/pod] $body"
    return 0
  fi
  boot="$(curl -fsS --max-time 20 "https://${pod}-8080.proxy.runpod.net/boot.log" 2>/dev/null)"
  if [ -z "$boot" ]; then
    note "pod-$arm" "[$arm/pod] booting, proxy not up yet"
    return 1
  fi
  if grep -aqE "Traceback|CUDA out of memory|OutOfMemoryError|ERROR|Killed|too old" <<<"$boot"; then
    echo "[$arm/pod] BOOT FAILURE: $(grep -aE 'Traceback|out of memory|ERROR|Killed|too old' <<<"$boot" | tail -1 | cut -c1-200)"
    return 1
  fi
  note "pod-$arm" "[$arm/pod] booting: $(tail -1 <<<"$boot" | cut -c1-110)"
  return 1
}

# --- vast boxes -------------------------------------------------------------------------
box_health () {                               # $1 = label, $2 = ip, $3 = port
  local label="$1" ip="$2" port="$3" out
  out="$(ssh -i ~/.ssh/msm_audit -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
         -o BatchMode=yes -p "$port" "root@$ip" \
         'cat /root/odcv/state.json 2>/dev/null | head -c 400; echo; \
          ls -1 /root/work/output/odcv_bench/*/*/agent_logs 2>/dev/null | wc -l' 2>/dev/null)"
  if [ -z "$out" ]; then
    note "box-$label" "[$label/box] unreachable at $ip:$port (vast remaps addresses mid-run)"
    return
  fi
  note "box-$label" "[$label/box] $(tr -d '\n' <<<"$out" | cut -c1-220)"
}

while true; do
  # Heartbeats FIRST: an exit-on-error later in the loop must not silently let the
  # watchdogs destroy live pods.
  touch "$HB/rows" "$HB/tokens" 2>/dev/null

  pod_health rows   "${POD_ROWS:?}"   || true
  pod_health tokens "${POD_TOKENS:?}" || true

  if [ -f "$BOXES" ]; then
    # shellcheck disable=SC1090
    . "$BOXES"
    for v in $(compgen -A variable | grep '^BOX_' || true); do
      addr="${!v}"; box_health "${v#BOX_}" "${addr%%:*}" "${addr##*:}"
    done
  fi
  sleep 180
done
