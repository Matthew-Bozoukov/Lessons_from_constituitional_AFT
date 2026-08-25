#!/usr/bin/env bash
# ABOUTME: Emits one line per meaningful change on both verbose-CoT training pods, and
# ABOUTME: exits once both are done or either has failed. Silence here means "no news".
#
# Coverage matters more than tidiness: a filter that only matched the success marker would
# stay silent through an OOM, a mask-gate rejection, a crashloop or a dead proxy, and
# silence would look exactly like healthy training. So every terminal state emits.
set -uo pipefail

ROWS=hs87m2hxk9frxg
TOKENS=jbhtgbz0f6sqzu
FAIL='Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|AssertionError|Killed|torch.distributed.elastic|ERROR|does not match|refus'

declare -A LAST=( [rows]="" [tokens]="" )
declare -A DONE=( [rows]=0 [tokens]=0 )

probe () {                       # $1 = arm label, $2 = pod id
  local arm="$1" pod="$2" url body step
  url="https://${pod}-8080.proxy.runpod.net"
  # -f matters: without it a 404 returns the proxy's HTML error page as a NON-EMPTY body,
  # so the fallback to boot.log never fires and the failure grep ends up scanning HTML --
  # where a stray "ERROR" would mark a healthy arm dead and stop the watch.
  body="$(curl -fsS --max-time 25 "${url}/train.log" 2>/dev/null)"
  if [ -z "$body" ]; then
    body="$(curl -fsS --max-time 25 "${url}/boot.log" 2>/dev/null)"
    if [ -z "$body" ]; then
      # Say this ONCE. A pod takes ~10 minutes to expose its proxy, and repeating the
      # same non-event every two minutes is how a monitor gets itself auto-stopped for
      # noise -- taking the failure alerts down with it.
      if [ "${LAST[$arm]}" != "nolog" ]; then
        LAST[$arm]="nolog"
        echo "[$arm] waiting for the pod proxy (boot + ~55GB base download)"
      fi
      return
    fi
  fi

  # Terminal success.
  if grep -q "TRAINING_DONE" <<<"$body"; then
    if [ "${DONE[$arm]}" -eq 0 ]; then
      DONE[$arm]=1
      echo "[$arm] TRAINING_DONE — pull the adapter and tear the pod down"
    fi
    return
  fi

  # Terminal failure, or anything that smells like one.
  local bad
  bad="$(grep -aoE "$FAIL" <<<"$body" | sort -u | tr '\n' ' ')"
  if [ -n "$bad" ]; then
    echo "[$arm] FAILURE SIGNATURE: $bad"
    echo "[$arm] $(grep -aE "$FAIL" <<<"$body" | tail -2 | cut -c1-220)"
    DONE[$arm]=2
    return
  fi

  # Progress: report only when the hundred-step bucket moves, so a 2.5h run emits ~6 lines.
  step="$(grep -aoE "'?epoch'?[^0-9]*[0-9.]+|[0-9]+/6[0-9][0-9] \[" <<<"$body" | tail -1)"
  step="$(grep -aoE "[0-9]+/6[0-9][0-9]" <<<"$body" | tail -1)"
  if [ -n "$step" ]; then
    local n bucket
    n="${step%%/*}"
    bucket=$(( n / 100 ))
    if [ "${LAST[$arm]}" != "$bucket" ]; then
      LAST[$arm]="$bucket"
      echo "[$arm] step $step"
    fi
  elif [ -z "${LAST[$arm]}" ] || [ "${LAST[$arm]}" = "nolog" ]; then
    # The proxy is live but no optimizer step has been logged yet. Worth exactly one line:
    # it is the transition from "is this pod even alive" to "it is working".
    LAST[$arm]="boot"
    echo "[$arm] log live, pre-training: $(tail -1 <<<"$body" | cut -c1-140)"
  fi
}

while true; do
  probe rows   "$ROWS"
  probe tokens "$TOKENS"
  if [ "${DONE[rows]}" -ne 0 ] && [ "${DONE[tokens]}" -ne 0 ]; then
    echo "both arms reached a terminal state (rows=${DONE[rows]} tokens=${DONE[tokens]}; 1=done 2=failed)"
    exit 0
  fi
  sleep 120
done
