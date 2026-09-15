#!/usr/bin/env bash
# ABOUTME: Watch a serving pod's boot and say plainly whether it is progressing or a dud.
# ABOUTME: Run: bash scratch/nonmoral/watch_serve.sh <pod_id> [dud_after_min]
#
# WHY A DEADLINE. A RunPod pod can sit in desiredStatus RUNNING with publicIp "" and both /http
# ports returning 404 forever: the container never starts, nothing listens, and it bills the full
# GPU rate the whole time. One did exactly that for an hour on 2026-09-02 before being noticed.
# The early state of a HEALTHY pod is identical, so the only thing that distinguishes them is
# how long it lasts -- hence a deadline rather than a status check.
set -uo pipefail
POD="${1:?pod id required}"
DUD_MIN="${2:-15}"
BOOT="https://${POD}-8080.proxy.runpod.net/boot.log"
API="https://${POD}-8000.proxy.runpod.net/v1/models"
deadline=$(( $(date +%s) + DUD_MIN * 60 ))

while true; do
  if curl -s -m 20 "$API" 2>/dev/null | grep -q '"id"'; then
    echo "SERVING — $API is answering"; exit 0
  fi
  code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' "$BOOT" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "booting OK (log server up); waiting for vLLM to finish loading weights"
    deadline=$(( $(date +%s) + DUD_MIN * 60 ))     # progress: reset the clock
  elif [ "$(date +%s)" -ge "$deadline" ]; then
    echo "DUD: ${DUD_MIN}m with no log server and no API on $POD — terminate and re-provision"
    exit 1
  fi
  sleep 60
done
