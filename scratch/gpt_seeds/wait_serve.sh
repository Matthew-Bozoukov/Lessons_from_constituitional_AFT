#!/usr/bin/env bash
# ABOUTME: Poll a serve_adapter_runpod.py pod's boot log until SERVE_READY (or a failure marker),
# ABOUTME: then print the log tail and the served model list. Run: bash scratch/gpt_seeds/wait_serve.sh <pod> [max_minutes]
set -u
POD="${1:?pod id}"
MAX="${2:-50}"
for i in $(seq 1 "$MAX"); do
  L=$(curl -s --max-time 20 "https://${POD}-8080.proxy.runpod.net/boot.log")
  if echo "$L" | grep -q "SERVE_READY\|NO MODELS RESPONSE\|Traceback\|Error:"; then break; fi
  sleep 60
done
echo "== boot.log tail =="
curl -s --max-time 20 "https://${POD}-8080.proxy.runpod.net/boot.log" | grep -v "^+ " | tail -8
echo "== models =="
curl -s --max-time 20 "https://${POD}-8000.proxy.runpod.net/v1/models" | head -c 800
echo
