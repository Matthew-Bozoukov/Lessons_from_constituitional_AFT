#!/usr/bin/env bash
# ABOUTME: On the pod: tool-check the three adapters, then launch all 6 sepsis cells (3 arms x single/org)
# ABOUTME: once per replicate prefix given, all detached and side by side on the one vLLM server.
# Run (on the pod, from /workspace/shen): bash pod_launch_wave.sh r3_ r4_     (no args = unprefixed tags)
set -uo pipefail
cd /workspace/shen
ARMS="qwen36_0_plain qwen36_0_da_15 qwen36_0_da_tools_15"
for m in $ARMS; do
  PORT=8011 setsid nohup bash pod_hello.sh "$m" > /dev/null 2>&1 < /dev/null &
done
PREFIXES=("$@")
[ ${#PREFIXES[@]} -eq 0 ] && PREFIXES=("")
for p in "${PREFIXES[@]}"; do
  for m in $ARMS; do
    PORT=8011 MAX_TURNS=100 setsid nohup bash pod_run.sh sepsis 1 "$m" "${p}single_sepsis_$m" > /dev/null 2>&1 < /dev/null &
    PORT=8011 MAX_TURNS=100 setsid nohup bash pod_run.sh sepsis 8 "$m" "${p}org8_sepsis_$m" > /dev/null 2>&1 < /dev/null &
    echo "launched ${p}single_sepsis_$m ${p}org8_sepsis_$m"
  done
done
