#!/bin/bash
# ABOUTME: Poll the deliberation eval pod's progress log and emit new lines plus any failure
# ABOUTME: signature. Usage: bash scratch/watch_pod.sh <pod-id>
POD="${1:?usage: watch_pod.sh <pod-id>}"
ROOT="https://${POD}-8080.proxy.runpod.net"
SEEN=0
while true; do
  CUR=$(curl -s --max-time 25 "$ROOT/progress.log" 2>/dev/null)
  if [ -n "$CUR" ]; then
    N=$(printf '%s\n' "$CUR" | wc -l | tr -d ' ')
    if [ "$N" -gt "$SEEN" ]; then
      printf '%s\n' "$CUR" | tail -n +$((SEEN + 1))
      SEEN=$N
    fi
    printf '%s\n' "$CUR" | grep -q "ALL DONE" && exit 0
  fi
  # Silence is not success: surface the failure signatures too, not just progress.
  for L in llmbar debate_speeches sycophancy; do
    HIT=$(curl -s --max-time 25 "$ROOT/${L}.log" 2>/dev/null | grep -aoE "Traceback|CUDA out of memory|vLLM exited|does not support|TimeoutError|RuntimeError" | tail -1)
    [ -n "$HIT" ] && echo "[$L] $HIT"
  done
  sleep 90
done
