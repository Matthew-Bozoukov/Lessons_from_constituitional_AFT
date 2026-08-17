#!/bin/bash
# ABOUTME: Wait for the pod's first real generation throughput reading and print it once.
# ABOUTME: Usage: bash scratch/probe_throughput.sh <pod-id>   (CLAUDE.md gotcha 7: measure, don't estimate)
POD="${1:?usage: probe_throughput.sh <pod-id>}"
ROOT="https://${POD}-8080.proxy.runpod.net"
for _ in $(seq 1 120); do
  LOG=$(curl -s --max-time 30 "$ROOT/output/llmbar/server/vllm.log" 2>/dev/null)
  HIT=$(printf '%s' "$LOG" | grep -aoE "Avg generation throughput: [0-9.]+ tokens/s.*" | tail -1)
  if [ -n "$HIT" ]; then
    echo "THROUGHPUT $HIT"
    printf '%s' "$LOG" | grep -aoE "GPU KV cache size: [0-9,]+ tokens" | tail -1
    exit 0
  fi
  printf '%s' "$LOG" | grep -aqE "Application startup complete" && echo "vLLM up, no throughput line yet"
  sleep 45
done
echo "no throughput reading after ~90 min — check the pod"
