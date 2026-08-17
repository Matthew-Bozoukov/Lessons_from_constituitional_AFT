#!/bin/bash
# ABOUTME: Wait for the first arm's results.json and print the numbers that fix the ETA —
# ABOUTME: call count and mean trace length. Usage: bash scratch/probe_first_arm.sh <pod-id>
POD="${1:?usage: probe_first_arm.sh <pod-id>}"
ROOT="https://${POD}-8080.proxy.runpod.net"
for _ in $(seq 1 160); do
  IDX=$(curl -s --max-time 30 "$ROOT/output/llmbar/" 2>/dev/null)
  ARM=$(printf '%s' "$IDX" | grep -oE 'href="[^"/]+/"' | sed 's/href="//;s/\/"//' | grep -v '^server$' | head -1)
  if [ -n "$ARM" ]; then
    RUN=$(curl -s --max-time 30 "$ROOT/output/llmbar/$ARM/" | grep -oE 'href="[^"/]+/"' | sed 's/href="//;s/\/"//' | head -1)
    JSON=$(curl -s --max-time 60 "$ROOT/output/llmbar/$ARM/$RUN/results.json" 2>/dev/null)
    if printf '%s' "$JSON" | grep -q "adversarial_accuracy"; then
      echo "ARM $ARM"
      printf '%s' "$JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); t=d.get('trace',{}); print('n_calls', t.get('n_calls'), '| think_chars_mean', t.get('think_chars_mean'), '| answer_chars_mean', t.get('answer_chars_mean'), '| empty_think_rate', t.get('empty_think_rate')); print('accuracy', d['accuracy']['rate'], '| consistency', d['consistency']['rate'], '| adversarial', d.get('adversarial_accuracy'))"
      exit 0
    fi
  fi
  sleep 60
done
echo "first arm did not land in ~2.5h — check the pod"
