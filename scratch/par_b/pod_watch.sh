#!/usr/bin/env bash
# ABOUTME: Poll a training pod's status every 3 min and print only NEW lines that need action
# ABOUTME: (pod state changes, TRAINING_* markers, errors) -- a quiet event stream.
# Run: bash scratch/par_b/pod_watch.sh <pod_id> [pattern]
set -u
POD="${1:?pod id required}"
PAT="${2:-status:|TRAINING_|Traceback|Error|OOM|Killed|unreachable}"
PREV="output/logs/pod_${POD}_prev.txt"
: > "$PREV"
while true; do
  s=$(uv run python scratch/par_b/train_pod.py status --pod "$POD" 2>/dev/null)
  cur=$(printf '%s\n' "$s" | grep -E "$PAT")
  new=$(printf '%s\n' "$cur" | grep -vxF -f "$PREV" || true)
  if [ -n "$new" ]; then printf '%s\n' "$new"; fi
  printf '%s\n' "$cur" > "$PREV"
  sleep 180
done
