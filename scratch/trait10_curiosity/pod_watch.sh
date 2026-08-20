#!/usr/bin/env bash
# ABOUTME: Poll a training pod's status every 3 min and print only NEW log lines of interest
# ABOUTME: (state changes, TRAINING_* markers, losses, errors) -- a filtered event stream.
# Run: bash scratch/trait10_curiosity/pod_watch.sh <pod_id>
set -u
POD="${1:?pod id required}"
PREV="output/logs/pod_${POD}_prev.txt"
: > "$PREV"
while true; do
  s=$(uv run python scratch/trait10_curiosity/train_pod.py status --pod "$POD" 2>/dev/null)
  cur=$(printf '%s\n' "$s" | grep -E "status:|TRAINING_|Traceback|Error|OOM|Killed|train_loss|'loss'|epoch|>>> |not yet|unreachable")
  new=$(printf '%s\n' "$cur" | grep -vxF -f "$PREV" || true)
  if [ -n "$new" ]; then printf '%s\n' "$new"; fi
  printf '%s\n' "$cur" > "$PREV"
  sleep 180
done
