#!/usr/bin/env bash
# ABOUTME: Judges the six extra grok rollout passes, one per pass dir, sequentially.
# ABOUTME: Judging is OpenRouter-only, so it needs no pod and cannot contend with rollouts.
set -uo pipefail
cd "$(dirname "$0")/.."
SP="${1:?usage: judge_grok_r3.sh <scratchpad>}"
set -a; . .env; set +a
for tag in seed0_p1 seed0_p2 seed42_p1 seed42_p2 seed69_p1 seed69_p2; do
  log="output/logs/grokR3_judge_${tag}.log"
  if [ -f "$log" ] && grep -q "judging cost" "$log"; then
    echo "=== $tag already judged, skipping"; continue
  fi
  run=$(find "$SP/grokR3/out_$tag" -maxdepth 2 -name "2026*" -type d 2>/dev/null | head -1)
  [ -z "$run" ] && { echo "!!! $tag: no run dir"; continue; }
  echo "=== judging $tag $(date +%H:%M:%S)"
  uv run python scratch/odcv_judge_run.py "$run" "$SP/grokR3/${tag}.yaml" > "$log" 2>&1
  grep -E "ours      MR" "$log" | tail -1
done
echo ALL_JUDGED
