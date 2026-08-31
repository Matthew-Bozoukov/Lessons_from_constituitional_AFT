#!/usr/bin/env bash
set -euo pipefail
cd "/home/matthewb/git repos/teaching_claude_why_replication"
unset OPENROUTER_API_KEY OPENROUTER_API_KEY_ALT || true
for arm in numina synthdoc; do
  echo "=== judging $arm $(date +%H:%M:%S) ==="
  uv run python scratch/odcv_judge_run.py \
    "output/odcv_worst3/$arm/combined_flash" \
    "configs/eval/odcv_worst3/${arm}_judge.yaml"
done
echo "=== JUDGING DONE $(date +%H:%M:%S) ==="
