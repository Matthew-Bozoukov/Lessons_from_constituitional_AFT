#!/usr/bin/env bash
# ABOUTME: Combine both principle-scoped seeds' passes, judge each with the seed-0 judges, and
# ABOUTME: publish. Run AFTER both seeds' rollouts finish: bash scratch/da716_seeds/judge_seeds.sh
#
# Judging is not part of odcv_box_run.py -- that publishes rollouts per pass so a dead laptop
# loses nothing, and scoring happens once at the end over the merged directory. The judges and
# the 15 exclusions come from each seed's config, which is byte-identical to seed 0's but for
# `model` and `model_key`, so all three seeds are scored the same way on the same cells.
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

for SEED in 42 69; do
  CFG="scratch/da716_seeds/2026-08-31_odcv_bench_table2_9284_difficult_advice_principle_scoped_702_seed_${SEED}_2_65.yaml"
  KEY="$(uv run --quiet python -c "import yaml;print(yaml.safe_load(open('$CFG'))['model_key'])")"
  LOG="output/logs/odcv_principle_scoped_s${SEED}_judge.log"

  echo "############ seed $SEED combine  $(date '+%H:%M:%S')"
  uv run python scratch/odcv_combine_passes.py --config "$CFG" 2>&1 | tee -a "$LOG" | tail -n 5

  COMBINED=$(ls -td "output/odcv_bench/${KEY}"/combined*/ 2>/dev/null | head -n 1)
  [ -n "$COMBINED" ] || { echo "FATAL: seed $SEED has no combined dir"; exit 1; }
  N=$(find "$COMBINED" -name messages_record.txt -size +0 | wc -l | tr -d ' ')
  echo "seed $SEED combined -> $COMBINED  ($N non-empty transcripts)"
  # Judging is the expensive half (~90% of an arm's cost); refuse to pay it for a directory
  # that does not hold two passes' worth of transcripts.
  [ "$N" -ge 100 ] || { echo "FATAL: only $N transcripts in $COMBINED — expected ~128"; exit 1; }

  echo "############ seed $SEED judge  $(date '+%H:%M:%S')"
  uv run python scratch/odcv_judge_cli.py --rollout_dir "$COMBINED" --config "$CFG" 2>&1 \
    | tee -a "$LOG" | tail -n 12
done
echo "JUDGING_DONE $(date '+%H:%M:%S')"
