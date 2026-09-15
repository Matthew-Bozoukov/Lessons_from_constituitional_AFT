#!/usr/bin/env bash
# ABOUTME: Drive the GPT seed-replicate ODCV runs from the laptop: pin the serving pod into both
# ABOUTME: eval configs, then per seed 2 passes x 65 cells (sequential), combine, judge.
# Run: bash scratch/gpt_seeds/run_odcv.sh <serve pod id> [seeds="42 69"]   (from the repo root; logs under output/logs/)
set -uo pipefail
POD="${1:?serve pod id (scratch/serve_adapter_runpod.py up)}"
SEEDS="${2:-42 69}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs
for SEED in $SEEDS; do
  CFG="configs/eval/odcv_bench_t2_9284_gptresp685_s${SEED}_r64_paired_2x65.yaml"
  # The config is written with a SERVE_POD placeholder so the pod id is pinned here, once,
  # and recorded in the committed config for the run's provenance.
  sed -i '' -e "s#https://SERVE_POD-8000#https://${POD}-8000#" "$CFG"
  grep -q "https://${POD}-8000" "$CFG" || { echo "base_url not pinned in $CFG"; exit 1; }
  KEY="qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed${SEED}"
  echo "############ seed $SEED rollouts $(date '+%H:%M:%S')"
  bash scratch/odcv_repeat_rollouts.sh "$CFG" 2 2>&1 | tee "output/logs/odcv_gptseed${SEED}_rollouts.log" | grep -v "^\[.*\] .*ETA" | tail -n 3
  echo "############ seed $SEED combine $(date '+%H:%M:%S')"
  uv run python scratch/odcv_combine_passes.py --config "$CFG" 2>&1 | tee "output/logs/odcv_gptseed${SEED}_combine.log" | tail -n 8
  COMBINED=$(ls -td "output/odcv_bench/${KEY}"/combined*/ | head -n 1)
  echo "############ seed $SEED judge -> $COMBINED $(date '+%H:%M:%S')"
  uv run python scratch/odcv_judge_cli.py --rollout_dir "$COMBINED" --config "$CFG" 2>&1 | tee "output/logs/odcv_gptseed${SEED}_judge.log" | tail -n 12
done
echo "ODCV_ALL_DONE $(date '+%H:%M:%S')"
