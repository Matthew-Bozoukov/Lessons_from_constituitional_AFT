#!/usr/bin/env bash
# ABOUTME: Drive the channel-swap ODCV runs from the laptop: 2 passes x 65 cells for each swap arm,
# ABOUTME: sequentially (each pass already runs 12 docker scenarios), then combine + judge each arm.
# Run: bash scratch/channel_swap/run_odcv.sh   (from the repo root; logs under output/logs/)
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p output/logs
for ARM in gtrace_sreply703 strace_greply703; do
  CFG="configs/eval/odcv_bench_t2_9284_${ARM}_r64_paired_2x65.yaml"
  echo "############ $ARM rollouts $(date '+%H:%M:%S')"
  bash scratch/odcv_repeat_rollouts.sh "$CFG" 2 2>&1 | tee "output/logs/odcv_${ARM}_rollouts.log" | grep -v "^\[.*\] .*ETA" | tail -n 3
  echo "############ $ARM combine $(date '+%H:%M:%S')"
  uv run python scratch/odcv_combine_passes.py --config "$CFG" 2>&1 | tee "output/logs/odcv_${ARM}_combine.log" | tail -n 8
  COMBINED=$(ls -td output/odcv_bench/qwen3_6-27b-lora-t2-9284-${ARM//_/-}-paired-r64/combined*/ | head -n 1)
  echo "############ $ARM judge -> $COMBINED $(date '+%H:%M:%S')"
  uv run python scratch/odcv_judge_cli.py --rollout_dir "$COMBINED" --config "$CFG" 2>&1 | tee "output/logs/odcv_${ARM}_judge.log" | tail -n 12
done
echo "ODCV_ALL_DONE $(date '+%H:%M:%S')"
