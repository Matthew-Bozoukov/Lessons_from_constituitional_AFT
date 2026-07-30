#!/bin/bash
# ABOUTME: Driver for the ODCV-Bench replication: agent rollouts then 4-judge scoring,
# ABOUTME: teeing all output to a timestamped log under output/odcv_bench/.
set -euo pipefail

CONFIG="${1:-configs/odcv_bench.yaml}"
SMOKE_FLAG="${2:-}"   # pass "--smoke" for the 2-scenario wiring check

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="output/odcv_bench/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_${TS}${SMOKE_FLAG:+_smoke}.log"

echo "=== ODCV-Bench replication | config=${CONFIG} ${SMOKE_FLAG} ==="
echo "=== logging to ${LOG} ==="

{
  echo "### [1/2] agent rollouts"
  uv run python -m src.eval.misalignment.odcv_rollout --config "$CONFIG" ${SMOKE_FLAG}

  # The rollout writes to output/odcv_bench/<model_key>/<timestamp>/; pick the newest.
  MODEL_KEY="$(uv run python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('${CONFIG}').model_key)")"
  ROLLOUT_DIR="$(ls -1dt output/odcv_bench/${MODEL_KEY}/*/ | head -1)"
  echo "### rollout dir: ${ROLLOUT_DIR}"

  echo "### [2/2] judging"
  uv run python -m src.eval.misalignment.odcv_judge --rollout_dir "${ROLLOUT_DIR}" \
    --config "$CONFIG" ${SMOKE_FLAG}

  echo "### DONE -> ${ROLLOUT_DIR}results.json"
} 2>&1 | tee "$LOG"
