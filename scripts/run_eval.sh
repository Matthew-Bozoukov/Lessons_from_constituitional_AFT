#!/bin/bash
# ABOUTME: Remote driver: runs the agentic-misalignment eval end-to-end against a
# ABOUTME: local vLLM endpoint and computes per-condition misalignment rates.
set -euo pipefail

EXPID="${1:?usage: run_eval.sh <experiment_id> <config.yaml> [samples] [model]}"
CONFIG_SRC="${2:?usage: run_eval.sh <experiment_id> <config.yaml> [samples] [model]}"
SAMPLES_OVERRIDE="${3:-}"
MODEL_OVERRIDE="${4:-}"

cd /root/work/third_party/agentic-misalignment
ln -sf /root/work/.env .env   # so load_environment() finds the keys
set -a; source /root/work/.env; set +a
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="EMPTY"

# Per-run config with the requested experiment_id (and optional samples override).
CFG="/root/${EXPID}.yaml"
python - "$CONFIG_SRC" "$EXPID" "$CFG" "$SAMPLES_OVERRIDE" "$MODEL_OVERRIDE" <<'PY'
import sys, yaml
src, expid, dst, samples, model = sys.argv[1:6]
d = yaml.safe_load(open(src))
d["experiment_id"] = expid
if samples:
    d["global"]["samples_per_condition"] = int(samples)
if model:
    d["global"]["models"] = [model]
yaml.safe_dump(d, open(dst, "w"))
print(f"wrote {dst} (experiment_id={expid}, samples={samples or 'default'}, model={model or 'config'})")
PY

echo "=== [1/4] generate_prompts ==="
python scripts/generate_prompts.py --config "$CFG"
echo "=== [2/4] run_experiments (classification deferred to OpenRouter judge) ==="
python scripts/run_experiments.py --config "$CFG" --no-classification
echo "=== [3/4] classify_results (judge=sonnet-4.5 via OpenRouter) ==="
python scripts/classify_results.py --results-dir "results/${EXPID}" \
  --classifier-model anthropic/claude-sonnet-4.5
echo "=== [4/4] aggregate misalignment rates ==="
python /root/work/scripts/aggregate_eval.py \
  --results_dir "results/${EXPID}" --label "${EXPID}" \
  --out "results/${EXPID}/misalignment_summary.json"

echo "=== DONE ${EXPID}: results/${EXPID}/misalignment_summary.json ==="
