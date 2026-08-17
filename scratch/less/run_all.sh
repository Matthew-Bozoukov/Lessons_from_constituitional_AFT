#!/bin/bash
# ABOUTME: End-to-end LESS run on one pod — warmup, fanned-out datastores (real + control),
# ABOUTME: influence scoring. Usage: run_all.sh <out_root> [n_gpus]
#
# Ordering is forced by dependency, not preference: the warmup is sequential across epochs
# (checkpoint i+1 trains from checkpoint i) and everything after it is embarrassingly
# parallel. So the warmup runs on one GPU and the datastores fan out across all of them.
#
# The control datastore reuses the TRAIN features unchanged and only recomputes the 60
# validation gradients -- which is why the negative control costs minutes rather than
# doubling the run.
set -euo pipefail

OUT="${1:?usage: run_all.sh <out_root> [n_gpus]}"
N="${2:-$(nvidia-smi --list-gpus | wc -l)}"
CFG=configs/train/lora_qwen36_less_warmup_r64.yaml

export HF_HOME=${HF_HOME:-/workspace/hf}
export PYTHONPATH="$(pwd):$(pwd)/scratch/less"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

echo "############ 1/4  warmup LoRA (sequential, 1 GPU) ############"
CUDA_VISIBLE_DEVICES=0 python3 scratch/less/warmup.py --config "$CFG" 2>&1 \
  | tee "$OUT/warmup.log"
WARMUP=$(ls -dt output/less_warmup/*/ | head -1)
echo ">>> warmup checkpoints: $WARMUP"

echo "############ 2/4  gradient datastore, $N GPU(s) ############"
bash scratch/less/fanout.sh "$WARMUP" data/less/d_full.jsonl train "$OUT/grads" "$N"
bash scratch/less/fanout.sh "$WARMUP" data/less/dval.jsonl    val   "$OUT/grads" "$N"

echo "############ 3/4  negative control (val only — train features reuse) ############"
cp "$OUT"/grads/train_*.pt "$OUT/grads_control/" 2>/dev/null || {
  mkdir -p "$OUT/grads_control"; cp "$OUT"/grads/train_*.pt "$OUT/grads_control/"; }
bash scratch/less/fanout.sh "$WARMUP" data/less/dval_control.jsonl val "$OUT/grads_control" "$N"

echo "############ 4/4  influence scoring ############"
python3 scratch/less/influence.py --grads "$OUT/grads" \
  --control-grads "$OUT/grads_control" --out "$OUT/scores" 2>&1 | tee "$OUT/influence.log"

echo "=== DONE. Ranking: $OUT/scores/scores.jsonl"
echo "=== Diagnostics: $OUT/scores/diagnostics.json"
