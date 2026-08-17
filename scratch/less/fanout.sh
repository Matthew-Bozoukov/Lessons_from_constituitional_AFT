#!/bin/bash
# ABOUTME: Fans the gradient datastore across every visible GPU, one shard per device,
# ABOUTME: with per-shard logs and a failure-aware wait. Usage: fanout.sh <warmup> <rows> <split> <out> [n]
#
# Sharding is BY EXAMPLE: each worker loads the 54GB base once and swaps adapter weights
# across the four checkpoints, so the base-model load is paid once per GPU rather than
# once per (GPU, checkpoint). See scratch/less/gradients.py for why the shards are
# independent, and scratch/less/verify_shards.py for the proof that they are.
set -uo pipefail

WARMUP="${1:?usage: fanout.sh <warmup_dir> <rows_jsonl> <split> <out_dir> [n_gpus]}"
ROWS="${2:?}"
SPLIT="${3:?}"
OUT="${4:?}"
N="${5:-$(nvidia-smi --list-gpus | wc -l)}"
shift 5 2>/dev/null || shift 4          # remaining args pass through to gradients.py

export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/scratch/less"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

echo "=== fanning $SPLIT across $N GPU(s) -> $OUT"
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES="$i" nohup python3 scratch/less/gradients.py \
      --warmup "$WARMUP" --rows "$ROWS" --split "$SPLIT" --out "$OUT" \
      --shard "$i" --num-shards "$N" "$@" \
      > "$OUT/shard${i}.log" 2>&1 </dev/null &
  pids+=($!)
  echo "    shard $i -> GPU $i (pid ${pids[-1]}, log $OUT/shard${i}.log)"
done

# Wait on each pid individually: a bare `wait` returns the LAST exit status, so one shard
# dying while the others succeed would look like a clean run and silently produce a
# datastore missing a quarter of D.
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!!! shard $i FAILED (see $OUT/shard${i}.log)"
    tail -20 "$OUT/shard${i}.log"
    fail=1
  fi
done

produced=$(ls "$OUT"/${SPLIT}_ckpt_epoch*_shard*.pt 2>/dev/null | wc -l)
echo "=== $SPLIT done: $produced shard files, fail=$fail"
exit "$fail"
