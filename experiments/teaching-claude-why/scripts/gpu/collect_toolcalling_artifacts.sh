#!/usr/bin/env bash
# ABOUTME: Gathers the training run's artifacts on the box into one directory for download.
# ABOUTME: Everything except the adapter weights, which are pulled separately.

set -euo pipefail
cd /workspace

RUN=$(ls -1d output/train_lora_toolcalling/2* | tail -1)
OUT=/workspace/artifacts
rm -rf "$OUT"; mkdir -p "$OUT"
echo "run dir: $RUN"

# TRL writes trainer_state.json into the checkpoint dir, not the run root, when
# save_strategy is per-epoch. Take the last checkpoint's copy.
STATE=$(ls -1 "$RUN"/checkpoint-*/trainer_state.json 2>/dev/null | tail -1 || true)
if [ -z "$STATE" ]; then
  STATE=$(find "$RUN" -name trainer_state.json | tail -1)
fi
cp "$STATE" "$OUT/trainer_state.json"
echo "trainer_state: $STATE"

cp "$RUN/run_meta.json" "$OUT/run_meta.json"
cp "$RUN/adapter/adapter_config.json" "$OUT/adapter_config.json"
cp /workspace/config.yaml "$OUT/resolved_config.json.yaml"

# Strip the carriage-return progress-bar spam so the published log is readable.
tr '\r' '\n' < /workspace/logs/train.log \
  | grep -vE "Loading weights:|Fetching [0-9]+ files:|examples/s\]|it/s\]$" \
  > "$OUT/training.log"

# peft prints the trainable-parameter count once; keep it as a first-class number.
grep -oE "trainable params: [0-9,]+ \|\| all params: [0-9,]+ \|\| trainable%: [0-9.]+" \
  "$OUT/training.log" | tail -1 > "$OUT/trainable_params.txt" || true

{
  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo
  echo "=== peak memory during run (from log, if reported) ==="
  grep -oE "max_memory[^,]*" "$OUT/training.log" | tail -1 || echo "(not reported)"
  echo
  echo "=== packages ==="
  python3 - <<'PY'
import torch, transformers, trl, peft, datasets, accelerate
for m in (torch, transformers, trl, peft, datasets, accelerate):
    print(f"{m.__name__:14} {m.__version__}")
PY
  echo
  echo "=== fast-path note ==="
  echo "causal-conv1d / flash-linear-attention are NOT installed; transformers falls back to"
  echo "the torch implementation. Expected - they only matter on the training/merge path and"
  echo "fail to build without a long detour."
} > "$OUT/environment.txt"

# Convert the yaml config to json so the published artifact is machine-readable.
python3 - <<'PY'
import json, yaml, pathlib
p = pathlib.Path("/workspace/artifacts/resolved_config.json.yaml")
cfg = yaml.safe_load(p.read_text())
pathlib.Path("/workspace/artifacts/resolved_config.json").write_text(json.dumps(cfg, indent=2))
p.unlink()
PY

echo "=== collected ==="
ls -la "$OUT"
echo "=== adapter ==="
ls -la "$RUN/adapter"
echo "ADAPTER_DIR=$RUN/adapter"
