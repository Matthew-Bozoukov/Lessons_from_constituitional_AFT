#!/usr/bin/env bash
# ABOUTME: Installs the pinned training stack and downloads Qwen3.6-27B onto the SFT pod.
# ABOUTME: Training only - no vLLM here, so none of the vLLM/CUDA-13 constraints apply.
#
# WHY THESE PINS
#
# transformers 5.14.1 / trl 1.9.2 / peft 0.20.0 / datasets 5.0.1 is the exact set that
# produced the three published Qwen3.6-27B adapters. transformers below 5.x does not know
# Qwen3_5ForConditionalGeneration at all.
#
# torch is CONSTRAINED to whatever the image already ships. Letting pip resolve torch
# freely risks a 2GB download that swaps the cu128 build for one that does not match the
# host driver, which would fail at the first .cuda() call after ~10 minutes of install.

set -euo pipefail

WORK=/workspace
mkdir -p "$WORK/logs" "$WORK/models"

export HF_HUB_ENABLE_HF_TRANSFER=1
# Qwen/Qwen3.6-27B is public. Downloading anonymously means no credential is ever written
# to a rented machine's disk, which is worth more than the marginal convenience of a token.
unset HF_TOKEN 2>/dev/null || true

echo "=== [1/4] pinning torch to the image's build ==="
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "torch already installed: $TORCH_VER"
echo "torch==$TORCH_VER" > /tmp/constraints.txt

echo "=== [2/4] installing the training stack ==="
pip install --quiet --upgrade pip
PIP_CONSTRAINT=/tmp/constraints.txt pip install --quiet \
  "transformers==5.14.1" \
  "trl==1.9.2" \
  "peft==0.20.0" \
  "datasets==5.0.1" \
  "accelerate" \
  "omegaconf" \
  "fire" \
  "huggingface_hub[hf_transfer]>=0.34" \
  "wandb"

python3 - <<'PY'
import torch, transformers, trl, peft, datasets
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("peft        ", peft.__version__)
print("datasets    ", datasets.__version__)
assert torch.cuda.is_available(), "CUDA disappeared - pip probably replaced torch"
assert torch.cuda.is_bf16_supported(), "bf16 unsupported on this device"
print("cuda ok:", torch.cuda.get_device_name(0))
PY

echo "=== [3/4] downloading Qwen/Qwen3.6-27B (~54GB) ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3.6-27B", local_dir="/workspace/models/base", max_workers=16,
    ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
)
PY

echo "=== [4/4] verifying what arrived ==="
python3 - <<'PY'
import json, pathlib, sys
base = pathlib.Path("/workspace/models/base")
shards = sorted(base.glob("*.safetensors"))
gb = sum(p.stat().st_size for p in shards) / 1e9
cfg = json.loads((base / "config.json").read_text())
print(f"shards: {len(shards)}, {gb:.1f} GB")
print("architecture:", cfg.get("architectures", ["?"])[0])
if gb < 45:
    print(f"FAIL: only {gb:.1f} GB - download truncated"); sys.exit(1)
print("base model ok")
PY

echo "=== BOOTSTRAP COMPLETE ==="
