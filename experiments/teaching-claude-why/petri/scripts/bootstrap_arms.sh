#!/usr/bin/env bash
# ABOUTME: Installs vLLM and downloads Qwen3.6-27B plus the three LoRA arms onto the rented box.
# ABOUTME: Does NOT reuse the sibling experiment's bootstrap, which pins vLLM 0.11.0 - too old for this arch.
#
# WHY A SEPARATE BOOTSTRAP
#
# vulnerabilities/scripts/remote/bootstrap.sh pins vllm==0.11.0, which predates
# support for Qwen3.6-27B's hybrid Mamba/linear-attention stack. The sibling
# ODCV run established that this model needs vLLM 0.26. Reusing that bootstrap
# would fail at model load, after paying for a 54GB download.
#
# Also note, from the same run: causal-conv1d and flash-linear-attention are
# IRRELEVANT here. vLLM uses its own kernels (FlashInfer GDN prefill,
# vllm::mamba_mixer2, vllm::qwen_gdn_attention_core). Those libraries only matter
# on the transformers path (training/merging), where their absence cost 44s/step.
# They also fail to build, because causal-conv1d's setup.py guesses a wheel URL
# for the wrong tags. Do not waste an hour on them.

set -euo pipefail

WORK=/workspace
MODELS="$WORK/models"
mkdir -p "$MODELS" "$WORK/logs"

export HF_HUB_ENABLE_HF_TRANSFER=1
: "${HF_TOKEN:?HF_TOKEN must be injected into this process}"

echo "=== [1/4] python / cuda baseline ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [2/4] installing vLLM (>=0.26 required for this architecture) ==="
pip install --quiet --upgrade pip
pip install --quiet "vllm>=0.26" "huggingface_hub[hf_transfer]>=0.34" || {
  echo "!! vllm>=0.26 failed to install" >&2; exit 1; }
python3 - <<'PY'
import vllm, torch, transformers
from packaging.version import Version
print("vllm", vllm.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__)
assert Version(vllm.__version__) >= Version("0.26"), (
    f"vllm {vllm.__version__} is too old for Qwen3.6-27B's hybrid Mamba/linear-attention stack"
)
PY

echo "=== [3/4] downloading base + three arms ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download
import os

base = "Qwen/Qwen3.6-27B"
print(f"--- base: {base} (~54GB) ---", flush=True)
snapshot_download(
    base, local_dir="/workspace/models/base", max_workers=16,
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
)

arms = {
    "dose-10-90": "LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-10-90",
    "dose-20-80": "LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-20-80",
    "dose-40-60": "LASR-Callum/qwen3.6-27b-difficult-advice-tulu-lora-40-60",
}
for name, repo in arms.items():
    print(f"--- {name}: {repo} ---", flush=True)
    snapshot_download(
        repo, local_dir=f"/workspace/models/{name}", max_workers=8,
        token=os.environ["HF_TOKEN"],
    )
PY

echo "=== [4/4] verifying what arrived ==="
python3 - <<'PY'
import json, pathlib, hashlib, sys

problems = []
base = pathlib.Path("/workspace/models/base")
shards = sorted(base.glob("*.safetensors"))
gb = sum(p.stat().st_size for p in shards) / 1e9
print(f"base: {len(shards)} shards, {gb:.1f} GB")
if gb < 45:
    problems.append(f"base weights only {gb:.1f} GB - expected ~54 GB, download likely truncated")

cfg = json.loads((base / "config.json").read_text())
arch = cfg.get("architectures", ["?"])[0]
print(f"base architecture: {arch}")

# The chat template is the landmine: an adapter template without tool support
# makes vLLM reject every tool-bearing request with HTTP 400, which silently
# produced 30 empty transcripts in the sibling experiment's pilot v1. Verified
# identical against Hugging Face before provisioning; re-verified here on the
# bytes that actually landed.
base_tpl = json.loads((base / "tokenizer_config.json").read_text())["chat_template"]
base_sha = hashlib.sha256(base_tpl.encode()).hexdigest()
print(f"base chat_template: {len(base_tpl)} chars, tool refs {base_tpl.count('tools')}, sha {base_sha[:16]}")
if base_tpl.count("tools") == 0:
    problems.append("base chat template has NO tool support - every audit would 400")

for name in ("dose-10-90", "dose-20-80", "dose-40-60"):
    d = pathlib.Path("/workspace/models") / name
    ac = json.loads((d / "adapter_config.json").read_text())
    r = ac.get("r")
    tpl = (d / "chat_template.jinja").read_text(encoding="utf-8")
    sha = hashlib.sha256(tpl.encode()).hexdigest()
    same = sha == base_sha
    size_mb = (d / "adapter_model.safetensors").stat().st_size / 1e6
    print(f"{name}: r={r}  adapter {size_mb:.0f} MB  template {'IDENTICAL to base' if same else 'DIFFERS'}")
    if r != 32:
        problems.append(f"{name}: adapter rank {r} != 32, --max-lora-rank must be raised")
    if not same:
        problems.append(f"{name}: chat template differs from base - check tool support before serving")
    if tpl.count("tools") == 0:
        problems.append(f"{name}: chat template has NO tool support - audits would 400")

if problems:
    print("\nFAIL:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("\nall arms present and consistent")
PY

echo "=== bootstrap complete. Next: bash scripts/serve_arms.sh ==="
