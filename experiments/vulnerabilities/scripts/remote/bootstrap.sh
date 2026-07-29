#!/usr/bin/env bash
# Remote bootstrap for the MSM audit target server.
#
# Pinned revisions - every download is by explicit commit, never by branch, so
# the served weights are reproducible and hashable.
#
# HF_TOKEN arrives via the environment (injected by the local infra wrapper into
# this process only). It is never written to disk here.
set -uo pipefail

BASE_REPO="Qwen/Qwen3-32B"
BASE_REV="9216db5781bf21249d130ec9da846c4624c16137"
ADAPTER_REPO="chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot"
ADAPTER_REV="9a00c85c80d195c6153a56373e6901413ba6f519"

WORK=/workspace
LOG="$WORK/bootstrap.log"
mkdir -p "$WORK/models" "$WORK/logs"

# The HF token is delivered as a chmod-600 file written over stdin, never as a
# command-line argument (argv is visible in the remote process listing). It is
# shredded as soon as the downloads finish.
if [ -f "$WORK/.hfenv" ]; then
  # shellcheck disable=SC1091
  . "$WORK/.hfenv"
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap start $(date -u +%FT%TZ) ==="

echo "--- [1/5] environment ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version
pip --version

echo "--- [2/5] installing vllm ---"
pip install --quiet --upgrade pip
# vLLM brings its own pinned torch; let it resolve.
pip install --quiet "vllm==0.11.0" "huggingface_hub[hf_transfer]>=0.34" || {
  echo "!! pinned vllm failed; falling back to latest"
  pip install --quiet vllm "huggingface_hub[hf_transfer]>=0.34"
}
# vLLM 0.11.0 calls tokenizer.all_special_tokens_extended, which transformers 5.x
# removed. Without this pin the server dies on startup with an AttributeError.
# Pinning here rather than applying it by hand, so a rebuild cannot lose it.
pip install --quiet "transformers>=4.56,<5"
python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__); print('torch', torch.__version__); print('cuda', torch.version.cuda); print('transformers', transformers.__version__)"

echo "--- [3/5] downloading base ($BASE_REPO @ $BASE_REV) ---"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$WORK/hf"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="${BASE_REPO}", revision="${BASE_REV}",
    local_dir="${WORK}/models/base", max_workers=16,
    token=os.environ.get("HF_TOKEN"),
)
print("BASE_PATH", p)
PY

echo "--- [4/5] downloading adapter ($ADAPTER_REPO @ $ADAPTER_REV) ---"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="${ADAPTER_REPO}", revision="${ADAPTER_REV}",
    local_dir="${WORK}/models/adapter", max_workers=8,
    token=os.environ.get("HF_TOKEN"),
)
print("ADAPTER_PATH", p)
PY

echo "--- [4b/5] downloading matched control adapters ---"
python3 - <<'PY'
import os, json, hashlib
from huggingface_hub import snapshot_download
CONTROLS = {
 "msm-aft-no-cot": ("chloeli/qwen-3-32b-philosophy-spec-msm-aft-no-cot","1385a422577fac66b7c34d6fd6d3d3c3fa24f60d"),
 "aft-cot":        ("chloeli/qwen-3-32b-philosophy-spec-aft-cot","264efa3ed44a8eb131825cf53740d17e2fc3c33f"),
 "aft-no-cot":     ("chloeli/qwen-3-32b-philosophy-spec-aft-no-cot","548c8fb05d4e6ffe936a8ae4bc934dc206e81e6e"),
 "msm-only":       ("chloeli/qwen-3-32b-philosophy-spec-msm","17a315f4620f09da6bfb852c53191966d8a1f66f"),
 "id-baseline":    ("chloeli/qwen-3-32b-id-baseline","fee84084ebbc41d7670b115581db762451e0220e"),
}
out={}
for name,(repo,rev) in CONTROLS.items():
    p = snapshot_download(repo_id=repo, revision=rev,
                          local_dir=f"/workspace/models/controls/{name}",
                          max_workers=4, token=os.environ.get("HF_TOKEN"))
    w = os.path.join(p,"adapter_model.safetensors")
    h = hashlib.sha256()
    with open(w,"rb") as fh:
        for c in iter(lambda: fh.read(1<<20), b""): h.update(c)
    cfg = json.load(open(os.path.join(p,"adapter_config.json")))
    out[name] = {"repo":repo,"revision":rev,"sha256":h.hexdigest(),
                 "r":cfg.get("r"),"alpha":cfg.get("lora_alpha"),
                 "base":cfg.get("base_model_name_or_path")}
    print(name,"OK",out[name]["sha256"][:16],"r=",cfg.get("r"),"alpha=",cfg.get("lora_alpha"))
open("/workspace/logs/control-adapters.json","w").write(json.dumps(out,indent=2))
PY

echo "--- token no longer needed; shredding ---"
if [ -f "$WORK/.hfenv" ]; then
  shred -u "$WORK/.hfenv" 2>/dev/null || rm -f "$WORK/.hfenv"
fi
unset HF_TOKEN

echo "--- [5/5] hashing artifacts ---"
python3 - <<'PY'
import hashlib, json, os, pathlib
out = {}
targets = {
    "adapter_model.safetensors": "/workspace/models/adapter/adapter_model.safetensors",
    "adapter_config.json":       "/workspace/models/adapter/adapter_config.json",
    "adapter_tokenizer.json":    "/workspace/models/adapter/tokenizer.json",
    "adapter_tokenizer_config":  "/workspace/models/adapter/tokenizer_config.json",
    "adapter_chat_template":     "/workspace/models/adapter/chat_template.jinja",
    "base_tokenizer.json":       "/workspace/models/base/tokenizer.json",
    "base_tokenizer_config":     "/workspace/models/base/tokenizer_config.json",
    "base_config.json":          "/workspace/models/base/config.json",
}
for name, path in targets.items():
    if not os.path.exists(path):
        out[name] = {"present": False}
        continue
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out[name] = {"present": True, "sha256": h.hexdigest(), "bytes": os.path.getsize(path)}
# every base shard, so the served weights are fully pinned
shards = sorted(pathlib.Path("/workspace/models/base").glob("*.safetensors"))
out["base_shards"] = {"count": len(shards),
                      "total_bytes": sum(p.stat().st_size for p in shards)}
print(json.dumps(out, indent=2))
open("/workspace/logs/artifact-hashes.json", "w").write(json.dumps(out, indent=2))
PY

echo "=== bootstrap complete $(date -u +%FT%TZ) ==="
touch "$WORK/BOOTSTRAP_DONE"
