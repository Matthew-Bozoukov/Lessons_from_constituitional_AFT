#!/usr/bin/env bash
# ABOUTME: Prepares one 2xH200 pod for a numina-control seed replicate: deps, code, .env,
# ABOUTME: base weights, then a --smoke DDP run. Does NOT start real training.
#
# Usage: prep_seed_pods.sh <pod-id> <config-path>
#
# Stops after the smoke run on purpose. The smoke exercises the SAME torchrun path as the
# real run -- DDP init, the mask gate, the device map -- so a green smoke is what makes the
# go/no-go decision meaningful instead of a guess.
set -euo pipefail

POD="${1:?usage: prep_seed_pods.sh <pod-id> <config-path>}"
CONFIG="${2:?config path required}"
REPO="/home/matthewb/git repos/teaching_claude_why_replication"
BASE="Qwen/Qwen3.6-27B"

cd "$REPO"
read -r IP PORT < <(uv run python -c "
import sys; sys.path.insert(0,'.')
from src.eval.misalignment.internalization.scripts.runpod import call
p=call('GET','/pods/$POD')
print(p['publicIp'], p['portMappings']['22'])
")
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT"
echo "=== $POD  $IP:$PORT  config=$CONFIG"

# Same exclusions as deploy_train.sh: src/ carries a 7.2GB vendored harness the trainer
# never imports, and shipping it delayed an earlier launch by several minutes.
tar -czf /tmp/prep_code_$POD.tar.gz \
  --exclude='*/third_party/*' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.venv' --exclude='*.jsonl' --exclude='*.parquet' \
  src scripts configs pyproject.toml
$SSH 'mkdir -p /workspace/repo /workspace/hf'
$SCP /tmp/prep_code_$POD.tar.gz "root@$IP:/workspace/code.tar.gz"
# .env carries HF_TOKEN: the pod fetches the pinned mixture itself and pushes its own adapter.
$SCP .env "root@$IP:/workspace/repo/.env"
$SSH 'cd /workspace/repo && tar -xzf /workspace/code.tar.gz && ls -d src scripts configs'

echo "=== installing deps + base weights ($POD)"
$SSH "set -e
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/hf
set -a; . /workspace/repo/.env; set +a
# python-dotenv is NOT optional: src/huggingface.py imports it to read the token, so
# without it every rank dies at hf_api() before the first step.
python3 -m pip install --no-cache-dir -q 'transformers>=5.14' 'trl>=0.27' 'peft==0.20.0' \
  datasets accelerate omegaconf fire huggingface_hub hf_transfer wandb python-dotenv
hf download $BASE >/dev/null
echo DEPS_OK"

echo "=== smoke ($POD)"
$SSH "cd /workspace/repo && set -e
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
set -a; . /workspace/repo/.env; set +a
set -o pipefail
torchrun --nproc_per_node=2 scripts/train/train_lora.py --config $CONFIG --smoke 2>&1 | tail -45
echo SMOKE_OK"
