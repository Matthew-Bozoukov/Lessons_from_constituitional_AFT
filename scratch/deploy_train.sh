#!/usr/bin/env bash
# ABOUTME: Stages the repo's training code onto a prepared pod and launches 2-rank DDP
# ABOUTME: training detached. Run: bash scratch/deploy_train.sh <pod-id> <config-path>

set -euo pipefail

POD="${1:?usage: deploy_train.sh <pod-id> <config-path>}"
CONFIG="${2:?config path required}"
NPROC="${3:-2}"
REPO="/home/matthewb/git repos/teaching_claude_why_replication"

read -r IP PORT < <(cd "$REPO" && uv run python -c "
import sys; sys.path.insert(0,'.')
from src.eval.misalignment.internalization.scripts.runpod import call
p=call('GET','/pods/$POD')
print(p['publicIp'], p['portMappings']['22'])
")
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT"

echo "pod $POD  ssh $IP:$PORT  config=$CONFIG  nproc=$NPROC"

# Only the code the trainer imports — not data (the pod fetched the mixture from HF itself)
# and not output/, which is large and irrelevant on the pod.
cd "$REPO"
# Exclusions matter: src/ is ~7.9GB, of which src/eval/audits/surf/third_party is 7.2GB
# (a vendored harness plus its own venv and caches). Without these the tarball is 133MB
# compressed and expands to 7.9GB on the pod, which delayed one launch by several minutes
# for code the trainer never imports.
tar -czf /tmp/train_code.tar.gz \
  --exclude='*/third_party/*' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.venv' --exclude='*.jsonl' --exclude='*.parquet' \
  src scripts configs pyproject.toml
$SCP /tmp/train_code.tar.gz "root@$IP:/workspace/train_code.tar.gz"
$SSH 'cd /workspace/repo && tar -xzf /workspace/train_code.tar.gz && ls -d src scripts configs'

# setsid: the run must outlive this SSH connection. PYTHONPATH: the repo is not pip-installed
# on the pod, so `import src.*` resolves via the working directory only if it is on the path.
# W&B key comes from the CONTAINER's env (set on the pod at creation), read out of
# /proc/1/environ rather than passed on this command line -- an ssh argument would land in
# the pod's process list and shell history. WANDB_MODE falls back to disabled if no key is
# present, so a keyless pod still trains instead of erroring at the first log step.
$SSH "cd /workspace/repo && \
  export \$(tr '\0' '\n' < /proc/1/environ | grep -E '^WANDB_API_KEY=' | head -1) 2>/dev/null; \
  export WANDB_MODE=\$([ -n \"\${WANDB_API_KEY:-}\" ] && echo online || echo disabled); \
  export WANDB_PROJECT=\"\${WANDB_PROJECT:-teaching-claude-why}\"; \
  echo \"wandb: \$WANDB_MODE project=\$WANDB_PROJECT\"; \
  HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup torchrun --nproc_per_node=$NPROC scripts/train/train_lora.py \
    --config $CONFIG </dev/null > /workspace/train.log 2>&1 & \
  sleep 25; echo '--- first log lines ---'; head -20 /workspace/train.log"

echo
echo "poll:  ssh -p $PORT root@$IP 'tail -5 /workspace/train.log'"
