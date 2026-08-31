#!/usr/bin/env bash
# ABOUTME: Rsync the repo to a RunPod 2xH200 box, uv sync, and launch a DDP LoRA run.
# ABOUTME: Usage: train_on_pod.sh <pod_id> <config.yaml> [nproc]
set -euo pipefail
POD="$1"; CFG="$2"; NPROC="${3:-2}"
REPO="/home/matthewb/git repos/teaching_claude_why_replication"
cd "$REPO"
set -a; source .env; set +a

read -r HOST PORT < <(uv run python scratch/runpod_train_2xh200.py ssh --pod "$POD" 2>/dev/null \
  | sed -n 's/^HOST=\(.*\) PORT=\(.*\)$/\1 \2/p')
[ -n "${HOST:-}" ] && [ -n "${PORT:-}" ] || { echo "no ssh endpoint for $POD"; exit 1; }
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ServerAliveInterval=30 "root@$HOST")
echo "=== $POD  $HOST:$PORT  config $CFG ==="

"${SSH[@]}" 'mkdir -p /workspace/repo' 
rsync -az --delete -e "ssh -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude '.git' --exclude '.venv' --exclude 'output' --exclude 'data' \
  --exclude 'node_modules' --exclude 'dashboard' --exclude 'third_party' \
  --exclude '__pycache__' --exclude '*.jsonl' \
  ./ "root@$HOST:/workspace/repo/"
scp -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  .env "root@$HOST:/workspace/repo/.env"
echo "  repo + .env synced"

# uv sync is slow and must finish before the trainer starts; run it in the foreground.
"${SSH[@]}" 'cd /workspace/repo && (command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh) >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH; uv sync --frozen 2>&1 | tail -3'
echo "  uv sync done"

# nohup + </dev/null so the SSH channel closes instead of hanging on the background job.
"${SSH[@]}" "cd /workspace/repo && export PATH=\$HOME/.local/bin:\$PATH && \
  mkdir -p /workspace/logs && \
  uv pip install -q hf_transfer 2>/dev/null; export HF_HUB_ENABLE_HF_TRANSFER=1 && \
  nohup env HF_HUB_ENABLE_HF_TRANSFER=1 uv run torchrun --nproc_per_node=$NPROC scripts/train/train_lora.py --config $CFG \
  > /workspace/logs/train.log 2>&1 </dev/null & echo launched pid \$!"
echo "  training launched; poll with: ssh -p $PORT root@$HOST 'tail -f /workspace/logs/train.log'"
