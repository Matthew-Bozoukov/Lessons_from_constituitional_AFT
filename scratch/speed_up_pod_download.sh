#!/usr/bin/env bash
# ABOUTME: Restart a pod's trainer with hf_transfer enabled, to fix single-stream model download.
# ABOUTME: Usage: speed_up_pod_download.sh <pod_id> <config.yaml>
set -euo pipefail
POD="$1"; CFG="$2"
cd "/home/matthewb/git repos/teaching_claude_why_replication"
set -a; source .env; set +a
read -r H P < <(uv run python scratch/runpod_train_2xh200.py ssh --pod "$POD" 2>/dev/null \
  | sed -n 's/^HOST=\(.*\) PORT=\(.*\)$/\1 \2/p')
SSH=(ssh -p "$P" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o ServerAliveInterval=30 "root@$H")
echo "=== $POD ==="
# Stop the trainer; the partially fetched blobs stay on disk and HF resumes from them.
"${SSH[@]}" 'pkill -f "[t]rain_lora.py" || true; pkill -f "[t]orchrun" || true; sleep 4; echo "  stopped: $(pgrep -fc "[t]rain_lora.py" || echo 0) trainer procs left"'
"${SSH[@]}" 'cd /workspace/repo && export PATH=$HOME/.local/bin:$PATH && uv pip install -q hf_transfer 2>&1 | tail -2; uv run python -c "import hf_transfer; print(\"  hf_transfer\", hf_transfer.__version__)"'
"${SSH[@]}" "cd /workspace/repo && export PATH=\$HOME/.local/bin:\$PATH && \
  export HF_HUB_ENABLE_HF_TRANSFER=1 && \
  mv /workspace/logs/train.log /workspace/logs/train.prev.log 2>/dev/null || true && \
  nohup uv run torchrun --nproc_per_node=2 scripts/train/train_lora.py --config $CFG \
  > /workspace/logs/train.log 2>&1 </dev/null & echo '  relaunched with hf_transfer'"
