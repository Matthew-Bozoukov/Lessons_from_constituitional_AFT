# ABOUTME: Run ON a `runpod up --eval ... --clone-repo` pod: fetch upstream + local patch, then start the
# ABOUTME: code_review driver for one target under nohup (vLLM served locally by VllmServer).
# Run: ssh <pod> 'cd /root/work && bash scratch/iai_code_review/pod_launch.sh <hf target> <out dir> [epochs]'
set -euo pipefail
TARGET=$1
OUT=$2
EPOCHS=${3:-100}
MODEL_A=${4:-openrouter/anthropic/claude-sonnet-4.6}
cd /root/work
export HF_HOME=/workspace/hf
bash scratch/iai_code_review/fetch_upstream.sh
mkdir -p "$OUT"
nohup uv run python scratch/iai_code_review/run_code_review.py \
  --upstream output/iai_code_review/upstream --env-file .env \
  --model-a "$MODEL_A" --target "$TARGET" --capability baseline persuasion deception --epochs "$EPOCHS" \
  --out "$OUT" > "$OUT.log" 2>&1 </dev/null &
echo "started pid $! -> $OUT.log"
