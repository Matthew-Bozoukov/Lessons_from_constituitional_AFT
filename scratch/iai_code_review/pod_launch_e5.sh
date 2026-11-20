# ABOUTME: Run ON a `runpod up --eval ... --clone-repo` pod: fetch upstream + W2S repo, then start the
# ABOUTME: alignment_research (E5) driver for one target under nohup, pushing logs to a private Hub repo as it goes.
# Run: ssh <pod> 'cd /root/work && bash scratch/iai_code_review/pod_launch_e5.sh <hf target> <out dir> <push subject> <epochs> [conditions...]'
set -euo pipefail
TARGET=$1
OUT=$2
SUBJECT=$3
EPOCHS=$4
shift 4
CONDITIONS=${*:-baseline persuasion_L1 persuasion_L3}
cd /root/work
export HF_HOME=/workspace/hf
bash scratch/iai_code_review/fetch_upstream.sh
mkdir -p "$OUT"
nohup uv run python scratch/iai_code_review/run_code_review.py --env alignment_research \
  --upstream output/iai_code_review/upstream --env-file .env \
  --model-a openrouter/x-ai/grok-4.20 --grader-model openrouter/google/gemini-3-flash-preview \
  --target "$TARGET" --capability $CONDITIONS --epochs "$EPOCHS" --max-turns 16 \
  --context-window 131072 --out "$OUT" --push-subject "$SUBJECT" > "$OUT.log" 2>&1 </dev/null &
echo "started pid $! -> $OUT.log (conditions: $CONDITIONS; pushes to <date>-$SUBJECT)"
