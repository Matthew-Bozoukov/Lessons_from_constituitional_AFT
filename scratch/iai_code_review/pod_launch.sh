# ABOUTME: Run ON a `runpod up --eval ... --clone-repo` pod: fetch upstream + local patches, then start the
# ABOUTME: code_review driver for one target under nohup, pushing its logs to a private Hub repo as it goes.
# Run: ssh <pod> 'cd /root/work && bash scratch/iai_code_review/pod_launch.sh <hf target> <out dir> <push subject> [epochs] [model_a] [conditions...]'
set -euo pipefail
TARGET=$1
OUT=$2
SUBJECT=$3
EPOCHS=${4:-100}
MODEL_A=${5:-openrouter/google/gemini-3-flash-preview}
shift 5 || shift $#
CONDITIONS=${*:-baseline persuasion deception}
cd /root/work
export HF_HOME=/workspace/hf
bash scratch/iai_code_review/fetch_upstream.sh
mkdir -p "$OUT"
nohup uv run python scratch/iai_code_review/run_code_review.py \
  --upstream output/iai_code_review/upstream --env-file .env \
  --model-a "$MODEL_A" --target "$TARGET" --capability $CONDITIONS --epochs "$EPOCHS" \
  --out "$OUT" --push-subject "$SUBJECT" > "$OUT.log" 2>&1 </dev/null &
echo "started pid $! -> $OUT.log (conditions: $CONDITIONS; pushes to <date>-$SUBJECT)"
