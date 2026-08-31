#!/usr/bin/env bash
# ABOUTME: Drives the SURF traits sweep against a RunPod-served Qwen3.6-27B LoRA arm,
# ABOUTME: teeing all output to a timestamped log. Run: bash scratch/run_surf_sweep.sh <pod-id> [smoke]
#
# The EM loop, query model and judge all run here (the OpenRouter key is machine-local);
# only the target lives on the pod, reached over its public proxy endpoint.

set -euo pipefail

POD="${1:?usage: run_surf_sweep.sh <pod-id> [smoke]}"
MODE="${2:-full}"

REPO="/home/matthewb/git repos/teaching_claude_why_replication"
SURF="$REPO/src/eval/audits/surf/third_party/SURF"
TS="$(date +%Y%m%d_%H%M%S)"
TARGET="https://${POD}-8000.proxy.runpod.net/v1:t2synth"

if [ "$MODE" = "smoke" ]; then
  RUNS=1; ITERS=2; CANDIDATES=20; TAG="smoke"
else
  RUNS=3; ITERS=20; CANDIDATES=120; TAG="t2synth_traits"
fi

OUT="$REPO/output/surf/${TS}_${TAG}"
mkdir -p "$OUT"
LOG="$OUT/sweep.log"

echo "target:      $TARGET"
echo "output:      $OUT"
echo "scale:       ${RUNS} runs x ${ITERS} iterations x ${CANDIDATES} candidates"
echo "judge:       openrouter:openai/gpt-5.6-terra"
echo "log:         $LOG"

cd "$SURF"
uv run -m surf.cli.main sweep \
  --attributes seoirsem/CHUNKY-tulu3-SFT-25k-attributes \
  --rubric rubrics/traits.yaml \
  --output-dir "$OUT" \
  --num-runs "$RUNS" \
  --iterations "$ITERS" \
  --candidates "$CANDIDATES" \
  --target-model "$TARGET" \
  --judge-model "openrouter:openai/gpt-5.6-terra" \
  --query-model "openrouter:meta-llama/llama-3.1-70b-instruct" \
  2>&1 | tee "$LOG"

echo "=== judge errors (nonzero means the run has holes) ==="
grep -c '!!! judge error' "$LOG" || true
