#!/usr/bin/env bash
# ABOUTME: Launch the SURF fabrication sweep against a live pod under systemd-run, so it
# ABOUTME: survives the terminal/session that started it.
#
# Run: bash scratch/launch_surf_fabrication.sh <pod-id>
#
# systemd-run --user, not nohup: a plain background process is still in the session's
# cgroup and dies when the session is torn down. This sweep runs ~2.5h, so it has to
# outlive the shell that starts it.

set -euo pipefail

POD="${1:?usage: launch_surf_fabrication.sh <pod-id>}"
ROOT="/home/matthewb/git repos/teaching_claude_why_replication"
SURF="$ROOT/src/eval/audits/surf/third_party/SURF"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/output/surf/${STAMP}_t2synth_fabrication"
LOG="$OUT/sweep.log"

mkdir -p "$OUT"

# Preflight, because SURF does not fail on a dead target: it logs "Target model error" per
# candidate, scores nothing, and still exits 0 with "SWEEP COMPLETE". One run burned 3,111
# query-generation calls against a pod whose vLLM was mid-restart and reported success.
echo "preflight: $POD"
uv run python "$ROOT/scratch/surf_preflight.py" "$POD"

systemd-run --user --unit="surf-fab-${STAMP}" --working-directory="$SURF" \
  --setenv=PYTHONUNBUFFERED=1 \
  -- bash -lc "uv run -m surf.cli.main sweep \
      --attributes seoirsem/CHUNKY-tulu3-SFT-25k-attributes \
      --rubric rubrics/fabrication.yaml \
      --output-dir '$OUT' \
      --num-runs 3 --iterations 15 --candidates 120 \
      --target-model 'https://${POD}-8000.proxy.runpod.net/v1:t2synth' \
      --judge-model openrouter:openai/gpt-5.6-terra \
      --query-model openrouter:meta-llama/llama-3.1-70b-instruct \
      > '$LOG' 2>&1"

echo "unit:    surf-fab-${STAMP}"
echo "out:     $OUT"
echo "log:     $LOG"
echo "follow:  tail -f '$LOG'"
echo "stop:    systemctl --user stop surf-fab-${STAMP}"
