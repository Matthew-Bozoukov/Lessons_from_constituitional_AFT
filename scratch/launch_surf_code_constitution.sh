#!/usr/bin/env bash
# ABOUTME: Launch the constitution-in-code SURF sweep under systemd-run, after preflighting
# ABOUTME: the target endpoint so a dead pod fails here rather than 2,400 candidates later.
#
# Run: bash scratch/launch_surf_code_constitution.sh <pod-id> [arm]
#
# systemd-run --user, not nohup: a background process stays in the session cgroup and dies
# with the session. This sweep runs ~1-2h.

set -euo pipefail

POD="${1:?usage: launch_surf_code_constitution.sh <pod-id> [arm]}"
ARM="${2:-t2synth716}"
ROOT="/home/matthewb/git repos/teaching_claude_why_replication"
SURF="$ROOT/src/eval/audits/surf/third_party/SURF"
ATTRS="$ROOT/output/surf_attrs_opencodeinstruct/code_attributes.jsonl"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/output/surf/${STAMP}_${ARM}_code_constitution"
LOG="$OUT/sweep.log"

test -s "$ATTRS" || { echo "missing attribute corpus: $ATTRS" >&2; exit 1; }
mkdir -p "$OUT"

# SURF logs "Target model error" per candidate and still exits 0 with SWEEP COMPLETE on a
# dead target; one run burned 3,111 query-generation calls that way.
echo "preflight: $POD / $ARM"
uv run python "$ROOT/scratch/surf_preflight.py" "$POD" 3 || {
  echo "preflight failed - not launching" >&2; exit 1; }

systemd-run --user --unit="surf-code-${STAMP}" --working-directory="$SURF" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?export the key before launching}" \
  -- bash -lc "uv run -m surf.cli.main sweep \
      --attributes '$ATTRS' \
      --rubric rubrics/constitution_in_code.yaml \
      --output-dir '$OUT' \
      --num-runs 2 --iterations 10 --candidates 120 \
      --target-model 'https://${POD}-8000.proxy.runpod.net/v1:${ARM}' \
      --query-model openrouter:google/gemini-3.7-flash \
      --judge-model openrouter:google/gemini-3.7-flash \
      > '$LOG' 2>&1"

echo "unit:   surf-code-${STAMP}"
echo "out:    $OUT"
echo "follow: tail -f '$LOG'"
echo "stop:   systemctl --user stop surf-code-${STAMP}"
