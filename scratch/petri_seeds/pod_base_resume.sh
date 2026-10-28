#!/usr/bin/env bash
# ABOUTME: Runs ON the GPU pod, unattended: resumes the no-constitution base Qwen3.6-27B Petri audit from
# ABOUTME: the shipped .eval log, backs the raw log up to HF every 10 min, stops on no credit, pushes, self-deletes.
# Run (on the pod, detached): nohup bash scratch/petri_seeds/pod_base_resume.sh <retry.eval> [conn] </dev/null &
set -uo pipefail
cd /root/work
export PATH=/usr/local/bin:/root/.local/bin:$PATH HF_HOME=/workspace/hf
# Serving on the box (LocalExec): without this vLLM's engine dies JIT-compiling flashinfer's sampler
# with an nvcc the image lacks (docs/GOTCHAS.md 2026-09-21).
export VLLM_USE_FLASHINFER_SAMPLER=0
unset PETRI_SYSTEM_PREAMBLE_FILE
set -a; . ./.env; set +a
# An SSH-launched shell does not inherit the container's pod id and pod-scoped key; read them from PID 1.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
[ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || { echo "!!! no RUNPOD_POD_ID/RUNPOD_API_KEY: cannot self-delete"; exit 1; }
RETRY="$1"
CONN="${2:-36}"
TARGET="Qwen/Qwen3.6-27B"
REPO="2026-10-15-petri-qwen36"
RUN_DIR="$(dirname "$(dirname "$RETRY")")"
DEADLINE_S=${DEADLINE_S:-21600}   # 6 h: the pod deletes itself by then whatever happens
mkdir -p output/logs
LOG=output/logs/pod_base_resume.log
exec > >(tee -a "$LOG") 2>&1
echo ">>> start $(date -u +%FT%TZ) pod=$RUNPOD_POD_ID retry=$RETRY conn=$CONN deadline=${DEADLINE_S}s"

self_delete() {
  echo ">>> deleting pod ${RUNPOD_POD_ID} at $(date -u +%FT%TZ): $1"
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" -o /dev/null -w "    delete HTTP %{http_code}\n"
}
( sleep "$DEADLINE_S"; self_delete "deadline ${DEADLINE_S}s reached" ) &

backup_logs() {   # raw .eval logs + this log to HF under backup/, so a dead pod loses at most 10 min
  .venv/bin/python - "$RUN_DIR" "$REPO" "$LOG" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
run_dir, repo, log = sys.argv[1:]
a = HfApi(token=os.environ["HF_TOKEN"]); r = f"{os.environ['HF_ORG']}/{repo}"
a.create_repo(r, repo_type="dataset", exist_ok=True)
for p in sorted(Path(run_dir, "logs").glob("*.eval")) + [Path(log)]:
    a.upload_file(path_or_fileobj=str(p), path_in_repo=f"backup/{p.name}", repo_id=r, repo_type="dataset")
print(f">>> backup: {len(list(Path(run_dir, 'logs').glob('*.eval')))} .eval + driver log -> {r}/backup/", flush=True)
PY
}

credit_left() {
  .venv/bin/python -c "
import json, os, urllib.request
req = urllib.request.Request('https://openrouter.ai/api/v1/credits', headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']})
d = json.load(urllib.request.urlopen(req, timeout=30))['data']; print(round(d['total_credits'] - d['total_usage'], 2))"
}

# eval-retry re-reads the seed directory from the ABSOLUTE path the log recorded (the laptop's). Missing,
# Petri takes the path string as one seed and silently starts a fresh 1-sample audit (2026-10-15), so
# recreate that path here as a link to this checkout's seeds, and refuse to run if it still fails.
SEEDS_AT=$(.venv/bin/python -c "
from inspect_ai.log import read_eval_log
print(read_eval_log('$RETRY', header_only=True).eval.task_args['seed_instructions'])")
mkdir -p "$(dirname "$SEEDS_AT")" && ln -sfn /root/work/src/eval/audits/petri/petri-subscription/seeds-constitution "$SEEDS_AT"
[ "$(ls "$SEEDS_AT"/*.md | wc -l)" = 90 ] || { echo "!!! recorded seed dir $SEEDS_AT does not hold 90 seeds"; self_delete "seed path"; exit 1; }
echo ">>> recorded seed dir resolves: $SEEDS_AT (90 seeds)"

echo ">>> pre-run backup"
backup_logs || echo "!!! pre-run backup failed (continuing; the shipped log is also on the laptop)"

AOUT=output/logs/pod_base_resume_audit.txt
.venv/bin/python -u scratch/petri_seeds/run_audit.py --target "$TARGET" --connections "$CONN" \
  --port 8000 --retry "$RETRY" > "$AOUT" 2>&1 &
AUDIT=$!
echo ">>> audit pid $AUDIT, output $AOUT"

# Watch: back up every 10 min; stop the audit when credit is gone (balance lags, so 402s count too).
n=0
while kill -0 "$AUDIT" 2>/dev/null; do
  sleep 60; n=$((n + 1))
  left=$(credit_left 2>/dev/null || echo "?")
  errs=$(grep -ciE "Error code: 402|insufficient credits|requires more credits" "$AOUT")
  echo ">>> $(date -u +%H:%M:%S) credit=\$$left credit_errors=$errs"
  if [ "$errs" -gt 0 ] || { [ "$left" != "?" ] && .venv/bin/python -c "import sys; sys.exit(0 if $left < 0.5 else 1)"; }; then
    echo "!!! OpenRouter credit exhausted -- stopping the audit"
    pkill -INT -f "inspect eval-retry"; kill -INT "$AUDIT"
    for _ in $(seq 1 24); do kill -0 "$AUDIT" 2>/dev/null || break; sleep 5; done
    pkill -KILL -f "inspect eval-retry"; kill -KILL "$AUDIT" 2>/dev/null
    break
  fi
  [ $((n % 10)) -eq 0 ] && { backup_logs || echo "!!! backup failed"; }
done
wait "$AUDIT"; echo ">>> audit exited $? at $(date -u +%FT%TZ)"
grep -v "open failed" "$AOUT" | tail -15

# Push every scored seed (partial allowed, each missing seed named), then delete the pod.
.venv/bin/python - "$RUN_DIR" "$REPO" "$TARGET" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scratch/petri_seeds")
from push_results import SEEDS, valid_scores
from push_results import main as push
run_dir, repo, target = sys.argv[1:]
scores, _ = valid_scores(list(Path(run_dir, "logs").glob("*.eval")))
missing = sorted({p.stem for p in SEEDS.glob("*.md")} - set(scores))
print(f">>> scored {len(scores)}/90, missing {len(missing)}", flush=True)
push(run_dirs=run_dir, target=target, repo=repo, allow_missing=missing, context_window=32768)
PY
echo ">>> push rc=$?"
backup_logs || echo "!!! final backup failed"
self_delete "run finished"
