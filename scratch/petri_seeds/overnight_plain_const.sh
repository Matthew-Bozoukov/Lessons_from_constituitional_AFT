#!/usr/bin/env bash
# ABOUTME: Runs ON the GPU pod, unattended: Petri audit of the plain control with the whole
# ABOUTME: constitution prepended -- verify injection on 2 seeds, run all 90, push, delete the pod.
# Run (on the pod, detached): nohup bash scratch/petri_seeds/overnight_plain_const.sh </dev/null &
set -uo pipefail
cd /root/work
export PATH=/usr/local/bin:/root/.local/bin:$PATH HF_HOME=/workspace/hf
# Serving on the box (LocalExec) skips the two pod facts SshExec sets; without this one vLLM's
# engine dies at start-up JIT-compiling flashinfer's sampler with an nvcc the image lacks
# (docs/GOTCHAS.md 2026-09-21).
export VLLM_USE_FLASHINFER_SAMPLER=0
set -a; . ./.env; set +a
# RunPod gives the container its own id and a pod-scoped API key; an SSH-launched shell does not
# inherit the container's environment, so read them from PID 1.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
[ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || { echo "!!! no RUNPOD_POD_ID/RUNPOD_API_KEY: cannot self-delete"; exit 1; }
export PETRI_SYSTEM_PREAMBLE_FILE=/root/work/constitutions/claude_distilled_09_principles/constitution.md
TARGET=dougalldeepmind/2026-10-05-qwen36-0-plain
CTX=65536
DEADLINE_S=${DEADLINE_S:-36000}   # 10 h: the pod deletes itself by then whatever happens
mkdir -p output/logs
LOG=output/logs/overnight_plain_const.log
exec > >(tee -a "$LOG") 2>&1
echo ">>> start $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-?} target=$TARGET ctx=$CTX deadline=${DEADLINE_S}s"

self_delete() {
  echo ">>> deleting pod ${RUNPOD_POD_ID} at $(date -u +%FT%TZ): $1"
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" -o /dev/null -w "    delete HTTP %{http_code}\n"
  runpodctl remove pod "${RUNPOD_POD_ID}" 2>/dev/null || true
}
# Backstop: delete the pod at the deadline even if everything below hangs.
( sleep "$DEADLINE_S"; self_delete "deadline ${DEADLINE_S}s reached" ) &

push_whatever() {   # push the scored seeds (partial allowed, each missing seed named) + this log
  local run_dir="$1"
  local repo; repo="$(date -u +%F)-petri-constitution-qwen36-0-plain"
  .venv/bin/python - "$run_dir" "$repo" <<'PY'
import subprocess, sys
from pathlib import Path
from src.eval.audits.petri.tenet_audit.scores import SEEDS, valid_scores
run_dir, repo = sys.argv[1], sys.argv[2]
scores, _ = valid_scores(list(Path(run_dir, "logs").glob("*.eval")))
missing = sorted({p.stem for p in SEEDS.glob("*.md")} - set(scores))
print(f">>> scored {len(scores)}/90, missing {len(missing)}")
if not scores:
    sys.exit(3)
cmd = [".venv/bin/python", "scripts/eval/audits/petri/push_results.py", "--run_dirs", run_dir,
       "--target", "dougalldeepmind/2026-10-05-qwen36-0-plain", "--repo", repo,
       "--preamble", "constitutions/claude_distilled_09_principles/constitution.md",
       "--context_window", "65536"]
if missing:
    cmd += ["--allow_missing", "[" + ",".join(missing) + "]"]
sys.exit(subprocess.run(cmd).returncode)
PY
  local rc=$?
  .venv/bin/python -c "
import os, sys
from huggingface_hub import HfApi
a = HfApi(token=os.environ['HF_TOKEN']); r = f\"{os.environ['HF_ORG']}/$repo\"
a.create_repo(r, repo_type='dataset', exist_ok=True)
a.upload_file(path_or_fileobj='$LOG', path_in_repo='logs/overnight_driver.log', repo_id=r, repo_type='dataset')
print('>>> driver log pushed to', r)"
  return $rc
}

# 1. Injection check on 2 seeds: the constitution must be first in every target system message.
echo ">>> VERIFY: 2 seeds"
.venv/bin/python -u scripts/eval/audits/petri/run_audit.py --target "$TARGET" --connections 4 --limit 2 \
  --context-window "$CTX"
VLOG=$(ls -t output/petri/*/logs/*.eval | head -1)
if ! .venv/bin/python scripts/eval/audits/petri/verify_target_preamble.py "$VLOG" "$PETRI_SYSTEM_PREAMBLE_FILE"; then
  echo "!!! constitution NOT first in every target system message ($VLOG) -- not running the 90"
  push_whatever "$(dirname "$(dirname "$VLOG")")" || true
  self_delete "verification failed"; exit 1
fi
echo ">>> injection verified in $VLOG"

# 2. The full 90 seeds, in a fresh run directory.
echo ">>> FULL: 90 seeds"
.venv/bin/python -u scripts/eval/audits/petri/run_audit.py --target "$TARGET" --connections 24 \
  --context-window "$CTX"
echo ">>> audit exited $? at $(date -u +%FT%TZ)"
RUN=$(dirname "$(dirname "$(ls -t output/petri/*/logs/*.eval | head -1)")")
.venv/bin/python scripts/eval/audits/petri/verify_target_preamble.py "$(ls -t $RUN/logs/*.eval | head -1)" \
  "$PETRI_SYSTEM_PREAMBLE_FILE" || echo "!!! full-run injection check FAILED -- see the pushed card/log"

# 3. Push (partial if the credit or the audit ran out), then delete the pod.
push_whatever "$RUN"; echo ">>> push rc=$?"
self_delete "run finished"
