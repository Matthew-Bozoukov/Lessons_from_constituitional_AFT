#!/usr/bin/env bash
# ABOUTME: Pod-side driver for one hospital arm: run the seed pieces, publish each, upload
# ABOUTME: this log, then TERMINATE THIS POD -- the only teardown that works unattended.
#
# RunPod accepts `terminateAfter` at creation and does not act on it (runpod/docs#820,
# runpodctl#331): the deadline is stored on no code path and pods bill straight past it.
# `up`'s other cap is a watchdog on the laptop, which dies when the laptop does. So for a
# run nobody is watching, the pod killing itself is the whole safety mechanism, and
# everything below is arranged so that it happens on every exit path:
#   - the work runs under `timeout`, so a hang cannot outlive the budget;
#   - termination sits in a trap, so a crash, a kill or a bad config still reaches it;
#   - the DELETE is retried and VERIFIED, because a 200 has been seen to leave a pod
#     listed and still billing (src/infra/runpod.py terminate()).
set -uo pipefail
exec > >(tee -a /root/run.log) 2>&1
set +x   # never xtrace: this shell holds credentials

source /root/.eval.env    # HF_TOKEN, HF_ORG, OPENROUTER_API_KEY, RUNPOD_API_KEY, POD_ID, ARM, TARGET
export PATH=/root/.local/bin:$PATH

upload_log () {
  cd /root/work 2>/dev/null || return 0
  ARM="$ARM" uv run python - <<'PY' 2>&1 | tail -2 || echo "log upload failed"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{os.environ.get('HF_ORG','dougalldeepmind')}/2026-10-06-hospital-run-logs"
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
api.upload_file(path_or_fileobj="/root/run.log", path_in_repo=f"{os.environ['ARM']}.log",
                repo_id=repo, repo_type="dataset")
print("log uploaded to", repo)
PY
}

self_terminate () {
  echo "=== $(date -u +%FT%TZ) terminating pod ${POD_ID} ==="
  for i in $(seq 1 10); do
    curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
         "https://rest.runpod.io/v1/pods/${POD_ID}" >/dev/null 2>&1
    sleep 10
    # Billing stops when it is GONE, not when the DELETE returns: verify by re-reading.
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
           "https://rest.runpod.io/v1/pods/${POD_ID}" 2>/dev/null)
    echo "  terminate attempt ${i}: pod GET -> ${code}"
    [ "$code" = "404" ] && { echo "  pod confirmed gone"; return 0; }
  done
  echo "  WARNING: pod still listed after 10 attempts -- it is STILL BILLING"
}

# Every exit path lands here, including a crash or an outside kill.
trap 'upload_log; self_terminate' EXIT

cd /root/work || { echo "FATAL: no clone at /root/work"; exit 1; }
echo "=== $(date -u +%FT%TZ) arm=$ARM target=$TARGET pod=$POD_ID ==="

for i in $(seq 1 160); do
  if uv run evals --help >/dev/null 2>&1; then echo ">>> toolchain ready after ${i} polls"; break; fi
  [ "$i" = 160 ] && { echo "FATAL: toolchain never came up"; exit 1; }
  sleep 30
done

run_piece () {
  local seeds="$1" label="$2"
  echo "=== $(date -u +%FT%TZ) piece ${label}: seeds=${seeds} ==="
  timeout 105m uv run evals --name colosseum_hospital --target "$TARGET" \
    --config scratch/colosseum_hospital/configs/2026-09-09_colosseum_hospital_board_access.yaml \
    condition=self_sacrificial "seeds=${seeds}"
  echo "=== $(date -u +%FT%TZ) piece ${label} EXIT=$? ==="
}

run_piece "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"          "1of2"
run_piece "[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]" "2of2"
echo "=== $(date -u +%FT%TZ) ALL PIECES DONE ==="
