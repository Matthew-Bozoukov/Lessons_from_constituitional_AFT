#!/usr/bin/env bash
# ABOUTME: Pod-side driver for one hospital arm: wait for the clone, run the seed pieces,
# ABOUTME: publish each, and upload this log whatever happens. Runs unattended, laptop off.
#
# The pod expires provider-side (terminateAfter, set at creation), so nothing here needs to
# terminate anything -- and this box deliberately holds no RunPod key, which is account-wide
# on a shared account. Seeds run in two 15-wide pieces because that is what the September
# runs did (metadata/pieces/), and because each piece PUBLISHES on its own: if the deadline
# fires mid-run the finished piece is already on the Hub instead of dying with the pod.
set -uo pipefail
exec > >(tee -a /root/run.log) 2>&1
set +x   # never xtrace: this shell holds credentials

source /root/.eval.env           # HF_TOKEN, HF_ORG, OPENROUTER_API_KEY -- 600, outside /workspace
export PATH=/root/.local/bin:$PATH
cd /root/work || { echo "FATAL: no clone at /root/work"; exit 1; }

echo "=== $(date -u +%FT%TZ) arm=$ARM target=$TARGET ==="

# Gate on the toolchain actually working rather than on a boot-log marker: the clone and
# `uv sync` finish at their own pace and a marker that moved would strand this run.
for i in $(seq 1 160); do
  if uv run evals --help >/dev/null 2>&1; then echo ">>> toolchain ready after ${i} polls"; break; fi
  [ "$i" = 160 ] && { echo "FATAL: toolchain never came up"; break; }
  sleep 30
done

run_piece () {
  local seeds="$1" label="$2"
  echo "=== $(date -u +%FT%TZ) piece ${label}: seeds=${seeds} ==="
  uv run evals --name colosseum_hospital --target "$TARGET" \
    --config scratch/colosseum_hospital/configs/2026-09-09_colosseum_hospital_board_access.yaml \
    condition=self_sacrificial "seeds=${seeds}"
  echo "=== $(date -u +%FT%TZ) piece ${label} EXIT=$? ==="
}

run_piece "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"       "1of2"
run_piece "[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]" "2of2"

echo "=== $(date -u +%FT%TZ) ALL PIECES DONE ==="
# The log is the only postmortem anyone gets for a run nobody watched, and it dies with the
# pod, so it goes to the Hub whether the pieces passed or failed.
uv run python - <<'PY' || echo "log upload failed"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f"{os.environ.get('HF_ORG','dougalldeepmind')}/2026-10-06-hospital-run-logs"
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
api.upload_file(path_or_fileobj="/root/run.log", path_in_repo=f"{os.environ['ARM']}.log",
                repo_id=repo, repo_type="dataset")
print("log uploaded to", repo)
PY
echo "=== $(date -u +%FT%TZ) RUNNER COMPLETE ==="
