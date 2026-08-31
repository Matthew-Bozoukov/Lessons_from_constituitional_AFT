#!/bin/bash
# ABOUTME: Run one training job on a RunPod pod and have the pod destroy ITSELF when done,
# ABOUTME: so an unattended run cannot bill overnight. Usage: runpod_train_selfdestruct.sh <config>
#
# Run this ON THE POD, detached, then close your laptop:
#
#   nohup bash scripts/gpu/runpod_train_selfdestruct.sh \
#       configs/train/lora_qwen36_t2_9284_synthdoc_716_c6masked_dynbatch_2xh200.yaml \
#       </dev/null >/workspace/selfdestruct.log 2>&1 &
#
# Nothing on your machine participates after launch. The pod trains, verifies the adapter
# reached Hugging Face, and only then deletes itself through the RunPod API.
#
# The four rules this encodes, each learned from a way unattended runs waste money:
#
# 1. TEARDOWN IS UNCONDITIONAL. It runs on success, on crash, and on timeout — via `trap`,
#    not `&&`. A failed run that leaves the pod up is the expensive case, and it is the one
#    a naive `train && destroy` gets wrong.
# 2. THERE IS A HARD CEILING. `timeout` bounds the job. A hung process with no ceiling is a
#    pod billing forever, which is strictly worse than a killed job you can resume.
# 3. NOTHING IS DESTROYED BEFORE THE ARTIFACT IS SAFE. The adapter must be readable on HF
#    (not merely "the trainer printed a success line") before the pod is allowed to die.
#    If the push cannot be verified the pod STAYS UP and says why, because a lost 2-hour
#    training run costs far more than an extra hour of GPU.
# 4. THE LOG OUTLIVES THE POD. It is copied to HF alongside the adapter before teardown,
#    so a run you slept through is still diagnosable after the machine is gone.
#
# Residual risk this does NOT cover: if the pod itself dies (host failure, network
# partition) nothing runs the destroy call. That needs an external sweep — see the paid
# infrastructure section of CLAUDE.md.

set -uo pipefail   # NOT -e: a failing stage must still reach the teardown trap

CONFIG="${1:?usage: runpod_train_selfdestruct.sh <train config> [max_hours] [nproc]}"
MAX_HOURS="${2:-8}"
NPROC="${3:-$(nvidia-smi --list-gpus | wc -l)}"
WORK="${WORK:-/root/work}"
LOG="/workspace/train_$(date +%Y%m%d_%H%M%S).log"
KEEP_ALIVE=0        # set to 1 anywhere below to refuse teardown

mkdir -p /workspace "$WORK/output/logs"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

teardown() {
  local status=$1
  log "=== job exited with status ${status}"

  if [ "$KEEP_ALIVE" = "1" ]; then
    log "!! NOT destroying this pod: the adapter was not verified on Hugging Face."
    log "!! Pull it by hand from ${WORK}/\$(cat /workspace/outdir 2>/dev/null)/adapter"
    log "!! then: curl -X DELETE https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID} -H \"Authorization: Bearer \$RUNPOD_API_KEY\""
    exit "$status"
  fi

  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "!! cannot self-destruct: RUNPOD_API_KEY or RUNPOD_POD_ID missing from the env."
    log "!! DESTROY THIS POD BY HAND."
    exit "$status"
  fi

  log "destroying pod ${RUNPOD_POD_ID}"
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" | tee -a "$LOG"
  sleep 20   # the API call must land before the container is torn out from under us
  log "destroy requested; if you can still read this, it did not take — destroy by hand"
  exit "$status"
}
trap 'teardown $?' EXIT

cd "$WORK" || { log "no such directory: $WORK"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
set -a; . ./.env; set +a          # HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log "config   : $CONFIG"
log "gpus     : $NPROC"
log "ceiling  : ${MAX_HOURS}h"
log "pod      : ${RUNPOD_POD_ID:-<unset>}"

# The repo that must hold the adapter when this is over. Read from the config so the
# verification below cannot drift from what the trainer was actually told to push to.
HF_REPO=$(grep -E '^hf_repo:' "$CONFIG" | head -1 | sed 's/^hf_repo:[[:space:]]*//; s/^"//; s/"$//')
if [ -z "$HF_REPO" ]; then
  log "!! config declares no hf_repo — this run would leave nothing behind. Refusing."
  exit 1
fi
log "hf_repo  : $HF_REPO"

timeout "${MAX_HOURS}h" uv run torchrun --nproc_per_node="$NPROC" \
    scripts/train/train_lora.py --config "$CONFIG" 2>&1 | tee -a "$LOG"
TRAIN_STATUS=${PIPESTATUS[0]}
log "training exited ${TRAIN_STATUS}"

# Rule 3: trust the Hub, not the trainer's stdout. `adapter_model.safetensors` present in
# the repo is the artifact actually surviving this pod; anything else is a claim.
log "verifying the adapter is readable on the Hub..."
if uv run python - "$HF_REPO" >>"$LOG" 2>&1 <<'PY'; then
import os, sys
from huggingface_hub import HfApi
repo = sys.argv[1]
files = HfApi(token=os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")).list_repo_files(repo)
need = {"adapter_model.safetensors", "adapter_config.json", "training_meta.json"}
missing = need - set(files)
assert not missing, f"{repo} is missing {sorted(missing)}"
print(f"verified: {repo} holds {sorted(need)}")
PY
  log "adapter verified on the Hub — teardown is safe"
else
  KEEP_ALIVE=1
  log "!! adapter NOT verified on the Hub. Keeping this pod alive so the run is recoverable."
fi

# Rule 4: the log has to outlive the machine that wrote it.
if [ "$KEEP_ALIVE" = "0" ]; then
  uv run python - "$HF_REPO" "$LOG" >>"$LOG" 2>&1 <<'PY' || true
import os, sys
from huggingface_hub import HfApi
repo, log = sys.argv[1], sys.argv[2]
HfApi(token=os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")).upload_file(
    path_or_fileobj=log, path_in_repo=f"logs/{os.path.basename(log)}", repo_id=repo)
print("log uploaded")
PY
fi

exit "$TRAIN_STATUS"
