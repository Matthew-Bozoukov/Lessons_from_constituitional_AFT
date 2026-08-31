#!/usr/bin/env bash
# ABOUTME: Runs ON a RunPod pod: finishes (or starts) one training run, verifies the adapter
# ABOUTME: reached Hugging Face, then destroys the pod itself. Nothing local participates.
#
# Usage (on the pod): nohup bash /workspace/pod_guardian.sh <config> <hf_repo> [max_hours] &
#
# Same four rules as scripts/gpu/runpod_train_selfdestruct.sh, which this is a variant of:
# teardown is unconditional (trap, not &&); there is a hard time ceiling; the pod is NEVER
# destroyed before the adapter is verified READABLE on HF; the log is written where it can
# be pulled. The difference is that this one ATTACHES to a run already in progress instead
# of only starting a fresh one -- needed when training was launched before the guardian.
set -uo pipefail

CONFIG="${1:?usage: pod_guardian.sh <config> <hf_repo> [max_hours]}"
HF_REPO="${2:?hf_repo required}"
MAX_HOURS="${3:-6}"
LOG=/workspace/guardian.log
KEEP_ALIVE=0

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Credentials come from a mode-600 file, never the command line: an ssh/nohup argument
# lands in the pod's process list.
source /workspace/.rp_env
set -a; . /workspace/repo/.env; set +a

verify_hf() {
  python3 - "$HF_REPO" <<'PY'
import os, sys
from huggingface_hub import HfApi
try:
    f = [x.rfilename for x in HfApi(token=os.environ["HF_TOKEN"]).model_info(sys.argv[1]).siblings]
    ok = "adapter_model.safetensors" in f and "training_meta.json" in f
    print("VERIFIED" if ok else f"INCOMPLETE {f}")
    sys.exit(0 if ok else 1)
except Exception as e:
    print(f"UNVERIFIED {type(e).__name__}: {e}")
    sys.exit(1)
PY
}

teardown() {
  local status=$1
  log "=== job exited status=${status}; verifying ${HF_REPO} on HF"
  if verify_hf | tee -a "$LOG" | grep -q VERIFIED; then
    log "adapter verified on HF -- safe to destroy"
  else
    KEEP_ALIVE=1
    log "!! adapter NOT verified on HF. KEEPING POD ALIVE so the run is not lost."
    log "!! pull it from /workspace/out/*/adapter, then destroy by hand."
  fi
  if [ "$KEEP_ALIVE" = "1" ]; then exit "$status"; fi
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "!! no RUNPOD_API_KEY/POD_ID -- DESTROY THIS POD BY HAND"; exit "$status"
  fi
  log "destroying pod ${RUNPOD_POD_ID}"
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" | tee -a "$LOG"
  sleep 20
  log "destroy requested; if you can still read this it did not take"
  exit "$status"
}
trap 'teardown $?' EXIT

cd /workspace/repo
export HF_HOME=/workspace/hf PYTHONPATH=/workspace/repo WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false

if pgrep -f "train_lora.py --config" >/dev/null 2>&1; then
  log "training already running -- attaching, will wait for it to finish"
  # Hard ceiling even on the attach path: a hung trainer must not bill forever.
  END=$(( $(date +%s) + MAX_HOURS * 3600 ))
  while pgrep -f "train_lora.py --config" >/dev/null 2>&1; do
    [ "$(date +%s)" -gt "$END" ] && { log "!! max_hours reached; killing trainer"; pkill -f train_lora.py; break; }
    sleep 60
  done
  log "trainer no longer running"
else
  log "no trainer running -- starting one for ${CONFIG}"
  timeout "${MAX_HOURS}h" torchrun --nproc_per_node=2 \
    scripts/train/train_lora.py --config "$CONFIG" >> /workspace/train.log 2>&1
  log "trainer exited $?"
fi
