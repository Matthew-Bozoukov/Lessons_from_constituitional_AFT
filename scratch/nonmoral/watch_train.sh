#!/usr/bin/env bash
# ABOUTME: Poll the training pod and emit ONE line when the run ends — completed, pushed, or died.
# ABOUTME: Run: bash scratch/nonmoral/watch_train.sh <ssh_port> <host>
#
# Deliberately silent while the run is healthy. A per-loss-line monitor over a two-hour epoch is
# ~40 notifications that all say the same thing; what is worth waking for is the end of the run.
# Coverage matters more than tidiness here: the grep must match the FAILURE signatures too, or a
# crashloop is indistinguishable from a long quiet epoch. The pgrep fallback catches a death that
# leaves nothing in the log at all.
set -uo pipefail
PORT="${1:?ssh port required}"
HOST="${2:?host required}"
SSH=(ssh -p "$PORT" -o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST")

while true; do
  out=$("${SSH[@]}" "tail -c 6000 /root/work/train.log 2>/dev/null | tr '\r' '\n' | grep -E 'train_runtime|train_loss|huggingface\.co/|Traceback|CUDA out of memory|OOM|Killed|ChildFailedError|AssertionError' | tail -4" 2>&1)
  if [ -n "$out" ]; then echo "$out"; break; fi
  if ! "${SSH[@]}" "pgrep -f train_lora.py >/dev/null" 2>/dev/null; then
    echo "TRAINER PROCESS GONE and no marker in the log — inspect /root/work/train.log"
    break
  fi
  sleep 300
done
