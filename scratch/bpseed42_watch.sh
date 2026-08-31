#!/usr/bin/env bash
# poll the pod's train.log until the run pushes the adapter or dies; print terminal state only
SSH="ssh -p 14465 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 root@152.236.142.240"
for i in $(seq 1 240); do   # up to ~4h at 60s
  L=$($SSH 'grep -aiE "pushed adapter|Traceback|CUDA out of memory|TRAINING_FAILED|RuntimeError|AssertionError" /root/work/train.log | tail -3' 2>/dev/null)
  if echo "$L" | grep -qiE "pushed adapter"; then echo "DONE_PUSHED: $L"; exit 0; fi
  if echo "$L" | grep -qiE "Traceback|CUDA out of memory|TRAINING_FAILED|RuntimeError|AssertionError"; then echo "FAILED: $L"; exit 1; fi
  sleep 60
done
echo "TIMEOUT: watcher gave up after ~4h"; exit 2
