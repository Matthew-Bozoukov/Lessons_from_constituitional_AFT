#!/usr/bin/env bash
# ABOUTME: Wait for the matboz-t6note pod's boot to report READY, then launch the da-t6-note
# ABOUTME: 15% SFT run on it under nohup so the training survives this shell.
set -uo pipefail
POD=obpykf45ttoso1
SSH="ssh -p 14491 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no root@157.157.221.201"
MIX=dougalldeepmind/2026-10-01-da-t6-note-15-mix

for i in $(seq 1 80); do
  if curl -s --max-time 15 "https://${POD}-8080.proxy.runpod.net/boot.log" 2>/dev/null | grep -q READY; then
    echo ">>> READY after $((i*30))s"; break
  fi
  [ "$i" = 80 ] && { echo "!!! pod never reported READY after 40min — NOT launching"; exit 1; }
  sleep 30
done

# nohup + setsid so the run outlives this ssh session (CLAUDE.md gotcha 6), teeing to a
# timestamped log on the pod disk so `tail -f` works over ssh.
$SSH 'cd /root/work && mkdir -p output/logs && \
  LOG=output/logs/$(date +%Y%m%d_%H%M%S)_train_da_t6_note_15.log && \
  echo "$LOG" > output/logs/.latest_t6note && \
  setsid nohup uv run train --config configs/train/sft.yaml model=qwen36 \
    data_repo='"$MIX"' seed=0 wandb=true > "$LOG" 2>&1 < /dev/null & \
  sleep 5; echo "launched; log: $(cat output/logs/.latest_t6note)"'

echo ">>> training launched on $POD"
sleep 90
$SSH 'cd /root/work && tail -25 "$(cat output/logs/.latest_t6note)"'
