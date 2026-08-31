#!/bin/bash
# ABOUTME: Keep an SSH tunnel alive from the laptop docker bridge (172.17.0.1:8000) to the
# ABOUTME: box vLLM (localhost:8000) so ODCV scenario containers can reach the served model.
while true; do
  ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -L 172.17.0.1:8000:localhost:8000 \
    odcv-gpu
  echo "[keeper] tunnel dropped ($(date '+%T')); reconnecting in 3s" >&2
  sleep 3
done
