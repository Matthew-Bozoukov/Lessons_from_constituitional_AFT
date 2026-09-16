#!/bin/bash
# ABOUTME: Run one command on a pod given as root@<ip>:<port>, with the throwaway-host ssh
# ABOUTME: options RunPod rentals need (ports are recycled, so no remembered host keys).
#
#   bash scratch/colosseum_hospital/pod_ssh.sh root@<ip>:<port> '<remote command>'
set -euo pipefail
HOST="$1"; shift
PORT="${HOST##*:}"; ADDR="${HOST%:*}"
exec ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=20 -o ServerAliveInterval=30 "${ADDR}" "$@"
