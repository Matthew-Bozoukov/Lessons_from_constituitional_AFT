#!/bin/bash
# ABOUTME: Per pod, print how many episodes each hospital arm dir on it has finished (blackboards.json
# ABOUTME: count per run dir) and the last line of its queue log — a one-glance progress view.
#
#   bash scratch/colosseum_hospital/progress.sh root@ip:port [root@ip:port ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE='for d in /root/work/output/colosseum_hospital/*/; do n=$(find "$d" -name blackboards.json 2>/dev/null | wc -l); echo "  $(basename "$d") finished=$n"; done; echo "  queue: $(tail -1 /root/work/output/logs/queue_8000.log 2>/dev/null)"; echo "  gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"'
for host in "$@"; do
    echo "== ${host}"
    bash "${HERE}/pod_ssh.sh" "${host}" "${REMOTE}" 2>/dev/null | grep -v "Permanently added"
done
