#!/bin/bash
# ABOUTME: Print, for each pod given, any terminal line of its queue log (QUEUE_DONE / EXIT n),
# ABOUTME: prefixed by the host — empty output means every queue is still running.
#
#   bash scratch/colosseum_hospital/poll_queues.sh root@ip:port [root@ip:port ...]
#   until bash scratch/colosseum_hospital/poll_queues.sh A B | grep -q .; do sleep 300; done
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
for host in "$@"; do
    out="$(bash "${HERE}/pod_ssh.sh" "${host}" 'grep -E "QUEUE_DONE|EXIT [0-9]" /root/work/output/logs/queue_8000.log 2>/dev/null' 2>/dev/null | grep -v "Permanently added" || true)"
    if [ -n "${out}" ]; then
        printf '%s\n' "${out}" | sed "s#^#${host} #"
    fi
done
