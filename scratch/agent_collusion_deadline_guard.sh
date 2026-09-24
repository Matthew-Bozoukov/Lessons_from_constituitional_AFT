#!/usr/bin/env bash
# ABOUTME: Per-pod deadline for an unattended agent_collusion run: at DEADLINE (UTC HH:MM) upload this
# ABOUTME: pod's newest run dir to the arm's handoff repo under LABEL, THEN remove the pod — a deadline costs the unfinished remainder, not the run.
# Run on the pod: DEADLINE=21:00 REPO=<handoff repo> LABEL=deadline-A setsid nohup bash scratch/agent_collusion_deadline_guard.sh > /workspace/deadline_guard.log 2>&1 < /dev/null &
#
# Uploads are retried for up to 30 min; the pod is removed after that whatever happened, so the
# guard always bounds billing. A finished run removes its own pod first, taking this guard with it.
set -uo pipefail
: "${DEADLINE:?}" "${REPO:?}" "${LABEL:?}"
cd /root/work
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null
log() { echo ">>> $(date -u +%FT%TZ) [guard $LABEL] $*" | tee -a /workspace/collusion_driver.log; }

target=$(date -u -d "$DEADLINE" +%s)
[ "$target" -le "$(date -u +%s)" ] && target=$(date -u -d "tomorrow $DEADLINE" +%s)
log "armed: pod $RUNPOD_POD_ID saves to $REPO/$LABEL then is removed at $(date -u -d @"$target" +%FT%TZ)"
sleep $(( target - $(date -u +%s) ))

run=$(ls -td output/agent_collusion/20*_qwen36_* 2>/dev/null | head -1)
cp /workspace/collusion_driver.log "$run/metadata/collusion_driver_at_deadline.log" 2>/dev/null
give_up=$(( $(date -u +%s) + 1800 ))
until [ -z "$run" ] || uv run --no-sync python scratch/agent_collusion_shard_handoff.py upload "$run" "$LABEL" "$REPO"; do
    [ "$(date -u +%s)" -ge "$give_up" ] && { log "!!! upload still failing after 30 min; removing pod anyway"; break; }
    log "upload failed; retrying in 60 s"
    sleep 60
done
log "deadline reached; saved ${run:-nothing}; removing pod $RUNPOD_POD_ID"
runpodctl remove pod "$RUNPOD_POD_ID"
