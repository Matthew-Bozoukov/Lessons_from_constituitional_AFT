#!/usr/bin/env bash
# ABOUTME: Unattended per-pod job for an agent_collusion run split over two pods: each runs its
# ABOUTME: shard (resumed from a checkpoint); pod B then merges both shards, publishes ONE run, and each pod removes itself.
# Run on each pod (checkpoint dirs already restored under output/agent_collusion/):
#   ROLE=A SEQS=1-25  REPO=<handoff repo> CKPT="<dir1>,<dir2>" setsid nohup bash scratch/agent_collusion_shard_job.sh <hf_target> > /workspace/collusion_driver.log 2>&1 < /dev/null &
#   ROLE=B SEQS=26-50 ...same...
#
# Failure handling, since nobody is watching:
#   * A shard that fails part-way still hands over what it finished; the final merge on B
#     re-runs anything unfinished on B's GPU, so one lost shard costs time, not the run.
#   * If A never hands over (B waits WAIT_MIN), B merges without it and runs A's remaining
#     trajectories itself.
#   * If the final publish fails, B holds 2 h so the run dirs can be recovered, then removes itself.
set -uo pipefail
TARGET="$1"
: "${ROLE:?}" "${SEQS:?}" "${REPO:?}"
CKPT="${CKPT:-}"   # comma-separated run dirs to resume from; empty = start fresh
WAIT_MIN="${WAIT_MIN:-300}"
# The sequences the PUBLISHED run covers (B's final merge); null = all 50. Must equal the
# union of the shards, or the final merge runs the missing ones itself.
FINAL_SEQS="${FINAL_SEQS:-}"
cd /root/work

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null
export HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0
log() { echo ">>> $(date -u +%FT%TZ) [$ROLE] $*"; }
teardown() { log "terminating pod $RUNPOD_POD_ID"; runpodctl remove pod "$RUNPOD_POD_ID"; }
newest_run() { ls -td output/agent_collusion/20*_qwen36_* 2>/dev/null | head -1; }
# EVAL overrides go BEFORE --target: it takes several arms and would swallow them.
# EXTRA: overrides applied to EVERY eval call here (the shard and the final merge alike).
evals() { uv run --no-sync evals ${EXTRA:-} "$@" --name agent_collusion --target "$TARGET"; }

before=$(newest_run)
log "shard $SEQS: resume_from=[${CKPT}], not pushed"
evals --no-push "sequence_ids=$SEQS" ${CKPT:+"resume_from=[$CKPT]"}
rc=$?
shard=$(newest_run)
if [ "$shard" = "$before" ]; then shard=""; fi
log "shard exit $rc; run dir: ${shard:-none}"

if [ "$ROLE" = A ]; then
    if [ -n "$shard" ] && uv run --no-sync python scratch/agent_collusion_shard_handoff.py upload "$shard" A "$REPO"; then
        log "handed over; done"
        sleep 30
        teardown
    else
        log "!!! handoff FAILED — holding 2 h so $shard can be recovered"
        sleep 7200
        teardown
    fi
    exit 0
fi

# ROLE B: gather A's shard (or go without it), then the final merge publishes one run.
dirs="$CKPT${shard:+,$(realpath "$shard")}"
dirs="${dirs#,}"
if a_dir=$(uv run --no-sync python scratch/agent_collusion_shard_handoff.py download /root/shards A "$REPO" "$WAIT_MIN" | tail -1) \
   && [ -d "$a_dir/rollouts" ]; then
    dirs="${dirs:+$dirs,}$a_dir"
    log "shard A received: $a_dir"
else
    log "!!! shard A not received — merging without it; B runs A's unfinished trajectories"
fi
log "final merge + publish: resume_from=[$dirs] sequence_ids=${FINAL_SEQS:-all}"
if evals "resume_from=[$dirs]" ${FINAL_SEQS:+"sequence_ids=$FINAL_SEQS"}; then
    log "EVAL OK — results pushed"
    sleep 60
    teardown
else
    log "!!! final publish FAILED — holding 2 h so output/agent_collusion can be recovered"
    sleep 7200
    teardown
fi
