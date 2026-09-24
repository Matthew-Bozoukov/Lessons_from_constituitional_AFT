#!/usr/bin/env bash
# ABOUTME: Unattended on-pod agent_collusion run: smoke, full eval (pushed to HF), then the pod
# ABOUTME: terminates itself — at once on success, after a 2 h recovery hold on failure.
# Run on the pod: setsid nohup bash scratch/agent_collusion_onbox.sh <hf_target> > /workspace/collusion_driver.log 2>&1 < /dev/null &
set -uo pipefail
TARGET="$1"
cd /root/work

# The pod-scoped RunPod key and id live in the container's init environment, not in an SSH
# session's; they are what lets the pod remove itself with no laptop attached.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null
# docs/GOTCHAS.md 2026-09-21: an on-box run must set what the SSH path sets for it.
export HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0

teardown() {
    echo ">>> $(date -u +%FT%TZ) terminating pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID"
}

# SKIP_SMOKE=1 goes straight to the full run, for a relaunch whose wiring a smoke on this
# same pod has already verified.
smoke() {
    [ "${SKIP_SMOKE:-0}" = 1 ] && { echo ">>> smoke skipped (SKIP_SMOKE=1)"; return 0; }
    echo ">>> $(date -u +%FT%TZ) smoke: 2 trajectories x 2 episodes, not pushed"
    uv run --no-sync evals --name agent_collusion --target "$TARGET" --no-push smoke=true
}
# EVAL_OVERRIDES go BEFORE --target: it takes several arms and would swallow them.
# --no-sync: the env was synced at boot without the train-only causal-conv1d kernel, and
# a re-sync here would try to build it and fail.
if smoke \
   && { echo ">>> $(date -u +%FT%TZ) full: 50 trajectories x 10 episodes"; \
        uv run --no-sync evals ${EVAL_OVERRIDES:-} --name agent_collusion --target "$TARGET"; }; then
    echo ">>> $(date -u +%FT%TZ) EVAL OK — results pushed"
    sleep 60
    teardown
else
    echo "!!! $(date -u +%FT%TZ) EVAL FAILED — holding the pod 2 h so partial trajectories" \
         "in /root/work/output/agent_collusion can be recovered"
    sleep 7200
    teardown
fi
