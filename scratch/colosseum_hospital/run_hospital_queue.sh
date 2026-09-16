#!/bin/bash
# ABOUTME: Runs ON the pod: several colosseum_hospital invocations back to back (one vLLM
# ABOUTME: server at a time fits one H100), detached, each with its own log.
#
#   ssh <pod> 'cd /root/work && bash scratch/colosseum_hospital/run_hospital_queue.sh \
#       <port> <target> [<target>...] -- <condition>:<lo>-<hi>[:<target>[,<target>]] ...'
#   (the seeds may also be an explicit comma list: <condition>:3,7,12:<target>)
#
# Every job runs `uv run evals --no-push` for one condition over the job's targets — the
# comma-separated third field, or the global list before `--` when a job names none. Arms
# run sequentially inside an invocation, LoRA-swapping on the one server. The queue is
# detached under nohup and returns at once; output/logs/queue_<port>.log records each
# job's start and exit code, output/logs/hospital_<condition>_<port>.log is each job's
# own log (appended: a rerun of the same condition continues the same file).
set -euo pipefail

PORT="$1"; shift
TARGETS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do TARGETS+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "usage: $0 <port> <target>... -- <condition>:<lo>-<hi>[:<targets>]..." >&2; exit 2; }
shift
JOBS=("$@")
[ "${#TARGETS[@]}" -ge 1 ] && [ "${#JOBS[@]}" -ge 1 ] || { echo "need targets and jobs" >&2; exit 2; }

cd /root/work
export PATH=/usr/local/bin:/root/.local/bin:$PATH
export HF_HOME=/workspace/hf
export COLOSSEUM_ROOT=/root/colosseum
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p output/logs
QLOG="output/logs/queue_${PORT}.log"

run_queue() {
    for job in "${JOBS[@]}"; do
        IFS=: read -r condition range job_targets <<< "${job}"
        # A job's seeds are a lo-hi range, or an explicit comma list (a top-up of the
        # seeds an earlier run dropped — scratch/colosseum_hospital/topup_plan.py).
        if [[ "${range}" == *,* ]]; then
            seeds="${range}"; lo="${range%%,*}"; hi="${range##*,}"
        else
            lo="${range%-*}"; hi="${range#*-}"
            seeds="$(python3 -c "print(','.join(str(s) for s in range(int('${lo}'), int('${hi}') + 1)))")"
        fi
        if [ -n "${job_targets:-}" ]; then
            IFS=, read -r -a targets <<< "${job_targets}"
        else
            targets=("${TARGETS[@]}")
        fi
        log="output/logs/hospital_${condition}_${PORT}.log"
        echo "$(date -u +%FT%TZ) START ${condition} ${lo}-${hi} targets=${targets[*]} -> ${log}"
        # ARGUMENT ORDER IS LOAD-BEARING: --target first, terminated by --name; the
        # OmegaConf overrides trail at the end (docs/GOTCHAS.md).
        set +e
        # EXTRA (env): further OmegaConf overrides, e.g. EXTRA="max_concurrent_runs=30".
        uv run evals --target "${targets[@]}" --name colosseum_hospital --no-push \
            --port "${PORT}" "condition=${condition}" "seeds=[${seeds}]" ${EXTRA:-} >> "${log}" 2>&1
        rc=$?
        set -e
        echo "$(date -u +%FT%TZ) EXIT ${rc} ${condition} ${lo}-${hi}"
    done
    echo "$(date -u +%FT%TZ) QUEUE_DONE"
}

nohup bash -c "$(declare -f run_queue); JOBS=($(printf '%q ' "${JOBS[@]}")); TARGETS=($(printf '%q ' "${TARGETS[@]}")); PORT=${PORT}; EXTRA=$(printf '%q' "${EXTRA:-}"); run_queue" \
    >> "${QLOG}" 2>&1 < /dev/null &
echo "queued ${#JOBS[@]} job(s) pid $! -> ${QLOG}"
