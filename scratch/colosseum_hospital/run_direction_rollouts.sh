#!/bin/bash
# ABOUTME: Runs ON the pod: one colosseum_hospital episode per direction config, back to back
# ABOUTME: (carried history, then board access), detached, one target in the pair, one seed.
#
#   ssh <pod> 'cd /root/work && bash scratch/colosseum_hospital/run_direction_rollouts.sh \
#       <port> <seed> <condition> <target> [<config> ...]'
#
# Each config is one `uv run evals` invocation (one vLLM server at a time fits one H100;
# the server is started and stopped per invocation). Default configs: the two 2026-09-09
# direction configs. Progress: output/logs/direction_<stem>_<port>.log per config and
# output/logs/directions_<port>.log for the sequence, which ends with DIRECTIONS_DONE.
set -euo pipefail

PORT="$1"; SEED="$2"; CONDITION="$3"; TARGET="$4"; shift 4
CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    CONFIGS=(
        configs/eval/2026-09-09_colosseum_hospital_carried_history.yaml
        configs/eval/2026-09-09_colosseum_hospital_board_access.yaml
    )
fi

cd /root/work
export PATH=/usr/local/bin:/root/.local/bin:$PATH
export HF_HOME=/workspace/hf
export COLOSSEUM_ROOT=/root/colosseum
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p output/logs
SLOG="output/logs/directions_${PORT}.log"

run_all() {
    for cfg in "${CONFIGS[@]}"; do
        stem="$(basename "${cfg}" .yaml)"
        log="output/logs/direction_${stem}_${PORT}.log"
        echo "$(date -u +%FT%TZ) START ${stem} seed=${SEED} ${CONDITION} -> ${log}"
        set +e
        # ARGUMENT ORDER IS LOAD-BEARING: --target first, terminated by --name; the
        # OmegaConf overrides trail at the end (docs/GOTCHAS.md).
        uv run evals --target "${TARGET}" --name colosseum_hospital --no-push \
            --port "${PORT}" --config "${cfg}" \
            "condition=${CONDITION}" "seeds=[${SEED}]" max_concurrent_runs=1 >> "${log}" 2>&1
        rc=$?
        set -e
        echo "$(date -u +%FT%TZ) EXIT ${rc} ${stem}"
    done
    echo "$(date -u +%FT%TZ) DIRECTIONS_DONE"
}

nohup bash -c "$(declare -f run_all); CONFIGS=($(printf '%q ' "${CONFIGS[@]}")); PORT=${PORT}; SEED=${SEED}; CONDITION=${CONDITION}; TARGET=$(printf '%q' "${TARGET}"); run_all" \
    >> "${SLOG}" 2>&1 < /dev/null &
echo "started pid $! -> ${SLOG}"
