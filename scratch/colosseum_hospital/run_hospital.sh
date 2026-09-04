#!/bin/bash
# ABOUTME: Runs ON the pod: one colosseum_hospital invocation for one condition, detached,
# ABOUTME: with its own port and log, so several conditions can run on several pods.
#
#   ssh <pod> 'cd /root/work && bash scratch/colosseum_hospital/run_hospital.sh \
#       <condition> <lo>-<hi> <port> <target> [<target>...]'
#
# Launches `uv run evals` under nohup and returns at once; progress is
# output/colosseum_hospital/<arm>/rollouts/colosseum/<ts>/progress.json and the log is
# output/logs/hospital_<condition>_<port>.log. `--no-push` because the judge and the push
# happen off the box (scripts/eval/publish_colosseum.py --eval colosseum_hospital).
set -euo pipefail

CONDITION="$1"; RANGE="$2"; PORT="$3"; shift 3
TARGETS=("$@")
[ "${#TARGETS[@]}" -ge 1 ] || { echo "usage: $0 <condition> <lo>-<hi> <port> <target>..." >&2; exit 2; }

LO="${RANGE%-*}"; HI="${RANGE#*-}"
SEEDS="$(python3 -c "print(','.join(str(s) for s in range(int('${LO}'), int('${HI}') + 1)))")"

cd /root/work
export PATH=/usr/local/bin:/root/.local/bin:$PATH
export HF_HOME=/workspace/hf
export COLOSSEUM_ROOT=/root/colosseum
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p output/logs
LOG="output/logs/hospital_${CONDITION}_${PORT}.log"

# ARGUMENT ORDER IS LOAD-BEARING: --target is nargs='+', so it goes first and is
# terminated by --name; the OmegaConf overrides trail at the end (docs/GOTCHAS.md).
nohup uv run evals --target "${TARGETS[@]}" --name colosseum_hospital --no-push \
    --port "${PORT}" "condition=${CONDITION}" "seeds=[${SEEDS}]" \
    > "${LOG}" 2>&1 < /dev/null &
echo "started pid $! -> ${LOG}"
