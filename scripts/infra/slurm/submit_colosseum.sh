#!/bin/bash
# ABOUTME: Submit colosseum_job.sh to Killarney with the account, GPU and walltime that
# ABOUTME: cluster needs, picking the shortest GPU partition whose limit fits the request.
# ============================================================================
# submit_colosseum.sh
# ============================================================================
#
#   bash scripts/infra/slurm/submit_colosseum.sh                     # all three experiments
#   EXPERIMENTS=collusion bash scripts/infra/slurm/submit_colosseum.sh
#   SEEDS=1-3 TIME=01:00:00 EXPERIMENTS=collusion \
#       bash scripts/infra/slurm/submit_colosseum.sh                 # smoke
#
# Everything that varies per cluster is passed on the sbatch COMMAND LINE, where it
# overrides the #SBATCH directives in the job script. #SBATCH lines are literal text and
# cannot read environment variables, so a value baked into the job file is correct on
# exactly one cluster.
#
#   ACCOUNT      charging account         (default: aip-s2ganapa)
#   GPU          gres spec                (default: h100:1 — see below)
#   TIME         walltime                 (default: 12:00:00)
#   MEM / CPUS   memory / cpus-per-task   (default: 64G / 16)
#   PARTITION    partition                (default: auto-selected from TIME)
#   DRY_RUN=1    print the sbatch line and stop

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

ACCOUNT="${ACCOUNT:-aip-s2ganapa}"
# H100 and not L40S, despite L40S having ~150 nodes and scheduling far sooner: Qwen3.6-27B
# is ~54GB in bf16 and the repo's vLLM passes no --tensor-parallel-size, so one server is
# one GPU and a 48GB card cannot hold it. The job script re-checks this on the node.
GPU="${GPU:-h100:1}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-96G}"
# 8, and the CPU count is what gates the QUEUE on this cluster, not walltime. Measured
# with `sbatch --test-only` for one H100 in gpubase_h100_b1, same 2h job, varying only
# the CPU ask: 16 CPUs -> start in 2h13m, 12 -> 33m, 8 -> immediately. Memory barely
# moved it. The free H100 nodes are mostly full of other jobs' cores, so asking for
# fewer is the difference between running now and running after lunch.
#
# 8 is still enough: Colosseum's max_concurrent_runs is THREADS on asyncio's default
# executor, which caps at min(32, cpu_count + 4) = 12 here, against a configured 5.
CPUS="${CPUS:-8}"

slurm_time_to_minutes() {
    local t="${1:-}" d=0 h=0 m=0 s=0
    [ -z "${t}" ] || [ "${t}" = "UNLIMITED" ] || [ "${t}" = "infinite" ] && { echo 99999999; return; }
    case "${t}" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    IFS=: read -r a b c <<<"${t}"
    case "$(awk -F: '{print NF}' <<<"${t}")" in
        3) h="${a}"; m="${b}"; s="${c}" ;;
        2) m="${a}"; s="${b}" ;;
        1) m="${a}" ;;
    esac
    echo $(( 10#${d:-0} * 1440 + 10#${h:-0} * 60 + 10#${m:-0} + (10#${s:-0} > 0 ? 1 : 0) ))
}

# Killarney bins GPU partitions by maximum walltime (gpubase_*_b1 3h, _b2 12h, _b3 24h…)
# and has no default that fits every request. Pick the SHORTEST bin that still fits, since
# longer bins are more contended.
pick_partition() {
    local gtype="${1%%:*}" want best best_min part limit gres pmin
    command -v sinfo >/dev/null 2>&1 || return 0
    want=$(slurm_time_to_minutes "${TIME}")
    best=""; best_min=99999999
    while IFS='|' read -r part limit gres; do
        part="${part%\*}"                       # sinfo marks the default with *
        [ -n "${part}" ] || continue
        case "${gres}" in *gpu:${gtype}*) ;; *) continue ;; esac
        # Interactive/debug partitions tie with the short batch bin on walltime but reject
        # or heavily restrict batch work, so exclude them by name rather than by luck.
        case "${part}" in *interac*|*interactive*|*debug*|*test*) continue ;; esac
        pmin=$(slurm_time_to_minutes "${limit}")
        if [ "${pmin}" -ge "${want}" ] && [ "${pmin}" -lt "${best_min}" ]; then
            best="${part}"; best_min="${pmin}"
        fi
    done < <(sinfo -h -o "%P|%l|%G" 2>/dev/null | sort -u)
    [ -n "${best}" ] && echo "${best}"
}

if [ -z "${PARTITION:-}" ]; then
    AUTO="$(pick_partition "${GPU}")"
    [ -n "${AUTO}" ] && { PARTITION="${AUTO}"; echo "auto-selected partition ${PARTITION} for ${GPU} @ ${TIME}"; }
fi

SBATCH_ARGS=(
    --account="${ACCOUNT}"
    --time="${TIME}"
    --mem="${MEM}"
    --cpus-per-task="${CPUS}"
    --gres="gpu:${GPU}"
    --job-name=colosseum
)
[ -n "${PARTITION:-}" ] && SBATCH_ARGS+=(--partition="${PARTITION}")

# Forwarded to the job, then on to `uv run evals`.
export EXPERIMENTS="${EXPERIMENTS:-collusion single cooperation}"
export SEEDS="${SEEDS:-}"
export EXTRA="${EXTRA:-} $*"

mkdir -p logs/slurm

echo "account     : ${ACCOUNT}"
echo "gpu         : ${GPU}"
echo "partition   : ${PARTITION:-<default>}"
echo "time/mem    : ${TIME} / ${MEM}, ${CPUS} cpus"
echo "experiments : ${EXPERIMENTS}"
echo "seeds       : ${SEEDS:-<config default>}"
echo
echo "sbatch ${SBATCH_ARGS[*]} scripts/infra/slurm/colosseum_job.sh"

[ "${DRY_RUN:-0}" = "1" ] && exit 0
sbatch "${SBATCH_ARGS[@]}" scripts/infra/slurm/colosseum_job.sh
