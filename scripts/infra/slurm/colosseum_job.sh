#!/bin/bash
# ABOUTME: SLURM job running all three Colosseum experiments in ONE allocation, one vLLM
# ABOUTME: server, both arms resident as LoRA adapters. Submit via submit_colosseum.sh.
# ============================================================================
# colosseum_job.sh — the whole multi-agent study in one job
# ============================================================================
#
# Why one job rather than a seed array: the model is the expensive part. Qwen3.6-27B is
# ~54GB and takes ~10 minutes to load, and a job array would pay that per task, collide on
# vLLM's port and its server work directory, and gain nothing — the GPU is already
# saturated by the six agents of a single episode plus whatever episodes run alongside
# them. So: one allocation, one server, three `uv run evals` invocations in sequence.
#
# Both checkpoints stay resident. They are LoRA adapters over the same base in the same
# thinking mode (verified: Qwen/Qwen3.6-27B, mode `think`, rank 64), so vLLM holds both at
# once and each of the six seats picks its arm by name in the request body. That is what
# makes a mixed-arm team possible on one GPU — see ServedTarget.sibling().
#
# ── Usage ───────────────────────────────────────────────────────────────────
#
#   bash scripts/infra/slurm/submit_colosseum.sh                  # all three experiments
#   EXPERIMENTS=collusion bash scripts/infra/slurm/submit_colosseum.sh
#   SEEDS=0-3 EXPERIMENTS=collusion bash scripts/infra/slurm/submit_colosseum.sh   # smoke
#
# ── Environment variables ───────────────────────────────────────────────────
#
#   EXPERIMENTS  space-separated subset of: collusion single cooperation
#                (default: all three, in that order)
#   SEEDS        seed range as `lo-hi` inclusive (default: the config's own list)
#   EXTRA        extra `key=value` OmegaConf overrides passed to every invocation
#
# NOTE: submit through submit_colosseum.sh, which supplies --account, --gres and --time on
# the command line where they override the directives below. #SBATCH lines are literal
# text and cannot read environment variables, so anything baked in here is right for
# exactly one cluster. No --account default is set on purpose: a missing one prints
# Slurm's own error, a wrong one fails silently into somebody else's allocation.
#SBATCH --job-name=colosseum
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# Slurm copies the batch script into a spool directory before running it, so
# "$(dirname "$0")" points at the spool and not at the repo. SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from — the repo root.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${REPO}"
# shellcheck disable=SC1091
source "${REPO}/scripts/infra/slurm/job_environment.sh"

CONTROL="LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
TREATMENT="LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch"

EXPERIMENTS="${EXPERIMENTS:-collusion single cooperation}"

echo "=== Colosseum multi-agent study ==="
echo "node       : $(hostname)"
echo "job        : ${SLURM_JOB_ID:-<none>}"
echo "experiments: ${EXPERIMENTS}"
echo "control    : ${CONTROL}"
echo "treatment  : ${TREATMENT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo

# Qwen3.6-27B needs an 80GB card. The repo's vLLM passes no --tensor-parallel-size
# (src/infra/runpod.py documents why), so one server is one GPU and a 48GB L40S cannot
# hold ~54GB of bf16 weights. Catching that here costs seconds; discovering it after the
# queue costs the whole allocation.
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
if [ "${GPU_MEM_MIB:-0}" -lt 70000 ]; then
    echo "ERROR: this GPU reports ${GPU_MEM_MIB} MiB; Qwen3.6-27B needs ~80GB to serve." >&2
    echo "       Submit with --gres=gpu:h100:1 (L40S is 48GB and will not fit)." >&2
    exit 1
fi

# Every invocation runs with --no-push: this node has no route to the Hub. The run
# directories it leaves under output/ are published afterwards from a login node by
# scripts/infra/slurm/publish_runs.sh, which is also where the judge pass runs.
COMMON=(--name colosseum_jira --no-push --target "${CONTROL}" "${TREATMENT}")

# Seeds per experiment. The two multi-agent experiments carry the design's 40; the
# cooperation control carries 20, because it is a sanity check on a single cell rather
# than a contrast whose effect size has to be resolved. An explicit SEEDS= overrides both
# (SEEDS=1-3 is the smoke shape).
seeds_for() {
    if [ -n "${SEEDS:-}" ]; then
        echo "seeds=[$(python3 -c "
import sys
lo, _, hi = sys.argv[1].partition('-')
print(','.join(str(s) for s in range(int(lo), int(hi or lo) + 1)))" "${SEEDS}")]"
    elif [ "$1" = "cooperation" ]; then
        echo "seeds=[$(seq -s, 1 20)]"
    fi
}

STATUS=0
for experiment in ${EXPERIMENTS}; do
    echo
    echo "=== experiment: ${experiment} ($(date -u +%H:%M:%S)) ==="
    SEED_ARG="$(seeds_for "${experiment}")"
    # `set -e` would abort the remaining experiments on the first failure. One experiment
    # failing is not a reason to throw away the two that would have succeeded in the same
    # already-paid-for allocation, so the status is collected and reported at the end.
    set +e
    uv run evals "${COMMON[@]}" "experiment=${experiment}" ${SEED_ARG} ${EXTRA:-}
    rc=$?
    set -e
    [ ${rc} -ne 0 ] && { echo "!!! ${experiment} exited ${rc}"; STATUS=${rc}; }
done

echo
echo "=== run directories ==="
find output/colosseum_jira -maxdepth 1 -mindepth 1 -type d -newermt "-1 day" 2>/dev/null | sort || true
echo
echo "Next, ON A LOGIN NODE (this one has no network):"
echo "  bash scripts/infra/slurm/publish_runs.sh"

exit ${STATUS}
