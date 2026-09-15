# ABOUTME: Shared SLURM job environment for Killarney: modules, uv venv, offline HF cache, paths.
# ABOUTME: Not run directly — sourced by every scripts/infra/slurm/*_job.sh.

# Killarney (Alliance) only. Every other host in this repo rents its GPU from RunPod
# (src/infra/runpod.py); the multi-agent work is the one exception, because Colosseum
# holds a six-seat episode open for minutes at a time and a preemptible spot pod is the
# wrong shape for that. Sourced, never executed:
#     source "${SLURM_SUBMIT_DIR}/scripts/infra/slurm/job_environment.sh"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"

# /project is the persistent, backed-up filesystem here and holds everything this pipeline
# owns: the repo (and so its output/), the venv, the model cache, the Colosseum checkout.
# $SCRATCH is deliberately unused — it is purge-eligible, and a run directory is the only
# copy of its rollouts until a login node pushes them to the Hub.
PROJECT_ROOT="${PROJECT_ROOT:-/project/aip-s2ganapa/${USER}}"
SCRATCH="${SCRATCH:-/scratch/${USER}}"
COLOSSEUM_VENV="${COLOSSEUM_VENV:-${PROJECT_ROOT}/venvs/colosseum}"
COLOSSEUM_ROOT="${COLOSSEUM_ROOT:-${PROJECT_ROOT}/colosseum}"

# ── Modules BEFORE the venv ───────────────────────────────────────────────────
# A module load rewrites PATH, so loading one after activation shadows the venv's
# python. Each module is loaded separately with fallbacks: as a single
# `module load a b c` line, one version that does not exist on this cluster makes the
# whole command fail and NOTHING loads — including python. (Lesson carried over from
# ~/projects/icrl/scripts/slurm/job_environment.sh, which learned it the hard way.)
cc_load_module() {
    for candidate in "$@"; do
        if module load "${candidate}" >/dev/null 2>&1; then
            echo "  module: ${candidate}"
            return 0
        fi
    done
    echo "  module: none of [$*] available — continuing"
    return 0
}

if command -v module >/dev/null 2>&1; then
    echo "Loading modules:"
    cc_load_module StdEnv/2023 StdEnv/2020
    cc_load_module gcc
    cc_load_module python/3.12.4 python/3.12 python
    cc_load_module cuda/12.6 cuda/12.9 cuda/12.2 cuda
fi

# ── Virtualenv ────────────────────────────────────────────────────────────────
if [ ! -f "${COLOSSEUM_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${COLOSSEUM_VENV}" >&2
    echo "       Build it ON A LOGIN NODE (compute nodes have no internet):" >&2
    echo "       bash scripts/infra/slurm/setup_killarney.sh" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${COLOSSEUM_VENV}/bin/activate"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# uv does NOT respect an activated venv: it resolves the project environment itself and
# syncs it, which on this node would mean creating `${REPO}/.venv` from scratch over a
# network that does not exist. Point it at the venv that IS activated, and forbid it from
# reaching out — so a stray `uv run` fails loudly instead of hanging on a dead socket.
# The job script calls `python -m src.eval.run_eval` directly and needs none of this;
# these are here so nothing else has to know.
export UV_PROJECT_ENVIRONMENT="${COLOSSEUM_VENV}"
export UV_OFFLINE=1
export UV_LINK_MODE=copy   # /project and uv's cache are different filesystems here

# ── Offline by construction ───────────────────────────────────────────────────
# Killarney compute nodes have NO outbound network. Everything a run needs off the
# internet — the base model, both adapters, the Colosseum package — is staged onto
# /project by setup_killarney.sh from a LOGIN node first. HF_HUB_OFFLINE makes a missing
# cache entry fail loudly here instead of hanging on a connection that cannot open.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# vLLM's usage telemetry dials home on startup and adds a stall on an offline node.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# Run output stays where run_eval puts it — `output/<eval>/<arm>_<time>/`, repo-relative,
# the same as every other eval in this repo — so publish_runs.sh and the analysis need no
# cluster-specific path. That means the repo must live on /project, not $SCRATCH, because
# $SCRATCH is purge-eligible and these directories are the only copy until they are
# pushed. Publishing and judging both need the network and both happen afterwards, from a
# login node; see scripts/infra/slurm/publish_runs.sh.
mkdir -p "${REPO}/output" "${REPO}/logs/slurm"

# .env carries OPENROUTER_API_KEY and HF_TOKEN. It is sourced because the LOGIN-node
# stages (prefetch, judge, push) need it; on a compute node nothing reads it, and no
# secret leaves this filesystem either way.
if [ -f "${REPO}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${REPO}/.env"
    set +a
fi

echo "repo    : ${REPO}"
echo "venv    : ${COLOSSEUM_VENV}"
echo "python  : $(command -v python) ($(python -V 2>&1))"
echo "HF_HOME : ${HF_HOME} (offline=${HF_HUB_OFFLINE})"
echo "out     : ${REPO}/output"

if ! python -c "import sys" >/dev/null 2>&1; then
    echo "ERROR: the venv's python does not run — the module that built it is" >&2
    echo "       probably not loaded. Check: module avail python" >&2
    return 1 2>/dev/null || exit 1
fi
