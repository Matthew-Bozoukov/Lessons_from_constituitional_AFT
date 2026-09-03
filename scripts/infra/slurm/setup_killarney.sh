#!/bin/bash
# ABOUTME: ONE-TIME Killarney setup, run on a LOGIN node: build the venv and stage every
# ABOUTME: byte a compute node will need — base model, both adapters, Colosseum — onto /project.
# ============================================================================
# setup_killarney.sh — the online half of the multi-agent pipeline
# ============================================================================
#
# Killarney compute nodes have NO outbound network. Anything a run downloads must
# therefore be downloaded HERE, on a login node, before the first sbatch. That split is
# the whole reason this script exists separately from the job scripts.
#
#   bash scripts/infra/slurm/setup_killarney.sh
#
# Idempotent: re-running re-verifies the venv and re-checks the cache, and skips whatever
# is already present. Budget ~30 minutes on a cold cache — Qwen3.6-27B is ~54GB.
#
# What lands where:
#   /project/aip-s2ganapa/$USER/venvs/colosseum   the environment every job activates
#   /project/aip-s2ganapa/$USER/hf_cache          base model + both LoRA adapters
#   /project/aip-s2ganapa/$USER/colosseum         the benchmark checkout
#   /scratch/$USER/colosseum                      run output (purge-eligible, pushed to HF)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

PROJECT_ROOT="${PROJECT_ROOT:-/project/aip-s2ganapa/${USER}}"
COLOSSEUM_VENV="${COLOSSEUM_VENV:-${PROJECT_ROOT}/venvs/colosseum}"
COLOSSEUM_ROOT="${COLOSSEUM_ROOT:-${PROJECT_ROOT}/colosseum}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"

BASE_MODEL="Qwen/Qwen3.6-27B"
CONTROL_ADAPTER="LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64"
TREATMENT_ADAPTER="LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch"

mkdir -p "${PROJECT_ROOT}" "${HF_HOME}" "${SCRATCH:-/scratch/${USER}}/colosseum" logs/slurm

echo "=== Killarney setup ==="
echo "repo        : ${REPO}"
echo "project root: ${PROJECT_ROOT}"
echo "HF_HOME     : ${HF_HOME}"
echo

# A login node has internet; a compute node does not. Fail here with the reason rather
# than let a job discover it two hours into a queue.
if ! curl -fsS -m 10 -o /dev/null https://huggingface.co; then
    echo "ERROR: no route to huggingface.co. Run this on a LOGIN node (klogin01..04)," >&2
    echo "       not inside an salloc/sbatch allocation." >&2
    exit 1
fi

# ── Modules, then the venv (a module load rewrites PATH and shadows the venv) ──
if command -v module >/dev/null 2>&1; then
    module load StdEnv/2023 >/dev/null 2>&1 || true
    module load python/3.12.4 >/dev/null 2>&1 || module load python/3.12 >/dev/null 2>&1 || true
    module load cuda/12.6 >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true
fi

if [ ! -f "${COLOSSEUM_VENV}/bin/activate" ]; then
    echo ">>> creating venv at ${COLOSSEUM_VENV}"
    python -m venv "${COLOSSEUM_VENV}"
fi
# shellcheck disable=SC1091
source "${COLOSSEUM_VENV}/bin/activate"
python -m pip install --upgrade pip >/dev/null

# uv resolves this repo's lock, which pins vllm 0.26 on linux — the version whose serving
# flags src/infra/endpoints/vllm.py encodes. Installing with bare pip would take whatever
# vllm resolves today and quietly change how the model under measurement is served.
echo ">>> installing the repo (uv, honouring uv.lock)"
python -m pip install --quiet uv
UV_PROJECT_ENVIRONMENT="${COLOSSEUM_VENV}" python -m uv sync --frozen

# ── Colosseum ─────────────────────────────────────────────────────────────────
# Pinned to a commit, not a branch: the benchmark defines the environment we measure in,
# so "which Colosseum" is part of the experiment's identity and belongs in the record.
COLOSSEUM_GIT="${COLOSSEUM_GIT:-https://github.com/umass-ai-safety/colosseum}"
COLOSSEUM_REF="${COLOSSEUM_REF:-ac0b405}"
PATCH="${REPO}/src/eval/misalignment/colosseum/third_party/per_agent_models.patch"
if [ ! -d "${COLOSSEUM_ROOT}/.git" ]; then
    echo ">>> cloning Colosseum into ${COLOSSEUM_ROOT}"
    git clone "${COLOSSEUM_GIT}" "${COLOSSEUM_ROOT}"
fi
git -C "${COLOSSEUM_ROOT}" fetch --all --quiet
# Hard reset, not checkout: this tree is patched below, and a re-run must start from the
# pinned upstream commit or the patch stacks on itself.
git -C "${COLOSSEUM_ROOT}" checkout --quiet --force "${COLOSSEUM_REF}"
git -C "${COLOSSEUM_ROOT}" reset --hard --quiet "${COLOSSEUM_REF}"
echo ">>> Colosseum at $(git -C "${COLOSSEUM_ROOT}" rev-parse --short HEAD)"

# Six agents in one run must be able to hold DIFFERENT checkpoints; upstream cannot
# express that. See third_party/README.md for what the patch does and why it is shaped to
# merge with the authors' unreleased branch rather than fork away from it.
echo ">>> applying per-agent model routing patch"
git -C "${COLOSSEUM_ROOT}" apply "${PATCH}"
git -C "${COLOSSEUM_ROOT}" diff --stat

# terrarium-agents is pinned to 0.1.1 in Colosseum's own lock. Do NOT install Terrarium
# from git: its main is 0.2.0, which reorganised the package (terrarium.core.*,
# terrarium.llm.*) and Colosseum imports the 0.1.1 layout (terrarium.logger, envs.dcops,
# llm_server.clients). Editable-installing the repo pulls 0.1.1 from PyPI, which is right.
UV_PROJECT_ENVIRONMENT="${COLOSSEUM_VENV}" python -m uv pip install --quiet -e "${COLOSSEUM_ROOT}"

# ── Stage the weights ─────────────────────────────────────────────────────────
# Compute nodes run with HF_HUB_OFFLINE=1, so a cache miss there is a hard failure with
# no way to recover. Everything the run touches is pulled now.
echo ">>> staging weights into ${HF_HOME} (Qwen3.6-27B is ~54GB; be patient)"
for repo in "${BASE_MODEL}" "${CONTROL_ADAPTER}" "${TREATMENT_ADAPTER}"; do
    echo "    ${repo}"
    python -m huggingface_hub.commands.huggingface_cli download "${repo}" >/dev/null
done

# The chat template is read off the BASE model's tokenizer_config.json at serve time and
# rewritten for thinking mode (pin_template). That read must hit the cache too — verify
# it now rather than let vLLM fail on the node.
echo ">>> verifying the offline path resolves both arms"
HF_HUB_OFFLINE=1 python - <<PY
from src.infra.endpoints.vllm import resolve_target
for path in ["${CONTROL_ADAPTER}", "${TREATMENT_ADAPTER}"]:
    s = resolve_target(path)
    print(f"    {s.model_key}: base={s.base_model} mode={s.mode} rank={s.lora_rank}")
PY

echo
echo "=== ready ==="
echo "Next: sbatch --account=aip-s2ganapa --gres=gpu:h100:1 \\"
echo "        scripts/infra/slurm/colosseum_job.sh"
