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
# `module` is a shell FUNCTION from the login profile, so it does not exist in a
# non-login shell — and `ssh host '<cmd>'` gives you a non-login shell. Guarding the
# block with `command -v module` and continuing therefore built the venv on
# /usr/bin/python instead, whose headers are incomplete: vLLM's inductor pass compiles
# C++ at engine startup and died on
#   /usr/include/python3.12/pyconfig.h: fatal error:
#       x86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory
# after loading 52 GiB of weights. So this is now a hard failure with the remedy in it.
if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: the module command is not available, so this shell is not a login" >&2
    echo "       shell and the cluster's python/cuda cannot be loaded. Building the" >&2
    echo "       venv on /usr/bin/python produces one whose headers are incomplete," >&2
    echo "       and vLLM only discovers that after loading 50+GB of weights." >&2
    echo "       Re-run under a login shell:" >&2
    echo "         ssh killarney 'bash -lc \"cd ${REPO} && bash scripts/infra/slurm/setup_killarney.sh\"'" >&2
    exit 1
fi
module load StdEnv/2023
module load python/3.12.4 || module load python/3.12
module load cuda/12.6 >/dev/null 2>&1 || module load cuda >/dev/null 2>&1 || true

if [ ! -f "${COLOSSEUM_VENV}/bin/activate" ]; then
    echo ">>> creating venv at ${COLOSSEUM_VENV} using $(command -v python)"
    python -m venv "${COLOSSEUM_VENV}"
fi

# A venv built on the wrong interpreter works for everything EXCEPT the one thing that
# matters, so it is checked rather than assumed. `home` in pyvenv.cfg is the bin
# directory of the interpreter that created it; /usr/bin means the module was not active.
VENV_HOME="$(awk -F'= *' '/^home/ {print $2}' "${COLOSSEUM_VENV}/pyvenv.cfg")"
case "${VENV_HOME}" in
    /usr/bin*|/bin*)
        echo "ERROR: the venv at ${COLOSSEUM_VENV} was built on ${VENV_HOME}/python," >&2
        echo "       the system interpreter, whose C headers are incomplete — vLLM's" >&2
        echo "       inductor pass will fail at engine startup, after loading the" >&2
        echo "       weights. Delete it and re-run this script under a LOGIN shell:" >&2
        echo "         rm -rf ${COLOSSEUM_VENV}" >&2
        exit 1
        ;;
esac
echo ">>> venv interpreter home: ${VENV_HOME}"
# shellcheck disable=SC1091
source "${COLOSSEUM_VENV}/bin/activate"
python -m pip install --upgrade pip >/dev/null

# uv resolves this repo's lock, which pins vllm 0.26 on linux — the version whose serving
# flags src/infra/endpoints/vllm.py encodes. Installing with bare pip would take whatever
# vllm resolves today and quietly change how the model under measurement is served.
# uv comes from its own installer, NOT from pip, and that is not a preference.
# Compute Canada's python module ships a pip config whose only source is a local
# wheelhouse, and its interpreter advertises ZERO manylinux tags
# (`pip debug --verbose | grep manylinux` -> nothing, platform `linux-x86_64`) — a
# deliberate choice to push users at the wheelhouse. So `pip install uv` cannot see a
# wheel from anywhere: it finds the sdist, bootstraps a whole rustup toolchain, and
# spends 20+ minutes compiling uv from source on a LOGIN node, which is also the sort of
# thing that gets processes killed. Adding --index-url does not help; --only-binary just
# turns it into a clean failure.
#
# uv itself does not inherit that strictness — it installs the manylinux wheels for
# torch/vllm into this same interpreter quite happily (verified with `uv pip install
# --dry-run vllm==0.26.0`). So: fetch the binary, use it for everything.
UV_BIN="${PROJECT_ROOT}/bin/uv"
if [ ! -x "${UV_BIN}" ]; then
    echo ">>> installing the uv binary into ${PROJECT_ROOT}/bin"
    mkdir -p "${PROJECT_ROOT}/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="${PROJECT_ROOT}/bin" sh >/dev/null
fi
echo ">>> uv $("${UV_BIN}" --version)"

# uv's cache lives on /home and the venv on /project, so hardlinking is impossible and
# every package warns as it falls back to a full copy. Say so once instead.
export UV_LINK_MODE=copy

echo ">>> installing the repo (uv, honouring uv.lock)"
UV_PROJECT_ENVIRONMENT="${COLOSSEUM_VENV}" "${UV_BIN}" sync --frozen

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

# Install Colosseum's DEPENDENCIES, not Colosseum. `pip install -e` on it fails —
#   error: Multiple top-level packages discovered in a flat-layout: ['dev', 'external',
#   'experiments']
# because setuptools refuses auto-discovery when a repo has several top-level directories
# and this one declares no package config. It does not matter: the driver is invoked as
# `python -m experiments.collusion.run` with cwd set to the checkout (see runner.py), so
# `experiments` is imported from the working directory and never needs to be installed.
#
# The dependency is read out of upstream's own pyproject rather than typed here, so the
# pinned terrarium-agents version cannot drift away from what Colosseum declares. The
# `[vllm]` extra is deliberately NOT installed: it would pull vllm 0.12.0 and its own
# torch, and the model under measurement is served by THIS repo's vllm 0.26 with the
# serving flags src/infra/endpoints/vllm.py encodes.
#
# Do NOT install Terrarium from git either: its main is 0.2.0, which reorganised the
# package (terrarium.core.*, terrarium.llm.*) while Colosseum imports the 0.1.1 layout
# (terrarium.logger, envs.dcops, llm_server.clients). PyPI's 0.1.1 is the right one.
echo ">>> installing Colosseum's dependencies (not Colosseum itself)"
mapfile -t COLOSSEUM_DEPS < <(python - "${COLOSSEUM_ROOT}/pyproject.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    print("\n".join(tomllib.load(fh)["project"]["dependencies"]))
PY
)
if [ "${#COLOSSEUM_DEPS[@]}" -eq 0 ]; then
    echo "ERROR: read no dependencies out of ${COLOSSEUM_ROOT}/pyproject.toml" >&2
    exit 1
fi
printf '    %s\n' "${COLOSSEUM_DEPS[@]}"
UV_PROJECT_ENVIRONMENT="${COLOSSEUM_VENV}" \
    "${UV_BIN}" pip install --quiet "${COLOSSEUM_DEPS[@]}"

# Prove the driver actually imports before anything downloads 54GB behind it. cwd is the
# checkout, exactly as the runner invokes it.
echo ">>> verifying the Colosseum driver imports"
( cd "${COLOSSEUM_ROOT}" && python -c "
import importlib
for mod in ('terrarium.utils', 'envs.dcops.jira_ticket.jira_ticket_env',
            'experiments.collusion.run'):
    importlib.import_module(mod)
    print(f'    ok  {mod}')
from experiments.collusion.run import _resolve_agent_llm_configs
print('    ok  per-agent model routing patch is live')
" )

# ── Stage the weights ─────────────────────────────────────────────────────────
# Compute nodes run with HF_HUB_OFFLINE=1, so a cache miss there is a hard failure with
# no way to recover. Everything the run touches is pulled now.
# Via the Python API, not a CLI module path: `python -m
# huggingface_hub.commands.huggingface_cli` is gone in current huggingface_hub (the CLI
# moved and was renamed), so invoking it that way breaks on a version bump. And the
# token comes from THIS repo's own resolution (src.huggingface.hf_token, which loads
# .env), because the adapters live in a private org — bare huggingface_hub would read
# only HF_TOKEN from the ambient environment and 401 on them.
echo ">>> staging weights into ${HF_HOME} (Qwen3.6-27B is ~54GB; be patient)"
python - "${BASE_MODEL}" "${CONTROL_ADAPTER}" "${TREATMENT_ADAPTER}" <<'PY'
import sys
from huggingface_hub import snapshot_download
from src.huggingface import hf_token

for repo in sys.argv[1:]:
    print(f"    {repo}", flush=True)
    path = snapshot_download(repo, token=hf_token())
    print(f"      -> {path}", flush=True)
PY

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
