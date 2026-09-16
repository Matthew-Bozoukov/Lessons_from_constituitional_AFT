#!/bin/bash
# ABOUTME: LOGIN-node step after colosseum_job.sh: judge the episodes and push every run
# ABOUTME: dir to HF. Never run inside an allocation — compute nodes have no network.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO}"

PROJECT_ROOT="${PROJECT_ROOT:-/project/aip-s2ganapa/${USER}}"
COLOSSEUM_VENV="${COLOSSEUM_VENV:-${PROJECT_ROOT}/venvs/colosseum}"

if command -v module >/dev/null 2>&1; then
    module load StdEnv/2023 >/dev/null 2>&1 || true
    module load python/3.12.4 >/dev/null 2>&1 || module load python/3.12 >/dev/null 2>&1 || true
fi
# shellcheck disable=SC1091
source "${COLOSSEUM_VENV}/bin/activate"

# The whole point of this script is that it has a network. Say so up front rather than
# failing later on a judge call: inside an allocation both of the following are gone.
if ! curl -fsS -m 10 -o /dev/null https://openrouter.ai; then
    echo "ERROR: no route to openrouter.ai — run this on a LOGIN node (klogin01..04)," >&2
    echo "       not inside salloc/sbatch." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
# The compute-node default is 1. Judging and pushing both need the Hub.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
set -a
# shellcheck disable=SC1091
[ -f "${REPO}/.env" ] && . "${REPO}/.env"
set +a

: "${OPENROUTER_API_KEY:?not set — the judge needs it; put it in ${REPO}/.env}"
: "${HF_TOKEN:?not set — the push needs it; put it in ${REPO}/.env}"

echo "=== finishing Colosseum runs ==="
# HF_ORG from .env above names the GROUP org; the driver overrides it with its own
# --hf-org (default: the personal namespace these runs belong to) and prints the
# destination before it pushes anything. Pass --hf-org here to send them elsewhere.
python scripts/eval/publish_colosseum.py "$@"
