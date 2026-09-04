#!/bin/bash
# ABOUTME: One-shot setup of a `runpod up --eval` pod for colosseum_hospital: this repo at a
# ABOUTME: commit, uv sync, Colosseum @ ac0b405 with both patches, its deps in the repo venv.
#
#   bash scratch/colosseum_hospital/pod_bootstrap.sh root@<ip>:<port> <branch> <sha>
#
# Idempotent. The pod already holds vLLM at /workspace/vllmenv and the weights under
# /workspace/hf (that is what `runpod up --eval` leaves); this adds the DRIVER half so the
# eval runs on the box rather than over a tunnel: a nine-agent episode ladder runs for
# hours, and a dropped SSH tunnel would lose every episode in flight.
set -euo pipefail

HOST="$1"; BRANCH="$2"; SHA="$3"
PORT="${HOST##*:}"; ADDR="${HOST%:*}"
REPO_URL="$(git remote get-url origin | sed -E 's#^git@([^:]+):#https://\1/#')"
COLOSSEUM_REF=ac0b405

ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "${ADDR}" \
  "REPO_URL='${REPO_URL}' BRANCH='${BRANCH}' SHA='${SHA}' COLOSSEUM_REF='${COLOSSEUM_REF}' bash -s" <<'REMOTE'
set -euo pipefail
export PATH=/usr/local/bin:/root/.local/bin:$PATH
export HF_HOME=/workspace/hf
echo ">>> repo"
if [ ! -d /root/work/.git ]; then
    git clone --quiet --branch "${BRANCH}" "${REPO_URL}" /root/work
fi
cd /root/work
git fetch --quiet origin "${BRANCH}"
git checkout --quiet --detach "${SHA}"
echo "    at $(git rev-parse --short HEAD)"
# The linux lock: vllm 0.26 + torch, the server run_eval launches from THIS venv.
uv sync --frozen --quiet
# HF_TOKEN + HF_ORG only, as `runpod up --push_env` left them in the serving workdir.
if [ -f /workspace/.env ] && [ ! -f /root/work/.env ]; then
    cp /workspace/.env /root/work/.env
fi
mkdir -p /root/work/output/logs

echo ">>> colosseum"
if [ ! -d /root/colosseum/.git ]; then
    git clone --quiet https://github.com/umass-ai-safety/colosseum /root/colosseum
fi
cd /root/colosseum
git checkout --quiet --force "${COLOSSEUM_REF}"
git reset --quiet --hard "${COLOSSEUM_REF}"
git apply /root/work/src/eval/misalignment/colosseum/third_party/per_agent_models.patch
git apply /root/work/src/eval/misalignment/colosseum/third_party/hospital_seating.patch
echo "    patched: $(git diff --stat | tail -1)"

echo ">>> colosseum deps into the repo venv (not the package: it has no package config)"
# The repo venv's python, not the image's python3: that one is 3.10 and has no tomllib.
DEPS="$(/root/work/.venv/bin/python -c 'import tomllib; print(" ".join(tomllib.load(open("/root/colosseum/pyproject.toml","rb"))["project"]["dependencies"]))')"
uv pip install --python /root/work/.venv/bin/python --quiet ${DEPS}

echo ">>> verify"
cd /root/colosseum
/root/work/.venv/bin/python -c "
import importlib
for mod in ('terrarium.utils', 'envs.dcops.hospital.hospital_env', 'experiments.agent_misalignment.run'):
    importlib.import_module(mod); print('    ok ', mod)
from experiments.agent_misalignment.run import _resolve_agent_llm_configs_by_seat
print('    ok  seating patch is live')
import vllm; print('    ok  vllm', vllm.__version__)
"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo BOOTSTRAP_DONE
REMOTE
