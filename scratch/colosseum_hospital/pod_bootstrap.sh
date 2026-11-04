#!/bin/bash
# ABOUTME: One-shot setup of a `runpod up --eval` pod for colosseum_hospital: this repo at a
# ABOUTME: commit, uv sync, judge key, the managed Colosseum harness built, the scripted smoke.
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

ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "${ADDR}" \
  "REPO_URL='${REPO_URL}' BRANCH='${BRANCH}' SHA='${SHA}' \
   OPENROUTER_KEY='${OPENROUTER_API_KEY:-}' bash -s" <<'REMOTE'
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
# WITHOUT causal-conv1d: it is a TRAINING kernel (sequence packing), it has no wheel for
# this torch/cu13 pair, and an eval pod carries vLLM but no CUDA toolkit — so building it
# here dies on a missing /usr/local/cuda/bin/nvcc and takes the whole bootstrap with it.
# The train-pod bootstrap builds it deliberately (runpod.py KERNEL_BUILD); the hospital
# driver never packs, so skipping it leaves nothing this eval needs unresolved.
uv sync --frozen --quiet --no-install-package causal-conv1d
# `uv run` re-syncs by default, which puts causal-conv1d straight back and fails the same
# way, so every later `uv run` on this pod must use the venv as synced above. Written to
# the profile as well, because the queue arrives over its own non-interactive ssh.
export UV_NO_SYNC=1
grep -q UV_NO_SYNC /root/.bashrc || echo 'export UV_NO_SYNC=1' >> /root/.bashrc
echo 'export UV_NO_SYNC=1' > /etc/profile.d/uv_no_sync.sh
# HF_TOKEN + HF_ORG only, as `runpod up --push_env` left them in the serving workdir.
if [ -f /workspace/.env ] && [ ! -f /root/work/.env ]; then
    cp /workspace/.env /root/work/.env
fi
# The judge key. colosseum_hospital judges its own episodes as it runs (hospital/runner.py
# asserts the key before the sweep), and this pod DRIVES the eval, so the key has to be
# here — `runpod up --push_env` deliberately carries only HF_TOKEN + HF_ORG + WANDB. It is
# a real secret on a rented host: it lives only in this file, for the life of the pod.
if [ -n "${OPENROUTER_KEY:-}" ] && ! grep -q '^OPENROUTER_API_KEY=' /root/work/.env 2>/dev/null; then
    echo "OPENROUTER_API_KEY=${OPENROUTER_KEY}" >> /root/work/.env
    echo "    judge key written to /root/work/.env"
elif [ -z "${OPENROUTER_KEY:-}" ]; then
    echo "!!! no OPENROUTER_API_KEY in the launching shell: the sweep will refuse to start" >&2
fi
mkdir -p /root/work/output/logs

echo ">>> harness"
# terrarium-agents is a dependency now (uv sync above installed it). The managed harness
# (src/eval/misalignment/colosseum/harness.py) clones Colosseum at its pin, applies every
# patch, and patches a COPY of terrarium in its cache -- never site-packages, so no later
# `uv sync` can undo it. Built once per patch set; the runner would build it on first use
# anyway, this just pays the clone before the queue starts.
unset COLOSSEUM_ROOT
ROOT="$(uv run python -c 'from src.eval.misalignment.colosseum.harness import ensure_harness; print(ensure_harness().root)' | tail -1)"
echo "    built ${ROOT}"
# The env-log helpers (pull_env_logs.sh, fleet.py) read /root/colosseum/logs: upstream
# writes the inventory snapshots under its checkout's logs/. Point that path at the build.
if [ -d /root/colosseum ] && [ ! -L /root/colosseum ]; then
    mv /root/colosseum "/root/colosseum.hand-built.$(date +%s)"
fi
ln -sfn "${ROOT}" /root/colosseum

echo ">>> verify"
# The runner's own refusals, run exactly as the eval runs them.
uv run python -c "
from omegaconf import OmegaConf
from src.eval.misalignment.colosseum.hospital import runner
h = runner._harness(OmegaConf.create({}))
print('    ok  checkout', runner._colosseum_root(None, h))
print('    ok  terrarium fixes', runner._terrarium_fixes_version(h))
import vllm; print('    ok  vllm', vllm.__version__)
"
# The scripted-model smoke: one episode per harness variant, no GPU, ~1 minute.
cd /root/work
# Keep the WHOLE smoke log, print the summary lines. The grep alone hid every failure:
# a crashed smoke matches none of these patterns, so the bootstrap died silently at its
# last step with the log ending on "ok vllm" and nothing to read (2026-09-25).
uv run python scratch/colosseum_hospital/fixes_smoke.py \
    --out /root/work/output/colosseum_hospital/fixes_smoke > /root/work/output/logs/fixes_smoke.log 2>&1 \
    || echo "    smoke exited $?"
grep -E '^(===|  \[FAIL\]|ALL CHECKS|[0-9]+ CHECK)' /root/work/output/logs/fixes_smoke.log || true
if ! grep -q "ALL CHECKS" /root/work/output/logs/fixes_smoke.log; then
    echo ">>> fixes_smoke did not pass; last 40 lines:"
    tail -40 /root/work/output/logs/fixes_smoke.log
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo BOOTSTRAP_DONE
REMOTE
