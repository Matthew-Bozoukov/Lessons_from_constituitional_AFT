#!/bin/bash
# ABOUTME: One-command prep of a fresh GPU host for --server mode: install uv, clone this
# ABOUTME: repo at the driver's current branch, uv sync. Usage: bootstrap_pod.sh <ssh-alias> [branch]
set -euo pipefail

HOST="${1:?usage: bootstrap_pod.sh <ssh-alias> [branch]}"
BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"
REPO_URL="$(git remote get-url origin)"
WORKDIR="/root/work"

echo "=== bootstrapping ${HOST}: ${REPO_URL} @ ${BRANCH} -> ${WORKDIR}"
ssh "$HOST" 'export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if [ -d '"$WORKDIR"'/.git ]; then
    cd '"$WORKDIR"' && git fetch -q origin && git checkout -q '"$BRANCH"' && git pull -q
  else
    git clone -q --branch '"$BRANCH"' '"$REPO_URL"' '"$WORKDIR"'
  fi
  cd '"$WORKDIR"' && uv sync
  echo "=== ready: $(git -C '"$WORKDIR"' log --oneline -1)"'
echo "=== done. Optional: provision the HF token with run_eval.py --server ${HOST} --push-env,"
echo "=== or scp your own .env to ${HOST}:${WORKDIR}/.env"
