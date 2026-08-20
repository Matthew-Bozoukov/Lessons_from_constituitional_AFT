#!/usr/bin/env bash
# ABOUTME: Prepare one rented vast VM to drive ODCV: docker, uv, the repo at a pinned sha,
# ABOUTME: a keepalive SSH tunnel to the serving pod, and a preflight that refuses a bad box.
#
# Run ON the box (scp this file over, then):
#   bash odcv_box_bootstrap.sh <git_sha> <pod_ip> <pod_ssh_port> <config> <box_id>
#
# Idempotent: safe to re-run after a partial failure or an IP remap.
#
# WHY A TUNNEL AND NOT THE HTTPS PROXY. docs/LOG.md 2026-08-09 records that RunPod's proxy
# times out on ODCV's long non-streaming rollouts, so the config points containers at
# host.docker.internal:8000 and this script is what makes that address mean anything. The
# tunnel is held by autossh with aggressive keepalives because the run outlives any single
# TCP connection and a silently dead tunnel turns every remaining scenario into a failure
# that still reports `ok`.
#
# WHY A PREFLIGHT GATE. A box whose docker cannot build, or whose checkout is missing
# scenario fixtures, produces a run that looks clean and is missing ~21% of its cells
# (docs/LOG.md 2026-08-18). Cheaper to refuse the box.

set -uo pipefail

SHA="${1:?git sha required}"
POD_IP="${2:?pod ip required}"
POD_PORT="${3:?pod ssh port required}"
CONFIG="${4:?odcv config path required}"
BOX_ID="${5:?box id required}"

REPO=https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git
WORK=/root/work
STATE=/root/odcv
mkdir -p "$STATE"

echo "=== [1/6] packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# docker.io + compose plugin: the harness shells out to `docker compose`, not `docker-compose`.
apt-get install -y -qq docker.io docker-compose-plugin git curl autossh jq >/dev/null
systemctl enable --now docker >/dev/null 2>&1 || service docker start || true
docker info >/dev/null 2>&1 || { echo "FATAL: docker unusable on this box"; exit 1; }
echo "docker OK: $(docker --version)"

echo "=== [2/6] uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv --version

echo "=== [3/6] repo @ $SHA ==="
if [ ! -d "$WORK/.git" ]; then
  git clone -q "$REPO" "$WORK"
fi
cd "$WORK"
git fetch -q origin
git checkout -q "$SHA"
echo "HEAD: $(git rev-parse HEAD)"
# Sync before the tunnel so a dependency failure is not misread as a network problem.
uv sync -q || { echo "FATAL: uv sync failed"; exit 1; }

echo "=== [4/6] ssh key for the tunnel ==="
# Generated HERE and never copied from the laptop: the box gets a key that reaches exactly
# one pod, so a compromised community box cannot reach anything else we own.
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519
fi
echo "--- THIS BOX'S PUBLIC KEY (must be in the pod's authorized_keys) ---"
cat ~/.ssh/id_ed25519.pub
echo "--------------------------------------------------------------------"

echo "=== [5/6] tunnel -> $POD_IP:$POD_PORT ==="
pkill -f "autossh.*8000:localhost:8000" 2>/dev/null || true
pkill -f "ssh -N -L 8000:localhost:8000" 2>/dev/null || true
# -M 0 plus ServerAlive*: autossh's own monitoring port is unnecessary when ssh itself is
# told to notice a dead peer within ~30s, and a monitoring port is one more thing to clash.
AUTOSSH_GATETIME=0 nohup autossh -M 0 -f -N \
  -o "StrictHostKeyChecking=accept-new" \
  -o "ServerAliveInterval=10" -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -L 8000:localhost:8000 -p "$POD_PORT" "root@$POD_IP" \
  >> "$STATE/tunnel.log" 2>&1
sleep 8
if curl -sf -m 20 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "tunnel OK; served models:"
  curl -s -m 20 http://127.0.0.1:8000/v1/models | jq -r '.data[].id' | sed 's/^/  /'
else
  echo "FATAL: tunnel up but endpoint not answering on 127.0.0.1:8000"
  tail -5 "$STATE/tunnel.log" 2>/dev/null
  exit 1
fi

echo "=== [6/6] preflight ==="
cd "$WORK"
uv run python scratch/odcv_preflight.py --config "$CONFIG" \
  --check_docker --base_url http://127.0.0.1:8000/v1 || {
    echo "FATAL: preflight failed - this box would silently drop cells"; exit 1; }

echo "$BOX_ID" > "$STATE/box_id"
echo
echo "=== BOX $BOX_ID READY ==="
echo "start the run with:"
echo "  cd $WORK && nohup uv run python scratch/odcv_box_run.py \\"
echo "    --config $CONFIG --passes 2 --box_id $BOX_ID \\"
echo "    --hf_repo <org/repo> > $STATE/supervisor.log 2>&1 &"
