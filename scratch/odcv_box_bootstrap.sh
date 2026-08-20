#!/usr/bin/env bash
# ABOUTME: Prepare one rented vast VM to drive ODCV: docker, uv, the repo at a pinned sha,
# ABOUTME: a systemd-supervised tunnel to the serving pod, and a preflight that refuses a bad box.
#
# Run ON the box (scp this file over, then):
#   bash odcv_box_bootstrap.sh <git_sha> <pod_ip> <pod_ssh_port> <config> <box_id>
#
# Pass pod_ip=NONE to prepare the box WITHOUT a tunnel (packages, uv, repo, keypair). Used
# for the no-GPU smoke: the box is proven end to end against a public endpoint before any
# GPU exists, which is the step whose absence cost a full day's pod time on 2026-08-19.
#
# Idempotent: safe to re-run after a partial failure or an IP remap.
#
# THREE THINGS HERE ARE SCAR TISSUE FROM 2026-08-19, and each cost real money:
#
#  1. A fresh Ubuntu VM runs unattended-upgrades, which holds /var/lib/dpkg/lock-frontend
#     for MINUTES. The original script ran `apt-get install ... >/dev/null` with no exit
#     check, so the failure was invisible and surfaced several steps later as a missing
#     binary. Every apt call now waits for the lock and fails loudly.
#  2. The tunnel is a systemd unit, not autossh and not nohup. A nohup'd process did NOT
#     survive the SSH session that started it (verified: pgrep found nothing afterwards),
#     and autossh is an extra package a held dpkg lock can silently fail to install.
#     systemd supervises, restarts on failure, and outlives every disconnect.
#  3. The preflight gate is not optional. A box whose docker cannot build produces a run
#     that reads clean and is missing ~21% of its cells (docs/LOG.md 2026-08-18).
#
# WHY A TUNNEL AND NOT THE HTTPS PROXY: docs/LOG.md 2026-08-09 records that RunPod's proxy
# times out on ODCV's long non-streaming rollouts, so the config points containers at
# host.docker.internal:8000 and this script is what makes that address mean anything.

set -uo pipefail

SHA="${1:?git sha required}"
POD_IP="${2:?pod ip required (or NONE to skip the tunnel)}"
POD_PORT="${3:?pod ssh port required (ignored when pod_ip=NONE)}"
CONFIG="${4:?odcv config path required}"
BOX_ID="${5:?box id required}"

REPO=https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT.git
WORK=/root/work
STATE=/root/odcv
mkdir -p "$STATE"

wait_dpkg() {
  for i in $(seq 1 90); do
    fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || return 0
    [ "$i" = 1 ] && echo "    waiting for unattended-upgrades to release the dpkg lock..."
    sleep 10
  done
  echo "FATAL: dpkg lock still held after 15 minutes"; exit 1
}

echo "=== [1/6] packages ==="
export DEBIAN_FRONTEND=noninteractive
wait_dpkg
apt-get update -qq || { echo "FATAL: apt-get update failed"; exit 1; }
wait_dpkg
# Install ONLY what is missing. The vast KVM image ships docker-ce preinstalled, and asking
# for docker.io on top of it makes apt fail with "pkgProblemResolver::Resolve generated
# breaks" -- the two packages conflict. So an unconditional install breaks exactly the
# boxes that were already usable.
PKGS=""
command -v docker >/dev/null 2>&1        || PKGS="$PKGS docker.io"
docker compose version >/dev/null 2>&1   || PKGS="$PKGS docker-compose-plugin"
command -v git  >/dev/null 2>&1          || PKGS="$PKGS git"
command -v curl >/dev/null 2>&1          || PKGS="$PKGS curl"
command -v jq   >/dev/null 2>&1          || PKGS="$PKGS jq"
if [ -n "$PKGS" ]; then
  echo "    installing:$PKGS"
  apt-get install -y -qq $PKGS || { echo "FATAL: apt-get install failed ($PKGS)"; exit 1; }
else
  echo "    all required packages already present"
fi
systemctl enable --now docker >/dev/null 2>&1 || service docker start || true
docker info >/dev/null 2>&1 || { echo "FATAL: docker unusable on this box"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "FATAL: docker compose plugin missing"; exit 1; }
echo "docker OK: $(docker --version | cut -c1-40) / $(docker compose version | head -1)"

echo "=== [2/6] uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null || { echo "FATAL: uv install"; exit 1; }
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q 'local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv --version

echo "=== [3/6] repo @ $SHA ==="
if [ ! -d "$WORK/.git" ]; then
  git clone -q "$REPO" "$WORK" || { echo "FATAL: clone failed"; exit 1; }
fi
cd "$WORK" || exit 1
git fetch -q origin
git checkout -q "$SHA" || { echo "FATAL: checkout $SHA failed"; exit 1; }
echo "HEAD: $(git rev-parse HEAD)"
# Sync before the tunnel so a dependency failure is not misread as a network problem.
uv sync -q || { echo "FATAL: uv sync failed"; exit 1; }

echo "=== [4/6] ssh key for the tunnel ==="
# Generated HERE and never copied from the laptop: the box gets a key that reaches exactly
# one pod, so a compromised community box cannot reach anything else we own.
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519
fi
echo "PUBKEY: $(cat ~/.ssh/id_ed25519.pub)"

if [ "$POD_IP" = "NONE" ]; then
  echo "=== [5/6] tunnel SKIPPED (pod_ip=NONE) ==="
  echo "=== [6/6] preflight (build check only, no endpoint) ==="
  uv run python scratch/odcv_preflight.py --config "$CONFIG" --check_docker \
    || { echo "FATAL: preflight failed"; exit 1; }
  echo "$BOX_ID" > "$STATE/box_id"
  echo; echo "=== BOX $BOX_ID PREPARED (no tunnel) ==="
  exit 0
fi

echo "=== [5/6] tunnel -> $POD_IP:$POD_PORT ==="
cat > /etc/systemd/system/odcv-tunnel.service <<UNIT
[Unit]
Description=ODCV vLLM tunnel to the serving pod
After=network-online.target

[Service]
ExecStart=/usr/bin/ssh -N -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 8000:localhost:8000 -p ${POD_PORT} root@${POD_IP}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable odcv-tunnel.service >/dev/null 2>&1
systemctl restart odcv-tunnel.service
sleep 10
if curl -sf -m 20 http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "tunnel OK; served models:"
  curl -s -m 20 http://127.0.0.1:8000/v1/models | jq -r '.data[].id' | sed 's/^/  /'
else
  echo "FATAL: tunnel not answering on 127.0.0.1:8000"
  systemctl status odcv-tunnel.service --no-pager -l 2>/dev/null | tail -10
  exit 1
fi

echo "=== [6/6] preflight ==="
uv run python scratch/odcv_preflight.py --config "$CONFIG" \
  --check_docker --base_url http://127.0.0.1:8000/v1 \
  || { echo "FATAL: preflight failed - this box would silently drop cells"; exit 1; }

echo "$BOX_ID" > "$STATE/box_id"
echo
echo "=== BOX $BOX_ID READY ==="
