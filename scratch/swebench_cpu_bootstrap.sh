#!/usr/bin/env bash
# ABOUTME: Install pinned CPU-only environments and persistent services on the Vast VM.
# ABOUTME: Requires /srv/lasr/repo, receipt.json and root-only credentials.env staged over SSH.
set -euo pipefail
umask 077
cd /srv/lasr/repo
test "$(ps -p 1 -o comm=)" = systemd
chmod 600 /srv/lasr/credentials.env /srv/lasr/receipt.json
mkdir -p /srv/lasr/runs /srv/lasr/cache
if ! command -v uv >/dev/null; then
  curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.12.0/install.sh -o /srv/lasr/install-uv.sh
  env UV_INSTALL_DIR=/usr/local/bin sh /srv/lasr/install-uv.sh
fi
export UV_CACHE_DIR=/srv/lasr/cache/uv
uv sync --frozen --project scratch/swebench_cpu_env

cat > /etc/systemd/system/lasr-vast-expiry.service <<'EOF'
[Unit]
Description=Stop only the receipted Vast VM at expiry; retain disk
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
WorkingDirectory=/srv/lasr/repo
ExecStart=/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python /srv/lasr/repo/scratch/swebench_cpu_watchdog.py --receipt /srv/lasr/receipt.json --env /srv/lasr/credentials.env
TimeoutStartSec=90
EOF
cat > /etc/systemd/system/lasr-vast-expiry.timer <<'EOF'
[Unit]
Description=Check the absolute Vast STOP deadline every minute
[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
AccuracySec=1s
Unit=lasr-vast-expiry.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now lasr-vast-expiry.timer
systemctl start lasr-vast-expiry.service

uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/agent
uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/harness
if ! test -f /root/.ssh/id_ed25519; then
  ssh-keygen -q -t ed25519 -N "" -C swebench-cpu-to-runpod -f /root/.ssh/id_ed25519
fi
cat > /etc/systemd/system/lasr-swebench-prepare.service <<'EOF'
[Unit]
Description=Prepare pinned SWE-bench Lite images and prove CPU readiness
After=network-online.target docker.service lasr-vast-expiry.timer
Requires=docker.service
[Service]
Type=oneshot
WorkingDirectory=/srv/lasr/repo
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=/srv/lasr/cache/huggingface
Environment=UV_CACHE_DIR=/srv/lasr/cache/uv
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_cpu_prepare
TimeoutStartSec=6h
StandardOutput=append:/srv/lasr/runs/prepare.log
StandardError=append:/srv/lasr/runs/prepare.log
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable lasr-swebench-prepare.service
systemctl start --no-block lasr-swebench-prepare.service
systemctl list-timers lasr-vast-expiry.timer --no-pager
