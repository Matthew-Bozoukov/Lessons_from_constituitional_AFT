#!/usr/bin/env bash
# ABOUTME: Install the durable, explicitly authorized two-wave plain Qwen campaign queue.
# ABOUTME: Does not start it; run only after fresh CPU and source qualification succeeds.
set -euo pipefail
cd /srv/lasr/repo
test -x scratch/swebench_cpu_env/.venv/bin/python
test -f scratch/swebench_plain_campaign/manifest.json
test -f /srv/lasr/credentials.env
if systemctl is-active --quiet lasr-swebench-plain-queue.service; then
  echo 'Queue is active; refusing to replace its service.' >&2
  exit 2
fi
cat > /etc/systemd/system/lasr-swebench-plain-queue.service <<'UNIT'
[Unit]
Description=Two paired SWE-bench Lite waves for the four plain Qwen LoRAs
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service
StartLimitIntervalSec=3600
StartLimitBurst=4

[Service]
Type=simple
WorkingDirectory=/srv/lasr/repo
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=/srv/lasr/cache/huggingface
Environment=UV_CACHE_DIR=/srv/lasr/cache/uv
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/srv/lasr/credentials.env
ExecStart=/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python -m scratch.swebench_plain_campaign.queue --manifest /srv/lasr/repo/scratch/swebench_plain_campaign/manifest.json --state /srv/lasr/runs/plain-four-20261007-queue.json
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=30
TimeoutStopSec=30
StandardOutput=append:/srv/lasr/runs/plain-four-20261007-queue.log
StandardError=append:/srv/lasr/runs/plain-four-20261007-queue.log

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
echo 'Queue installed but not started. Qualification must pass before explicit start.'
