#!/usr/bin/env bash
# ABOUTME: Install the CPU-host Lite service and independent GPU reaper without starting inference.
# ABOUTME: Run as root on the prepared VM: bash /srv/lasr/repo/scratch/swebench_lite_install.sh
set -euo pipefail
cd /srv/lasr/repo
test -x scratch/swebench_cpu_env/.venv/bin/python
test -f /srv/lasr/credentials.env
cat > /etc/systemd/system/lasr-swebench-lite.service <<'UNIT'
[Unit]
Description=Durable SWE-bench Lite supervisor (explicit paid launch only)
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
Environment=LITE_CONFIG=scratch/swebench_lite.yaml
EnvironmentFile=/srv/lasr/lite-launch.env
ExecStart=/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python -m src.eval.capabilities.swebench_mini.fleet ${LITE_ACTION} --config ${LITE_CONFIG} --budget-usd ${LITE_BUDGET_USD}
Restart=on-failure
RestartPreventExitStatus=2
RestartSec=30
TimeoutStopSec=240
KillMode=control-group
LimitNOFILE=65536
StandardOutput=append:/srv/lasr/runs/lite-coordinator.log
StandardError=append:/srv/lasr/runs/lite-coordinator.log

[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/systemd/system/lasr-swebench-gpu-reaper.service <<'UNIT'
[Unit]
Description=Reap only this Lite campaign's expired or abandoned RunPod GPUs
After=network-online.target
[Service]
Type=oneshot
WorkingDirectory=/srv/lasr/repo
Environment=LITE_CONFIG=scratch/swebench_lite.yaml
EnvironmentFile=/srv/lasr/lite-launch.env
ExecStart=/srv/lasr/repo/scratch/swebench_cpu_env/.venv/bin/python -m src.eval.capabilities.swebench_mini.fleet guard --config ${LITE_CONFIG}
TimeoutStartSec=240
UNIT
cat > /etc/systemd/system/lasr-swebench-gpu-reaper.timer <<'UNIT'
[Unit]
Description=Independent Lite GPU cleanup check every minute
[Timer]
OnBootSec=60
OnUnitActiveSec=60
Persistent=true
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now lasr-swebench-gpu-reaper.timer
echo 'Installed. Inference service is NOT started or enabled at boot.'
echo 'An explicit launch requires /srv/lasr/lite-launch.env with LITE_ACTION and LITE_BUDGET_USD.'
