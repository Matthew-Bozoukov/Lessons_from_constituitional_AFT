#!/usr/bin/env bash
# ABOUTME: Prepare a local Linux-only Inspect qualification container, never a cloud host.
# ABOUTME: Run inside the task-owned Docker container with repo at /work and dedicated venv volumes.
set -euo pipefail
cd /work
apt-get update -qq
apt-get install -y -qq docker.io curl >/tmp/inspect-apt.log
curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-28.0.4.tgz -o /tmp/inspect-docker.tgz
tar -xzf /tmp/inspect-docker.tgz -C /tmp docker/docker
install /tmp/docker/docker /usr/local/bin/docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL https://github.com/docker/compose/releases/download/v2.39.4/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version
uv sync --frozen --project scratch/swebench_cpu_env
uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/inspect
uv sync --frozen --project src/eval/capabilities/swebench_mini/envs/agent
echo INSPECT_TEST_ENV_READY
