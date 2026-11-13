# ABOUTME: The AgentHazard episode sandbox: unmodified Claude Code (version pinned) on a Debian
# ABOUTME: userland with the usual dev tools, run as root in /root like the paper's sandbox.
# Built by src/eval/misalignment/agenthazard/sandbox.py::build_image (uv run evals --name agenthazard).
FROM node:22-bookworm-slim

# Tools an ordinary dev box has and the benchmark's steps reach for (shell, python, git, curl,
# network and process utilities, cron, ssh client). socat is for the egress proxy container,
# which runs from this same image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl wget ca-certificates procps psmisc \
        iproute2 iputils-ping net-tools netcat-openbsd dnsutils openssh-client cron sudo \
        xxd jq less vim-tiny unzip zip file lsof socat \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION=2.1.323
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} && claude --version

# The logging hook and the settings Claude Code reads. Settings are copied into /root/.claude at
# episode start (sandbox.py), because /root is a size-capped tmpfs at run time.
COPY hook.py /opt/agenthazard/hook.py
COPY settings.json /opt/agenthazard/settings.json
RUN mkdir -p /opt/agenthazard/logs

WORKDIR /root
CMD ["sleep", "infinity"]
