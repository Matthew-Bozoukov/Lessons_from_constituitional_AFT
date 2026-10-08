# ABOUTME: Local Linux driver for the base-only shared-interface smoke on Docker Desktop.
# ABOUTME: Uses host Docker socket and isolated named environment volumes; no cloud rentals.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends docker.io git && rm -rf /var/lib/apt/lists/*
WORKDIR /work
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
