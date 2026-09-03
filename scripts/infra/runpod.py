#!/usr/bin/env python3
# ABOUTME: Thin shim over src.infra.runpod's CLI: rent a GPU pod holding this repo at this
# ABOUTME: commit. Prefer `uv run runpod up ...`; this is the same thing by path.
import fire

from src.infra.runpod import down, pods, status, up

if __name__ == "__main__":
    fire.Fire({"up": up, "status": status, "pods": pods, "down": down})
