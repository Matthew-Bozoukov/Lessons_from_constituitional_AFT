#!/usr/bin/env python3
# ABOUTME: Thin CLI for src/eval/managed.py — rent a GPU, run one eval against it, tear
# ABOUTME: the pod down. Run: uv run managed --eval_name <eval> --targets <hf_path>
from src.eval.managed import cli

if __name__ == "__main__":
    cli()
