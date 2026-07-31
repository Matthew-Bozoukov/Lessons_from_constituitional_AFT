#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for aggregating agentic-misalignment harness results; thin shim over src.eval.misalignment.aggregate_eval.main.
import fire

from src.eval.misalignment.aggregate_eval import main

if __name__ == "__main__":
    fire.Fire(main)
