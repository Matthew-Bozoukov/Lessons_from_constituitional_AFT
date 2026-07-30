#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for ODCV-Bench agent rollouts; thin shim over src.eval.misalignment.odcv_rollout.main.
import fire

from src.eval.misalignment.odcv_rollout import main

if __name__ == "__main__":
    fire.Fire(main)
