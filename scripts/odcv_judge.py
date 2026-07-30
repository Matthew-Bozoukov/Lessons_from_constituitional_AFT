#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for ODCV-Bench judge scoring; thin shim over src.eval.misalignment.odcv_judge.main.
import fire

from src.eval.misalignment.odcv_judge import main

if __name__ == "__main__":
    fire.Fire(main)
