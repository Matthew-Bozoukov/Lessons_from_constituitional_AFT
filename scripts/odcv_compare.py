#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for paired comparison of two ODCV arms; thin shim over src.eval.misalignment.odcv_compare.main.
import fire

from src.eval.misalignment.odcv_compare import main

if __name__ == "__main__":
    fire.Fire(main)
