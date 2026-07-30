#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for building the DPO preference set; thin shim over src.data.generate_rejected.main.
import fire

from src.data.generate_rejected import main

if __name__ == "__main__":
    fire.Fire(main)
