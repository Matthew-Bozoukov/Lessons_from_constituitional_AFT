#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for adding think traces to the SFT set; thin shim over src.train.augment_thinking.main.
import fire

from src.train.augment_thinking import main

if __name__ == "__main__":
    fire.Fire(main)
