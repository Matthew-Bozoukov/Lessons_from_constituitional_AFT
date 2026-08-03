#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for DPO training; thin shim over src.train.train_dpo.main.
import fire

from src.train.train_dpo import main

if __name__ == "__main__":
    fire.Fire(main)
