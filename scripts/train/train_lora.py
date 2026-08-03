#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for QLoRA SFT training; thin shim over src.train.train_lora.main.
import fire

from src.train.train_lora import main

if __name__ == "__main__":
    fire.Fire(main)
