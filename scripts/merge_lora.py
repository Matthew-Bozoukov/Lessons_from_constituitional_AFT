#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for merging a LoRA adapter for serving; thin shim over src.train.merge_lora.main.
import fire

from src.train.merge_lora import main

if __name__ == "__main__":
    fire.Fire(main)
