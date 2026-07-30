#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for difficult-advice SFT data generation; thin shim over src.train.generate_difficult_advice.main.
import fire

from src.train.generate_difficult_advice import main

if __name__ == "__main__":
    fire.Fire(main)
