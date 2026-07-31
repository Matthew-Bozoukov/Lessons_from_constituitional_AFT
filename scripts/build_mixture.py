#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for mixing datasets to a token budget; thin shim over src.data.build_mixture.main.
import fire

from src.data.build_mixture import main

if __name__ == "__main__":
    fire.Fire(main)
