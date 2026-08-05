#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for mixing datasets at per-source budgets (token- or example-share); thin shim over src.data.mixture.build_mixture.main.
import fire

from src.data.mixture.build_mixture import main

if __name__ == "__main__":
    fire.Fire(main)
