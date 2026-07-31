#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for LMSYS chat-quality win-rate eval; thin shim over src.eval.capabilities.lmsys_eval.main.
import fire

from src.eval.capabilities.lmsys_eval import main

if __name__ == "__main__":
    fire.Fire(main)
