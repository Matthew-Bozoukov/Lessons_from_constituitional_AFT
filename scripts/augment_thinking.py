#!/usr/bin/env python3
# ABOUTME: Pipeline CLI for adding think traces to the SFT set; thin shim over src.data.augment_thinking.main.
import sys
from pathlib import Path

import fire

# Entry-point bootstrap: make src/ importable without an installed package,
# e.g. on the GPU box where deps are plain pip and nothing is pip-installed -e.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.augment_thinking import main  # noqa: E402

if __name__ == "__main__":
    fire.Fire(main)
