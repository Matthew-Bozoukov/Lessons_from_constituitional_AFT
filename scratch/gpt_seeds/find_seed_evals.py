# ABOUTME: List ODCV eval repos on HF whose name mentions a training seed, so the seed-mean plot can
# ABOUTME: pick up grok/Sonnet seed replicates the moment a teammate publishes them.
# Run: uv run python scratch/gpt_seeds/find_seed_evals.py
from __future__ import annotations

import re

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi

PAT = re.compile(r"seed|s42|s69", re.I)


def main(authors: str = "LASR-Callum,matboz") -> None:
    load_dotenv()
    api = HfApi()
    for author in [a.strip() for a in authors.split(",") if a.strip()]:
        for d in api.list_datasets(author=author, limit=1000):
            if PAT.search(d.id) and "odcv" in d.id.lower():
                print(d.id, d.last_modified.date() if d.last_modified else "")


if __name__ == "__main__":
    fire.Fire(main)
