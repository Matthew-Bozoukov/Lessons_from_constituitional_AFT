# ABOUTME: Pulls the two matched training mixes (da-15, da-tools-15) the canary arms are built from.
# ABOUTME: Run: uv run python scratch/canary/pull_mixes.py  (writes to output/canary/mixes/)
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
ORG = os.environ["HF_ORG"]
MIXES = {
    "da-15": ("2026-09-25-da-15-mix", "73f66648"),
    "da-tools-15": ("2026-09-28-da-tools-15-mix", "41e990e1"),
}

if __name__ == "__main__":
    for arm, (repo, rev) in MIXES.items():
        path = hf_hub_download(
            f"{ORG}/{repo}",
            "mixture.jsonl",
            repo_type="dataset",
            revision=rev,
            local_dir=f"output/canary/mixes/{arm}",
        )
        print(arm, path)
