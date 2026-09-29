# ABOUTME: Writes the unused tool lists the DA+tools arm trained with (one per DA row that has tools) to
# ABOUTME: a JSON file for MASK's `inject_tools`. Run: uv run python -m scratch.canary.chat_tools
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
REPO, REV = "2026-09-29-da-tools-15-canary-mix", "28219f4d"
OUT = Path("output/canary/chat_tools.json")


def main() -> None:
    path = hf_hub_download(
        f"{os.environ['HF_ORG']}/{REPO}",
        "mixture.jsonl",
        repo_type="dataset",
        revision=REV,
    )
    lists = [
        row["tools"]
        for row in map(json.loads, open(path))
        if row["source"] == "da-tools" and row.get("tools")
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lists))
    print(
        f"{len(lists)} tool lists ({sum(map(len, lists))} tools) from {REPO} @ {REV} -> {OUT}"
    )


if __name__ == "__main__":
    main()
