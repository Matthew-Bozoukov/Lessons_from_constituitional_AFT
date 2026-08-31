# ABOUTME: Publishes the Arena-Hard rerun bundle (code + bench tree + config) to a private
# ABOUTME: HF dataset, which is how the RunPod pod receives everything it needs.

"""Publish the Arena-Hard rerun bundle.

Kept in scratch/ rather than /tmp: the staging directory and an earlier throwaway copy of
this script were both wiped by a /tmp sweep mid-session, which launched a pod against a
stale bundle.

    uv run python scratch/publish_arena_bundle.py --stage /tmp/ah_regen
"""

from __future__ import annotations

import os
from pathlib import Path

import fire
from dotenv import load_dotenv


def main(stage: str = "/tmp/ah_regen",
         repo: str = "LASR-Callum/2026-08-09-arena-hard-regen-bundle") -> None:
    """Upload the staged bundle.

    Args:
        stage: Directory holding code.tar.gz, bench.tar.gz, config.yaml and the pair scripts.
        repo: Destination HF dataset repo (private).
    """
    load_dotenv(override=True)
    from huggingface_hub import HfApi

    src = Path(stage)
    need = ["code.tar.gz", "bench.tar.gz", "config.yaml",
            "arena_hard_pair.py", "arena_hard_pair_overrides.yaml"]
    missing = [f for f in need if not (src / f).exists()]
    assert not missing, f"staging is incomplete, refusing to publish: {missing}"

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(folder_path=str(src), repo_id=repo, repo_type="dataset",
                      commit_message="arena-hard bundle")
    print(f">>> https://huggingface.co/datasets/{repo}")
    for f in sorted(src.iterdir()):
        print(f"    {f.name:<34}{f.stat().st_size/1e6:8.2f} MB")


if __name__ == "__main__":
    fire.Fire(main)
