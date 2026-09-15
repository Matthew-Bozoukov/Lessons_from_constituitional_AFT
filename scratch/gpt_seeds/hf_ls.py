# ABOUTME: Tiny HF helper for this arm: list a repo's files, or print one file, from the CLI
# ABOUTME: (the worktree Bash hook refuses inline python, so it lives here).
# Run: uv run python scratch/gpt_seeds/hf_ls.py ls <repo> [--repo_type dataset] | cat <repo> <file>
from __future__ import annotations

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download


def ls(repo: str, repo_type: str = "dataset", grep: str = "") -> None:
    load_dotenv()
    for f in HfApi().list_repo_files(repo, repo_type=repo_type):
        if grep in f:
            print(f)


def cat(repo: str, file: str, repo_type: str = "model", head: int = 0) -> None:
    load_dotenv()
    p = hf_hub_download(repo, file, repo_type=repo_type)
    text = open(p, encoding="utf-8").read()
    print(text if not head else "\n".join(text.splitlines()[:head]))


if __name__ == "__main__":
    fire.Fire({"ls": ls, "cat": cat})
