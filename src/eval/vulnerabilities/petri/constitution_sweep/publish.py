# ABOUTME: Uploads an export directory to Hugging Face as a public dataset repo.
# ABOUTME: Reads HF_TOKEN from the root .env; never prints or logs the value.
"""Publish a constitution-sweep export to the Hub.

The Hub copy is canonical: this repository keeps the code, the configs and the
writeup, and the bulk payload (transcript shards, the 4.3MB JSONL, the manifest)
lives here instead. The dashboard entry pins the revision this returns, so a
later re-upload cannot silently change the numbers under a published writeup.

Usage:
    set -a; . ./.env; set +a
    python -m src.eval.vulnerabilities.petri.constitution_sweep.publish \\
        LASR-Callum/<date>-petri-constitution-dose-sweep \\
        output/petri/exports/<date>-constitution-dose-sweep \\
        "what changed in this upload"
"""

from __future__ import annotations

import argparse
import os


def publish(repo: str, folder: str, message: str) -> str:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set - source the root .env first")

    api = HfApi(token=token)
    # Public: the dashboard fetches the manifest and shards straight from the CDN
    # with no credential, so a private repo would render as an unavailable entry.
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    info = api.upload_folder(
        folder_path=folder, repo_id=repo, repo_type="dataset", commit_message=message
    )

    revision = getattr(info, "oid", None) or api.list_repo_commits(
        repo, repo_type="dataset"
    )[0].commit_id

    print("uploaded:", f"https://huggingface.co/datasets/{repo}")
    print("revision:", revision)
    files = sorted(api.list_repo_files(repo, repo_type="dataset"))
    print(f"files   : {len(files)}")
    for f in files:
        if not f.startswith("transcripts/"):
            print("  ", f)
    shards = sum(1 for f in files if f.startswith("transcripts/"))
    if shards:
        print(f"   transcripts/*.json  x{shards}")
    return revision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo_id")
    ap.add_argument("folder")
    ap.add_argument("message", help="commit message for the Hub revision")
    a = ap.parse_args()
    publish(a.repo_id, a.folder, a.message)


if __name__ == "__main__":
    main()
