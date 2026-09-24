# ABOUTME: Hands one shard of a split agent_collusion run between pods through a private HF
# ABOUTME: dataset: `upload <run_dir> <shard>` on one pod, `download <dest> <shard>` (waits) on another.
"""Shard handoff for an agent_collusion run split across two pods.

Neither pod can reach the other, and the final publish must happen on a pod (no laptop), so
the finished shard travels through the Hub: pod A uploads its run dir under `<shard>/` plus a
`<shard>/DONE` marker; pod B polls for the marker, downloads the shard, and runs the final
`resume_from=[all shards]` pass that publishes ONE eval run. The repo is a working artifact,
not an eval run (it carries no `eval-run` tag), so the dashboard never lists it.

Run: uv run --no-sync python scratch/agent_collusion_shard_handoff.py upload <run_dir> A <repo>
     uv run --no-sync python scratch/agent_collusion_shard_handoff.py download <dest> A <repo> [wait_min]
"""

import sys
import time
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.infra.huggingface import card_markdown, gate_push, hf_api, hf_repo_id, hf_token
from src.utils import git_sha


def upload(run_dir: str, shard: str, repo: str) -> None:
    """Upload a finished shard run dir under `<shard>/`, then the DONE marker."""
    load_dotenv()
    repo_id = hf_repo_id(repo)
    fields = {
        "experiment": f"agent_collusion shard handoff between pods (shard {shard})",
        "date_generated": repo[:10],  # the name was minted as <YYYY-MM-DD>-<subject>
        "constitution": "none",
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": "dougalldeepmind/2026-09-22-qwen36-0-nosynth (both agents)",
        "generation_config": "configs/eval/agent_collusion.yaml (see each run's metadata/)",
        "schema": "<shard>/<run dir>/{rollouts,results,metadata}/ + <shard>/DONE",
        "provenance": "scratch/agent_collusion_shard_handoff.py upload <run_dir> <shard> <repo>",
    }
    gate_push(repo_id, fields, what="shard handoff")
    api = hf_api()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=card_markdown(fields).encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    src = Path(run_dir).resolve()
    api.upload_folder(folder_path=str(src), path_in_repo=f"{shard}/{src.name}",
                      repo_id=repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=src.name.encode(), path_in_repo=f"{shard}/DONE",
                    repo_id=repo_id, repo_type="dataset")
    print(f">>> uploaded shard {shard}: {src.name} -> {repo_id}", flush=True)


def download(dest: str, shard: str, repo: str, wait_min: int = 240) -> None:
    """Wait (up to wait_min) for `<shard>/DONE`, then download the shard; print its run dir."""
    load_dotenv()
    repo_id = hf_repo_id(repo)
    api = hf_api()
    deadline = time.time() + 60 * wait_min
    while True:
        files = (api.list_repo_files(repo_id, repo_type="dataset")
                 if api.repo_exists(repo_id, repo_type="dataset") else [])
        if f"{shard}/DONE" in files:
            break
        if time.time() > deadline:
            sys.exit(f"shard {shard} not uploaded to {repo_id} within {wait_min} min")
        print(f">>> waiting for shard {shard} in {repo_id} ...", flush=True)
        time.sleep(60)
    root = snapshot_download(repo_id, repo_type="dataset", allow_patterns=[f"{shard}/*"],
                             local_dir=dest, token=hf_token())
    name = (Path(root) / shard / "DONE").read_text().strip()
    print(Path(root, shard, name).resolve())


if __name__ == "__main__":
    fire.Fire({"upload": upload, "download": download})
