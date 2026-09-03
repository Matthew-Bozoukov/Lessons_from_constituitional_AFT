# ABOUTME: One-off migration of a synth HF repo to the synth->mixture contract layout:
# ABOUTME: stage snapshots move from the repo root into stages/, uploaded from the local
# ABOUTME: run dir (authoritative), root strays deleted, all in ONE commit.
#
# Usage:
#   uv run python scratch/migrate_synth_repo_layout.py \
#       --run_dir output/courtroom/<ts> --repo LASR-Callum/2026-08-14-courtroom
#
# Needed for a run that STARTED under the pre-contract engine (stage files pushed to the
# repo root) and finished under the contract engine (README `configs:` front-matter
# declares stages/<file> paths for every stage): until the files move, those configs
# dangle. Runs entirely from the local run dir, so it never downloads from the repo.
# Idempotent: re-running uploads the same bytes and re-deletes nothing.

from __future__ import annotations

from pathlib import Path

import fire


def main(run_dir: str, repo: str) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    from src.infra.huggingface import hf_token

    rd = Path(run_dir)
    stage_files = sorted(rd.glob("stage_*.jsonl"))
    assert stage_files, f"no stage_*.jsonl in {rd}"

    api = HfApi(token=hf_token())
    remote = set(api.list_repo_files(repo, repo_type="dataset"))

    ops = []
    for f in stage_files:
        if f"stages/{f.name}" not in remote:
            ops.append(CommitOperationAdd(path_in_repo=f"stages/{f.name}",
                                          path_or_fileobj=str(f)))
    for name in sorted(remote):
        if name.startswith("stage_") and name.endswith(".jsonl"):
            ops.append(CommitOperationDelete(path_in_repo=name))

    if not ops:
        print(f"{repo}: already conformant — nothing to do")
        return
    adds = sum(isinstance(o, CommitOperationAdd) for o in ops)
    api.create_commit(
        repo_id=repo, repo_type="dataset", operations=ops,
        commit_message=f"Migrate to the synth->mixture layout: {adds} snapshots into "
                       f"stages/, {len(ops) - adds} root strays removed")
    print(f"{repo}: moved {adds} snapshots into stages/, "
          f"deleted {len(ops) - adds} root copies")


if __name__ == "__main__":
    fire.Fire(main)
