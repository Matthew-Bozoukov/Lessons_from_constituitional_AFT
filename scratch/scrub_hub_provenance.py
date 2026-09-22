# ABOUTME: One-off: rewrite the provenance already published to the Hub so no card, run_meta or harness
# ABOUTME: log carries a venv path, a home directory or a pod's live SSH endpoint (2026-09-22).

"""Run: uv run python scratch/scrub_hub_provenance.py [--dry_run=True]

Every eval dataset repo and adapter repo of the org: README.md, metadata/*.json and
metadata/harness_logs/*.log (datasets) and training_meta.json (models). The rewrite is the
same one `src.eval.run_eval.public_command` now applies at publish time:

    /<anything>/.venv/bin/evals   -> uv run evals
    /<anything>/.venv/bin/train   -> uv run train
    --server root@<ip>:<port>     -> --server <pod>      (the pod id stays in the pod fields)
    root@<ip>:<port>              -> <pod>
    /Users/<name>/.../lasr/       -> (repo-relative)
    /root/work/                   -> (repo-relative)

Files that match nothing are left untouched; one commit per repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import fire
from huggingface_hub import HfApi, hf_hub_download

ORG = "dougalldeepmind"
RULES = [
    (re.compile(r"[^\s\"'|]*/\.venv/bin/evals\b"), "uv run evals"),
    (re.compile(r"[^\s\"'|]*/\.venv/bin/train\b"), "uv run train"),
    (re.compile(r"--server(=| )root@[0-9.]+:[0-9]+"), r"--server\1<pod>"),
    (re.compile(r"root@[0-9.]+:[0-9]+"), "<pod>"),
    (re.compile(r"/Users/[^/\s\"']+/[^\s\"']*?/lasr/"), ""),
    (re.compile(r"/root/work/"), ""),
]
NEEDLE = re.compile(r"\.venv/bin/(evals|train)|root@[0-9.]+:[0-9]+|/Users/|/root/work/")


def scrub(text: str) -> str:
    for pat, rep in RULES:
        text = pat.sub(rep, text)
    return text


def main(dry_run: bool = False, limit: int = 0) -> None:
    api = HfApi()
    datasets = sorted(d.id for d in api.list_datasets(author=ORG)
                      if re.search(r"-(odcv|mask|am|colosseum|psychosis|mmlu|arena|moralbench|ctfish)-", d.id)
                      or d.id.endswith("-eval"))
    models = sorted(m.id for m in api.list_models(author=ORG))
    repos = [(r, "dataset") for r in datasets] + [(r, "model") for r in models]
    if limit:
        repos = repos[:limit]
    print(f"{len(datasets)} eval datasets + {len(models)} models", flush=True)
    changed_repos = 0
    for repo, rt in repos:
        try:
            files = api.list_repo_files(repo, repo_type=rt)
        except Exception as e:  # noqa: BLE001
            print(f"!! {repo}: {str(e)[:80]}")
            continue
        targets = [f for f in files if f == "README.md" or f == "training_meta.json"
                   or (f.startswith("metadata/") and f.endswith((".json", ".log", ".yaml", ".txt")))]
        edits = []
        for f in targets:
            try:
                path = hf_hub_download(repo, f, repo_type=rt)
            except Exception as e:  # noqa: BLE001
                print(f"!! {repo}/{f}: {str(e)[:80]}")
                continue
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
            if not NEEDLE.search(raw):
                continue
            new = scrub(raw)
            if new != raw:
                edits.append((f, new))
        if not edits:
            continue
        changed_repos += 1
        print(f"{repo}: {[f for f, _ in edits]}", flush=True)
        if dry_run:
            continue
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=new.encode("utf-8")) for f, new in edits]
        api.create_commit(repo_id=repo, repo_type=rt, operations=ops,
                          commit_message="provenance: no venv path, home directory or pod SSH endpoint (public form of the launch command)")
    print(f"{'would change' if dry_run else 'changed'} {changed_repos} repos", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
