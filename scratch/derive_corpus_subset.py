# ABOUTME: Publish a row-filtered copy of a synth corpus as its own style (e.g. `da-no-t6`), so a
# ABOUTME: mixture can consume it through the ordinary `dataset:` intake with no builder change.
"""Derive a corpus by dropping rows, and publish it under the naming law.

The mixture builder has no row filter: a source is a dataset, a budget and a balance field.
An ablation that removes one trait therefore needs a corpus that lacks it. This script takes
a published synth corpus at an exact revision, drops the rows a field matches, and publishes
the rest as `<date>-<style>-synth` with a card that pins where every row came from. Nothing
is generated; rows are byte-identical to the source.

    uv run python scratch/derive_corpus_subset.py \
        --repo dougalldeepmind/2026-09-24-da-synth --revision 4f4ca43b... \
        --style da-no-t6 --exclude trait_id=t6
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd

from src.data.synth.ours.hf_cache import dataset_card, write_jsonl
from src.infra.huggingface import gate_push, hf_api, hf_download, hf_repo_id, training_data_tags
from src.naming import synth_name, today
from src.utils import git_sha, origin_url


def field_of(row: dict, key: str):
    md = row.get("metadata") or {}
    return md.get(key, row.get(key))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="source synth repo, org/name")
    ap.add_argument("--revision", required=True, help="exact commit sha of the source")
    ap.add_argument("--style", required=True, help="style-type of the derived corpus (its name)")
    ap.add_argument("--exclude", action="append", required=True, metavar="FIELD=VALUE",
                    help="drop rows whose metadata FIELD equals VALUE; repeatable")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    rules = [tuple(x.split("=", 1)) for x in args.exclude]
    src_rows = [json.loads(l) for l in open(
        hf_download(args.repo, "dataset.jsonl", repo_type="dataset", revision=args.revision),
        encoding="utf-8")]
    manifest = json.load(open(hf_download(args.repo, "manifest.json", repo_type="dataset",
                                          revision=args.revision), encoding="utf-8"))
    kept = [r for r in src_rows if not any(str(field_of(r, f)) == v for f, v in rules)]
    dropped = len(src_rows) - len(kept)
    assert kept and dropped, f"filter kept {len(kept)} of {len(src_rows)}: nothing to derive"
    by_trait = collections.Counter(str(field_of(r, "trait_id")) for r in kept)

    name = synth_name(args.style)
    repo_id = hf_repo_id(name)
    out = Path("output/derived") / name
    constitution = str((manifest.get("config") or {}).get("constitution") or "none")
    command = "uv run python " + " ".join(sys.argv)
    card = {
        "experiment": f"{args.style}: `{args.repo}` with rows dropped where "
                      + ", ".join(f"{f}={v}" for f, v in rules)
                      + f" ({dropped} of {len(src_rows)}); rows otherwise byte-identical",
        "date_generated": today(),
        "constitution": constitution,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"inherited from {args.repo} @ {args.revision} (nothing generated here)",
        "generation_config": f"derived: filter {rules}; source manifest run_id "
                             f"{manifest.get('run_id')} git_sha {manifest.get('git_sha')}",
        "schema": "dataset.jsonl (chat rows, synth contract) + provenance.json",
        "provenance": command,
        "derived_from": f"{args.repo} @ {args.revision}",
        "rows": f"{len(kept)} kept, {dropped} dropped; per trait "
                + ", ".join(f"{t} {n}" for t, n in sorted(by_trait.items())),
    }
    gate_push(repo_id, card, what="derived corpus")
    write_jsonl(out / "dataset.jsonl", kept)
    (out / "provenance.json").write_text(json.dumps({
        "derived_from": {"repo": args.repo, "revision": args.revision,
                         "run_id": manifest.get("run_id"), "git_sha": manifest.get("git_sha")},
        "exclude": rules, "rows_in": len(src_rows), "rows_kept": len(kept),
        "rows_dropped": dropped, "per_trait": dict(sorted(by_trait.items())),
        "command": command, "git_sha": git_sha()}, indent=2))
    readme = dataset_card(card, [], True, training_data_tags("synth", args.style, constitution))
    api = hf_api()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.create_commit(
        repo_id=repo_id, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()),
                    CommitOperationAdd(path_in_repo="dataset.jsonl",
                                       path_or_fileobj=str(out / "dataset.jsonl")),
                    CommitOperationAdd(path_in_repo="provenance.json",
                                       path_or_fileobj=str(out / "provenance.json"))],
        commit_message=f"derived: {len(kept)} rows from {args.repo}@{args.revision[:8]}, "
                       f"dropped {rules}")
    sha = api.dataset_info(repo_id).sha
    print(f">>> {len(kept)} kept / {dropped} dropped; per trait {dict(sorted(by_trait.items()))}")
    print(f">>> https://huggingface.co/datasets/{repo_id} @ {sha}")


if __name__ == "__main__":
    main()
