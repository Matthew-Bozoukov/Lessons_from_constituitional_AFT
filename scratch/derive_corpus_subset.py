# ABOUTME: Publish a row-filtered copy of a synth corpus as its own style (e.g. `da-no-t6`), optionally
# ABOUTME: splicing in a donor corpus's rows for the dropped slice (`da-new-t6`), with no builder change.
"""Derive a corpus by dropping rows (and optionally replacing them), and publish it.

The mixture builder has no row filter: a source is a dataset, a budget and a balance field.
An ablation that removes one trait, or swaps one trait's rows for a regeneration, therefore
needs a corpus that already has that shape. This script takes a published synth corpus at
an exact revision, drops the rows a field matches, optionally appends the rows matching the
same rule from a DONOR corpus, and publishes the result as `<date>-<style>-synth` with a
card that pins where every row came from. Nothing is generated; rows are byte-identical.

    # drop t6
    uv run python scratch/derive_corpus_subset.py \
        --repo dougalldeepmind/2026-09-24-da-synth --revision 4f4ca43b... \
        --style da-no-t6 --exclude trait_id=t6
    # replace t6 with the t6 rows of a newer corpus
    uv run python scratch/derive_corpus_subset.py \
        --repo dougalldeepmind/2026-09-24-da-synth --revision 4f4ca43b... \
        --style da-new-t6 --exclude trait_id=t6 \
        --replace-from dougalldeepmind/2026-09-25-da-synth --replace-revision <sha>
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


def read_rows(repo: str, revision: str) -> list[dict]:
    p = hf_download(repo, "dataset.jsonl", repo_type="dataset", revision=revision)
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def read_manifest(repo: str, revision: str) -> dict:
    p = hf_download(repo, "manifest.json", repo_type="dataset", revision=revision)
    return json.load(open(p, encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="base synth repo, org/name")
    ap.add_argument("--revision", required=True, help="exact commit sha of the base")
    ap.add_argument("--style", required=True, help="style-type of the derived corpus (its name)")
    ap.add_argument("--exclude", action="append", required=True, metavar="FIELD=VALUE",
                    help="drop base rows whose metadata FIELD equals VALUE; repeatable")
    ap.add_argument("--replace-from", help="donor synth repo whose rows MATCHING the rule are appended")
    ap.add_argument("--replace-revision", help="exact commit sha of the donor")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    assert bool(args.replace_from) == bool(args.replace_revision), "--replace-from needs --replace-revision"

    rules = [tuple(x.split("=", 1)) for x in args.exclude]
    matches = lambda r: any(str(field_of(r, f)) == v for f, v in rules)  # noqa: E731
    src_rows = read_rows(args.repo, args.revision)
    manifest = read_manifest(args.repo, args.revision)
    kept = [r for r in src_rows if not matches(r)]
    dropped = len(src_rows) - len(kept)
    assert kept and dropped, f"filter kept {len(kept)} of {len(src_rows)}: nothing to derive"

    added: list[dict] = []
    donor_manifest: dict = {}
    if args.replace_from:
        donor_rows = read_rows(args.replace_from, args.replace_revision)
        donor_manifest = read_manifest(args.replace_from, args.replace_revision)
        added = [r for r in donor_rows if matches(r)]
        assert added, f"{args.replace_from} has no rows matching {rules}"
        base_ids = {str(field_of(r, "scenario_id")) for r in kept}
        clash = [str(field_of(r, "scenario_id")) for r in added if str(field_of(r, "scenario_id")) in base_ids]
        assert not clash, f"{len(clash)} donor ids collide with base ids ({clash[0]} ...)"
    final = kept + added
    by_trait = collections.Counter(str(field_of(r, "trait_id")) for r in final)

    name = synth_name(args.style)
    repo_id = hf_repo_id(name)
    out = Path("output/derived") / name
    constitution = str((manifest.get("config") or {}).get("constitution") or "none")
    if donor_manifest:
        dc = str((donor_manifest.get("config") or {}).get("constitution") or "none")
        assert dc == constitution, f"donor constitution {dc} != base {constitution}"
    command = "uv run python " + " ".join(sys.argv)
    rule_txt = ", ".join(f"{f}={v}" for f, v in rules)
    experiment = (f"{args.style}: `{args.repo}` with rows dropped where {rule_txt} "
                  f"({dropped} of {len(src_rows)})")
    if added:
        experiment += (f", replaced by the {len(added)} rows matching the same rule from "
                       f"`{args.replace_from}`; rows otherwise byte-identical to their sources")
    card = {
        "experiment": experiment,
        "date_generated": today(),
        "constitution": constitution,
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": f"inherited from {args.repo} @ {args.revision}"
                  + (f" and {args.replace_from} @ {args.replace_revision}" if added else "")
                  + " (nothing generated here)",
        "generation_config": f"derived: filter {rules}; base manifest run_id "
                             f"{manifest.get('run_id')} git_sha {manifest.get('git_sha')}"
                             + (f"; donor manifest run_id {donor_manifest.get('run_id')} git_sha "
                                f"{donor_manifest.get('git_sha')}" if added else ""),
        "schema": "dataset.jsonl (chat rows, synth contract) + provenance.json",
        "provenance": command,
        "derived_from": f"{args.repo} @ {args.revision}"
                        + (f"; donor {args.replace_from} @ {args.replace_revision}" if added else ""),
        "rows": f"{len(kept)} kept, {dropped} dropped, {len(added)} added; per trait "
                + ", ".join(f"{t} {n}" for t, n in sorted(by_trait.items())),
    }
    gate_push(repo_id, card, what="derived corpus")
    write_jsonl(out / "dataset.jsonl", final)
    (out / "provenance.json").write_text(json.dumps({
        "derived_from": {"repo": args.repo, "revision": args.revision,
                         "run_id": manifest.get("run_id"), "git_sha": manifest.get("git_sha")},
        "donor": ({"repo": args.replace_from, "revision": args.replace_revision,
                   "run_id": donor_manifest.get("run_id"), "git_sha": donor_manifest.get("git_sha")}
                  if added else None),
        "exclude": rules, "rows_in": len(src_rows), "rows_kept": len(kept),
        "rows_dropped": dropped, "rows_added": len(added), "rows_out": len(final),
        "per_trait": dict(sorted(by_trait.items())),
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
        commit_message=f"derived: {len(final)} rows ({len(kept)} base + {len(added)} donor), "
                       f"dropped {rules}")
    sha = api.dataset_info(repo_id).sha
    print(f">>> {len(kept)} kept / {dropped} dropped / {len(added)} added; per trait "
          f"{dict(sorted(by_trait.items()))}")
    print(f">>> https://huggingface.co/datasets/{repo_id} @ {sha}")


if __name__ == "__main__":
    main()
