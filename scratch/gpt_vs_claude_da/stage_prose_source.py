# ABOUTME: Stage the all-GPT da corpus (2026-11-04-da-gpt-synth) and a seeded, trait-stratified smoke subset as local
# ABOUTME: stage-6-shaped source dirs for da-gpt-prose. Run: uv run python scratch/gpt_vs_claude_da/stage_prose_source.py
"""Writes two directories `load_source_run` can read:

    data/da_gpt_full_source/     every row of dataset.jsonl at the pinned revision
    data/da_gpt_prose_smoke/     --smoke_rows rows, dealt round-robin over traits, seeded

The flattening and the directory format are scratch/da_one_sided/stage_source.py's (imported, not
copied), so the manifest carries the corpus's constitution_sha256 and load_source_run's cross-arm
assert still applies.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scratch" / "ablation"))
sys.path.insert(0, str(REPO / "scratch" / "da_one_sided"))
from build_full_source import flatten  # noqa: E402
from stage_source import write  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="dougalldeepmind/2026-11-04-da-gpt-synth")
    ap.add_argument("--revision", default=None, help="default: the repo head, resolved and pinned")
    ap.add_argument("--full_dir", default="data/da_gpt_full_source")
    ap.add_argument("--smoke_dir", default="data/da_gpt_prose_smoke")
    ap.add_argument("--smoke_rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = dotenv_values(REPO / ".env").get("HF_TOKEN")
    rev = args.revision or HfApi(token=tok).dataset_info(args.repo).sha
    get = lambda f: hf_hub_download(args.repo, f, repo_type="dataset", token=tok, revision=rev)  # noqa: E731
    manifest = json.loads(Path(get("manifest.json")).read_text(encoding="utf-8"))
    src = [json.loads(line) for line in open(get("dataset.jsonl"), encoding="utf-8")]
    flat = [f for f in (flatten(r) for r in src) if f]
    assert len(flat) == len(src), f"{len(src) - len(flat)} rows lacked a system/user/reasoning/response part"
    ids = [r["scenario_id"] for r in flat]
    assert len(set(ids)) == len(ids), "duplicate scenario ids"
    staged = {"repo": args.repo, "revision": rev, "file": "dataset.jsonl"}

    write(REPO / args.full_dir, flat, manifest, staged,
          "dataset.jsonl flattened into the stage-6 field shape; rows verbatim")

    rng = random.Random(args.seed)
    by = defaultdict(list)
    for r in flat:
        by[r["trait_id"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    pick, traits = [], sorted(by)
    while len(pick) < args.smoke_rows:
        for t in traits:
            if by[t] and len(pick) < args.smoke_rows:
                pick.append(by[t].pop())
    write(REPO / args.smoke_dir, pick, manifest, {**staged, "seed": args.seed},
          f"smoke source for configs/data/synth/da-gpt-prose.yaml: {len(pick)} rows, round-robin over traits")

    print(f">>> {args.repo} @ {rev}: {len(flat)} rows -> {args.full_dir}")
    print(f">>> constitution_sha256 {str(manifest.get('constitution_sha256'))[:16]}...")
    print(f">>> smoke {len(pick)} rows -> {args.smoke_dir}: traits {dict(Counter(r['trait_id'] for r in pick))}")


if __name__ == "__main__":
    main()
