# ABOUTME: Stage the 2026-10-30 da corpus (all 2,001 rows) and a seeded, trait-stratified smoke subset
# ABOUTME: as local stage-6-shaped source dirs for da-one-sided. Run: uv run python scratch/da_one_sided/stage_source.py
"""Writes three directories `load_source_run` can read:

    data/da_1030_full_source/        every row of dataset.jsonl at the pinned revision
    data/da_1030_one_sided_smoke/    --smoke_rows rows, dealt round-robin over traits, seeded
    data/da_one_sided_source/        the full-run source: da-15's own 650 scenarios + a trait-balanced top-up

The flattening is scratch/ablation/build_full_source.py's (imported, not copied). The manifest
carries the corpus's constitution_sha256, so load_source_run's cross-arm assert still applies.
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
from build_full_source import SNAPSHOT, flatten  # noqa: E402


def write(out: Path, rows: list[dict], manifest: dict, staged_from: dict, note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / SNAPSHOT).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "constitution_sha256": manifest.get("constitution_sha256"),
        "staged_from": staged_from, "n_records": len(rows), "note": note}, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="dougalldeepmind/2026-10-30-da-synth")
    ap.add_argument("--revision", default=None, help="default: the repo head, resolved and pinned")
    ap.add_argument("--smoke_rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="splits/2026-10-30-da-15-abc-split.json", help="the corpus's published A/B/C labels")
    ap.add_argument("--topup_per_trait", type=int, default=28, help="rows per trait added from pool B to split label A")
    args = ap.parse_args()

    env = dotenv_values(REPO / ".env")
    tok = env.get("HF_TOKEN") or env.get("HF_TOKEN_MATBOZ")
    rev = args.revision or HfApi(token=tok).dataset_info(args.repo).sha
    get = lambda f: hf_hub_download(args.repo, f, repo_type="dataset", token=tok, revision=rev)
    manifest = json.loads(Path(get("manifest.json")).read_text(encoding="utf-8"))
    src = [json.loads(l) for l in open(get("dataset.jsonl"), encoding="utf-8")]
    flat = [f for f in (flatten(r) for r in src) if f]
    assert len(flat) == len(src), f"{len(src) - len(flat)} rows lacked a system/user/reasoning/response part"
    ids = [r["scenario_id"] for r in flat]
    assert len(set(ids)) == len(ids), "duplicate scenario ids"
    staged = {"repo": args.repo, "revision": rev, "file": "dataset.jsonl"}

    write(REPO / "data/da_1030_full_source", flat, manifest, staged,
          "dataset.jsonl flattened into the stage-6 field shape; rows verbatim")

    # Smoke: round-robin over traits so every trait appears, seeded within each trait.
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
    write(REPO / "data/da_1030_one_sided_smoke", pick, manifest, {**staged, "seed": args.seed},
          f"smoke source for configs/data/synth/da-one-sided.yaml: {len(pick)} rows, round-robin over traits")

    # Full-run source for da-one-sided: the 650 scenarios 2026-10-05-da-15-mix trained on (label A of the corpus's
    # published split) plus a seeded top-up dealt evenly over traits from pool B. One-sided rows run about 0.8x DA's
    # length and the 15% share is counted in supervised tokens, so the same share needs more rows than da-15's 650.
    labels = json.loads(Path(get(args.split)).read_text(encoding="utf-8"))["labels"]
    assert set(labels) == set(ids), "the split does not label exactly this corpus"
    a_rows = [r for r in flat if labels[r["scenario_id"]] == "A"]
    rng_b = random.Random(f"da-one-sided-topup-{args.seed}")
    pool_b = defaultdict(list)
    for r in flat:
        if labels[r["scenario_id"]] == "B":
            pool_b[r["trait_id"]].append(r)
    top = []
    for t in sorted(pool_b):
        rng_b.shuffle(pool_b[t])
        top += pool_b[t][:args.topup_per_trait]
    full = a_rows + top
    write(REPO / "data/da_one_sided_source", full, manifest, {**staged, "split": args.split, "seed": args.seed},
          f"full-run source for configs/data/synth/da-one-sided.yaml: split label A ({len(a_rows)} rows, the scenarios of "
          f"2026-10-05-da-15-mix) + {len(top)} rows from pool B, {args.topup_per_trait} per trait")
    print(f">>> full-run source {len(full)} rows: A {len(a_rows)} + B top-up {len(top)}; traits {dict(Counter(r['trait_id'] for r in full))}")

    print(f">>> {args.repo} @ {rev}: {len(flat)} rows -> data/da_1030_full_source")
    print(f">>> constitution_sha256 {str(manifest.get('constitution_sha256'))[:16]}...")
    print(f">>> smoke {len(pick)} rows: traits {dict(Counter(r['trait_id'] for r in pick))}, "
          f"ai {dict(Counter(r['ai'] for r in pick))}")
    print(f">>> shortcut present {sum(bool(r.get('shortcut')) for r in flat)}/{len(flat)}, "
          f"situation present {sum(bool(r.get('situation')) for r in flat)}/{len(flat)}")


if __name__ == "__main__":
    main()
