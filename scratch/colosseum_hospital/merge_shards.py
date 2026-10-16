# ABOUTME: Merge seed-sharded Hospital run dirs (finished shards + a still-running full run's
# ABOUTME: finished episodes) into one run dir that rescore_autorater.py can label and harvest.
"""Union the finished episodes of several pieces of ONE Hospital cell into one run dir.

    uv run python -m scratch.colosseum_hospital.merge_shards --out <merged dir> <piece dir> [...]

Unlike merge_cells.py this accepts pieces WITHOUT results/results.json (a full run whose
sweep is still going: its finished episodes already sit in its Colosseum tree) and carries
the autorater labels the finished shards already hold (results/autorater.json, keyed by
episode dir), so the follow-up

    uv run python scratch/colosseum_hospital/rescore_autorater.py <merged dir> --config <cfg>

labels only the episodes no shard labelled (autorate_arm reuses complete prior labels),
re-runs the action measures over all of them and writes per_seed.json / episodes.json /
results.json. Later pieces win a duplicated seed. The judge's measures are NOT merged:
the headline sabotage act and its refusal/plan split need the harvest + autorater only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scratch.colosseum_hospital.merge_cells import episodes_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pieces", nargs="+", type=Path, help="run dirs, oldest first")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--seeds", default=None, help="keep only these seeds, e.g. 1-30 or 1,2,3"
    )
    args = ap.parse_args()

    keep: set[int] | None = None
    if args.seeds:
        keep = set()
        for part in args.seeds.split(","):
            a, _, b = part.partition("-")
            keep.update(range(int(a), int(b or a) + 1))

    chosen: dict[int, Path] = {}
    origin: dict[int, str] = {}
    for p in args.pieces:
        found = episodes_of(p)
        print(f"{p.name}: {len(found)} finished episodes {sorted(found)}")
        for seed, d in found.items():
            if keep is not None and seed not in keep:
                continue
            if seed in chosen:
                print(f"  seed {seed}: {origin[seed]} superseded by {p.name}")
            chosen[seed] = d
            origin[seed] = p.name
    assert chosen, "no finished episodes in any piece"
    print(f"-> {len(chosen)} episodes: {sorted(chosen)}")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    src_root = next(iter(chosen.values())).parents[3]  # .../rollouts/colosseum/<ts>
    ts_root = out / "rollouts" / "colosseum" / src_root.name
    for seed, d in sorted(chosen.items()):
        shutil.copytree(
            d, ts_root / "runs" / d.parent.parent.name / d.parent.name / d.name
        )
    # Sweep-level files beside runs/ (the sweep config and summaries) from the newest piece
    # that has a tree.
    for p in reversed(args.pieces):
        trees = sorted(p.glob("rollouts/colosseum/*"))
        if trees:
            for entry in trees[-1].iterdir():
                if entry.name == "runs":
                    continue
                if entry.is_dir():
                    shutil.copytree(entry, ts_root / entry.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(entry, ts_root / entry.name)
            break

    results = out / "results"
    results.mkdir(parents=True)
    meta = out / "metadata"
    meta.mkdir()

    # Autorater labels from every piece that has them, keyed by episode dir name; only the
    # episodes chosen above.
    names = {d.name for d in chosen.values()}
    per_run: dict = {}
    template: dict | None = None
    for p in args.pieces:
        f = p / "results" / "autorater.json"
        if f.is_file():
            data = json.loads(f.read_text())
            template = data
            per_run.update(
                {k: v for k, v in (data.get("per_run") or {}).items() if k in names}
            )
    if template is not None:
        template = dict(template)
        template["per_run"] = per_run
        (results / "autorater.json").write_text(json.dumps(template, indent=1))
    print(f"labels carried for {len(per_run)} of {len(chosen)} episodes")

    # results.json / run_meta.json from the newest piece that finished; else a minimal one.
    summary: dict = {}
    for p in reversed(args.pieces):
        f = p / "results" / "results.json"
        if f.is_file():
            summary = json.loads(f.read_text())
            break
    for p in reversed(args.pieces):
        for rel in ("metadata/run_meta.json", "run_meta.json"):
            f = p / rel
            if f.is_file():
                shutil.copy2(f, meta / "run_meta.json")
                break
        else:
            continue
        break
    summary = dict(summary)
    summary["colosseum_run_root"] = str(ts_root.relative_to(out))
    summary["seeds"] = sorted(chosen)
    summary["n_episodes"] = len(chosen)
    summary["merged_from"] = {str(s): o for s, o in sorted(origin.items())}
    summary["measures"] = {}
    (results / "results.json").write_text(json.dumps(summary, indent=2))
    (results / "per_seed.json").write_text("{}\n")
    print(f"wrote {out}; now run rescore_autorater.py on it")


if __name__ == "__main__":
    main()
