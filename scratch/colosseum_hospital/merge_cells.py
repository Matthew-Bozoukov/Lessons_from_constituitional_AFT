# ABOUTME: Merge the several run dirs of one Hospital cell (pilot, main sweep, top-ups) into ONE
# ABOUTME: consolidated run dir per (condition, arm), so each cell publishes as a single Hub repo.

"""Consolidate a cell's pieces into one publishable run dir.

    uv run python scratch/colosseum_hospital/merge_cells.py [--root output/colosseum_hospital] \
        [--out output/colosseum_hospital/merged] [--date 2026-09-04] [--dry-run]

A cell's episodes arrived in pieces: a pilot, the main sweep, and top-up jobs that re-ran
the seeds the 32k context window dropped. Publishing each piece separately would push three
run dirs under ONE Hub name (repo_name_for names by cell + arm). This copies every finished
piece's episodes into one raw tree per cell — a later run dir wins a duplicated seed, the
rule analyse.py applies — re-harvests it with the eval's own harvest_run_root, and merges the
judge and false-claim files, so the consolidated dir satisfies publish_colosseum.py's
contract (metadata/run_meta.json + rollouts/colosseum/<ts>/runs + results/results.json).
Run dirs without results/results.json (still running) are listed and skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.hospital.config import PAIR
from src.eval.misalignment.colosseum.hospital.harvest import harvest_run_root
from src.eval.misalignment.colosseum.publish import arm_label
from src.utils import local_name

SEED = re.compile(r"_seed(\d+)$")


def episodes_of(run_dir: Path) -> dict[int, Path]:
    """seed -> finished episode dir, over every raw tree under a run dir."""
    out: dict[int, Path] = {}
    for d in sorted(run_dir.glob("rollouts/colosseum/*/runs/*/*/*")):
        if not (d / "blackboards.json").is_file():
            continue
        m = SEED.search(d.name)
        assert m, f"no seed in episode dir name {d.name}"
        out[int(m.group(1))] = d
    return out


def merge_json(
    pieces: list[Path], rel: str, *, keyed: str | tuple[str, ...]
) -> dict | None:
    """Union of the `keyed` dict(s) across pieces (later wins), other fields from the newest."""
    found = [p / rel for p in pieces if (p / rel).is_file()]
    if not found:
        return None
    merged: dict = json.loads(found[-1].read_text())
    for key in (keyed,) if isinstance(keyed, str) else keyed:
        union: dict = {}
        for f in found:
            union.update(json.loads(f.read_text()).get(key) or {})
        merged[key] = union
    return merged


def merge_cell(pieces: list[Path], dest: Path, *, dry_run: bool) -> dict:
    """Build one consolidated run dir from `pieces` (oldest first)."""
    seeds_from: dict[int, str] = {}
    chosen: dict[int, Path] = {}
    for p in (
        pieces
    ):  # later pieces overwrite: the top-up re-ran what an earlier job dropped
        for seed, d in episodes_of(p).items():
            if seed in chosen:
                print(f"    seed {seed}: {seeds_from[seed]} superseded by {p.name}")
            chosen[seed] = d
            seeds_from[seed] = p.name
    newest = pieces[-1]
    summary = json.loads((newest / "results" / "results.json").read_text())
    print(f"  -> {dest.name}: {len(chosen)} episodes from {len(pieces)} piece(s)")
    if dry_run:
        return {"episodes": len(chosen)}

    if dest.exists():
        shutil.rmtree(dest)
    src_root = next(iter(chosen.values())).parents[3]  # .../rollouts/colosseum/<ts>
    ts_root = dest / "rollouts" / "colosseum" / src_root.name
    for seed, d in sorted(chosen.items()):
        target = ts_root / "runs" / d.parent.parent.name / d.parent.name / d.name
        shutil.copytree(d, target)
    # Sweep-level files beside runs/ (configs, summaries) from the newest piece.
    newest_root = next(iter(episodes_of(newest).values())).parents[3]
    for entry in newest_root.iterdir():
        if entry.name == "runs":
            continue
        if entry.is_dir():
            shutil.copytree(entry, ts_root / entry.name)
        else:
            shutil.copy2(entry, ts_root / entry.name)
    for p in pieces:
        log = p / "rollouts" / "colosseum_driver.log"
        if log.is_file():
            shutil.copy2(log, dest / "rollouts" / f"colosseum_driver_{p.name[-6:]}.log")

    harvest = harvest_run_root(ts_root, expected_seats=summary["seats"], pair=PAIR)
    results = dest / "results"
    results.mkdir(parents=True)
    (results / "per_seed.json").write_text(json.dumps(harvest["measures"], indent=2))
    (results / "episodes.json").write_text(json.dumps(harvest["episodes"], indent=2))

    summary = dict(summary)
    summary["measures"] = harvest["measures"]
    summary["health"] = harvest["health"]
    summary["colosseum_run_root"] = str(ts_root.relative_to(dest))
    summary["seeds"] = sorted(chosen)
    summary["n_episodes"] = len(chosen)
    summary["merged_from"] = [p.name for p in pieces]

    judge = merge_json(pieces, "results/judge.json", keyed="per_run")
    if judge:
        judge["n_episodes"] = len(judge["per_run"])
        for k in ("n_calls", "unparsed"):
            judge[k] = sum(
                json.loads((p / "results" / "judge.json").read_text()).get(k, 0)
                for p in pieces
                if (p / "results" / "judge.json").is_file()
            )
        (results / "judge.json").write_text(json.dumps(judge, indent=1))
        summary["judge"] = {k: v for k, v in judge.items() if k != "per_run"}
    # Both the per-shift counts AND the per-message records, else a merged cell publishes
    # only the newest piece's messages (caught 2026-09-06).
    fc = merge_json(pieces, "results/false_claims.json", keyed=("summary", "per_run"))
    if fc:
        (results / "false_claims.json").write_text(json.dumps(fc, indent=1))
    (results / "results.json").write_text(json.dumps(summary, indent=2))
    (results / "results.md").write_text(
        f"# {dest.name}\n\nMerged from: {', '.join(p.name for p in pieces)}\n\n"
        f"Episodes: {len(chosen)} (seeds {min(chosen)}-{max(chosen)})\n\n"
        "See per_seed.json (measures), episodes.json, judge.json"
        + (", false_claims.json" if fc else "")
        + ".\n"
    )

    shutil.copytree(newest / "metadata", dest / "metadata")
    (dest / "metadata" / "merged_from.json").write_text(
        json.dumps(
            {
                "pieces": [str(p) for p in pieces],
                "seeds_from": {str(s): seeds_from[s] for s in sorted(seeds_from)},
            },
            indent=1,
        )
    )
    if (newest / "run_meta.json").is_file():
        shutil.copy2(newest / "run_meta.json", dest / "run_meta.json")
    return {"episodes": len(chosen), "dest": str(dest)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="output/colosseum_hospital")
    ap.add_argument("--out", default="output/colosseum_hospital/merged")
    ap.add_argument("--config", default="configs/eval/colosseum_hospital.yaml")
    ap.add_argument(
        "--date", default="2026-09-04", help="the day the episodes were produced"
    )
    ap.add_argument(
        "--only", default="", help="comma list of conditions to merge (default all)"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    root, out = Path(args.root), Path(args.out)

    cells: dict[tuple[str, str], list[Path]] = {}
    for d in sorted(root.glob("*/")):
        if not (d / "rollouts" / "colosseum").is_dir():
            continue
        res = d / "results" / "results.json"
        if not res.is_file():
            print(f"skip (still running, no results.json): {d.name}")
            continue
        s = json.loads(res.read_text())
        cells.setdefault((s["condition"], s["arm"]), []).append(d)

    only = {c for c in args.only.split(",") if c}
    for (condition, arm), pieces in sorted(cells.items()):
        if only and condition not in only:
            continue
        pieces.sort(key=lambda p: p.name[-6:])  # HHMMSS suffix, one day: later wins
        label = arm_label(arm, cfg)
        print(f"{condition} / {label}: {[p.name[-6:] for p in pieces]}")
        dest = out / local_name(
            f"colosseum_hospital_{condition}_{label}", date=args.date
        )
        merge_cell(pieces, dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
