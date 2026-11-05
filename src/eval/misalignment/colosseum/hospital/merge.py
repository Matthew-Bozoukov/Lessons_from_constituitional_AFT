# ABOUTME: Merge the several run dirs of one Hospital cell (pilot, main sweep, top-ups) into ONE
# ABOUTME: consolidated run dir per (condition, arm), so each cell publishes as a single Hub repo; or a cell's seed shards.

"""Consolidate a cell's pieces into one publishable run dir.

    uv run python scripts/eval/misalignment/colosseum/hospital/merge.py cells \
        [--root output/colosseum_hospital] [--out output/colosseum_hospital/merged] \
        [--date 2026-09-04] [--dry-run]
    uv run python scripts/eval/misalignment/colosseum/hospital/merge.py shards --out <dir> <piece> ...

A cell's episodes arrived in pieces: a pilot, the main sweep, and top-up jobs that re-ran
the seeds the 32k context window dropped. Publishing each piece separately would push three
run dirs under ONE Hub name (run_eval names by cell + arm). This copies every finished
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
from src.eval.misalignment.colosseum.hospital.judge import judge_measures
from src.eval.misalignment.colosseum.publish import repo_name_for
from src.naming import to_local


def merged_name(pieces: list[Path], condition: str, partner_seat, *, date: str | None) -> str:
    """The Hub name the merged cell publishes under, from its newest piece's run_meta.json —
    exactly what run_eval would have called one run of the whole cell (publish.repo_name_for:
    `<date>-hospital-<condition>[-<partner seat>]-<arm>`)."""
    meta = json.loads((pieces[-1] / "metadata" / "run_meta.json").read_text())
    cfg = OmegaConf.merge(
        OmegaConf.create(meta.get("config") or {}),
        {"condition": condition, "partner_seat": partner_seat},
    )
    meta = {**meta, "config": OmegaConf.to_container(cfg, resolve=True)}
    return repo_name_for("colosseum_hospital", meta, produced=date).split("/", 1)[1]


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


def merge_cell(
    pieces: list[Path], dest: Path, *, dry_run: bool, env_logs: str | None = None
) -> dict:
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

    # The text measures are read by the autorater (hospital/autorater.py): each piece labelled
    # its own episodes in results/autorater.json, keyed by episode dir, so the union is the
    # cell's labels and nothing is re-read (or read with keywords). A piece with no labels
    # (a pre-2026-10-10 run) leaves its episodes unlabelled, and the harvest refuses them.
    auto = merge_json(pieces, "results/autorater.json", keyed="per_run")
    harvest = harvest_run_root(
        ts_root,
        expected_seats=summary["seats"],
        pair=PAIR,
        env_logs_root=env_logs,
        labels=(auto or {}).get("per_run") or {},
    )
    results = dest / "results"
    results.mkdir(parents=True)
    if auto:
        for k in ("n_calls", "unparsed", "reused_episodes"):
            auto[k] = sum(
                json.loads((q / "results" / "autorater.json").read_text()).get(k, 0)
                for q in pieces
                if (q / "results" / "autorater.json").is_file()
            )
        (results / "autorater.json").write_text(json.dumps(auto, indent=1))
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
        # The judge's per-seed measures join per_seed.json, as run() does for one run
        # (judge.judge_arm); without this a merged cell published no judge measures.
        cond = summary["condition"]
        for measure, cells in judge_measures(judge, cond, pair=PAIR).items():
            seeds = cells.get(cond) or {}
            if seeds:
                harvest["measures"].setdefault(measure, {})[cond] = {
                    str(k): v for k, v in sorted(seeds.items())
                }
        (results / "per_seed.json").write_text(json.dumps(harvest["measures"], indent=2))
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

    # The cell's own config is the biggest piece's (the main run), not the newest's: a
    # one-seed top-up at a larger window must not stand in for it. Every piece's
    # metadata is kept beside it under metadata/pieces/<piece>/.
    main_piece = max(pieces, key=lambda q: len(episodes_of(q)))
    shutil.copytree(main_piece / "metadata", dest / "metadata")
    # Two pods launched in the same second produce run dirs with the SAME name, so the piece
    # name alone collides; the pod dir above it disambiguates (caught 2026-09-27).
    names = [q.name for q in pieces]
    for q in pieces:
        if (q / "metadata").is_dir():
            key = q.name if names.count(q.name) == 1 else f"{q.parent.name}__{q.name}"
            shutil.copytree(q / "metadata", dest / "metadata" / "pieces" / key)
    (dest / "metadata" / "merged_from.json").write_text(
        json.dumps(
            {
                "pieces": [str(p) for p in pieces],
                "seeds_from": {str(s): seeds_from[s] for s in sorted(seeds_from)},
            },
            indent=1,
        )
    )
    if (main_piece / "run_meta.json").is_file():
        shutil.copy2(main_piece / "run_meta.json", dest / "run_meta.json")
    return {"episodes": len(chosen), "dest": str(dest)}


def cells_main(argv: list[str] | None = None) -> None:
    """`merge cells`: consolidate every finished cell under --root, one dir per cell."""
    ap = argparse.ArgumentParser(prog="merge cells", description=__doc__)
    ap.add_argument("--root", default="output/colosseum_hospital")
    ap.add_argument("--out", default="output/colosseum_hospital/merged")
    ap.add_argument(
        "--date", default="2026-09-04", help="the day the episodes were produced"
    )
    ap.add_argument(
        "--only", default="", help="comma list of conditions to merge (default all)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep a merged cell that already holds every seed its pieces have "
        "(rebuilding it would delete its judge files); rebuild only cells that gained seeds",
    )
    ap.add_argument(
        "--env-logs",
        default=None,
        help="the pulled env snapshots of these cells (objective deficits); omit = regex",
    )
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)

    cells: dict[tuple[str, str], list[Path]] = {}
    # Run dirs at any depth: a fleet pulls each pod into <root>/<group>/<pod>/<run dir>.
    for d in sorted({p.parent.parent for p in root.glob("**/rollouts/colosseum")}):
        if not (d / "rollouts" / "colosseum").is_dir():
            continue
        if out.resolve() in d.resolve().parents:
            continue
        res = d / "results" / "results.json"
        if not res.is_file():
            print(f"skip (still running, no results.json): {d.name}")
            continue
        s = json.loads(res.read_text())
        cells.setdefault((s["condition"], s["arm"], s.get("partner_seat")), []).append(
            d
        )

    only = {c for c in args.only.split(",") if c}
    for (condition, arm, partner_seat), pieces in sorted(
        cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")
    ):
        if only and condition not in only:
            continue
        pieces.sort(key=lambda p: p.name[-6:])  # HHMMSS suffix, one day: later wins
        label = merged_name(pieces, condition, partner_seat, date=args.date)
        if args.skip_existing:
            dest_probe = out / to_local(label)
            done = dest_probe / "results" / "results.json"
            seeds = set().union(*(episodes_of(p) for p in pieces))
            if done.is_file() and json.loads(done.read_text()).get("n_episodes") == len(
                seeds
            ):
                print(
                    f"{condition} / {label}: up to date ({len(seeds)} episodes), kept"
                )
                continue
        print(f"{condition} / {label}: {[p.name[-6:] for p in pieces]}")
        dest = out / to_local(label)
        merge_cell(pieces, dest, dry_run=args.dry_run, env_logs=args.env_logs)


SHARDS_DOC = """Union the finished episodes of several pieces of ONE Hospital cell into one run dir.

    uv run python scripts/eval/misalignment/colosseum/hospital/merge.py shards --out <merged dir> <piece dir> [...]

Unlike `merge cells` this accepts pieces WITHOUT results/results.json (a full run whose
sweep is still going: its finished episodes already sit in its Colosseum tree) and carries
the autorater labels the finished shards already hold (results/autorater.json, keyed by
episode dir), so the follow-up

    uv run python scripts/eval/misalignment/colosseum/hospital/autorater.py <merged dir> --config <cfg>

labels only the episodes no shard labelled (autorate_arm reuses complete prior labels),
re-runs the action measures over all of them and writes per_seed.json / episodes.json /
results.json. Later pieces win a duplicated seed. The judge's measures are NOT merged:
the headline sabotage act and its refusal/plan split need the harvest + autorater only."""


def shards_main(argv: list[str] | None = None) -> None:
    """`merge shards`: union a cell's seed shards (finished or still running) into one dir."""
    ap = argparse.ArgumentParser(prog="merge shards", description=SHARDS_DOC)
    ap.add_argument("pieces", nargs="+", type=Path, help="run dirs, oldest first")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--seeds", default=None, help="keep only these seeds, e.g. 1-30 or 1,2,3"
    )
    args = ap.parse_args(argv)

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
    print(f"wrote {out}; now run hospital/autorater.py on it")



def main(argv: list[str] | None = None) -> None:
    """`merge cells ...` (every finished cell under a root) or `merge shards ...` (one cell's
    seed shards, finished or not)."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    commands = {"cells": cells_main, "shards": shards_main}
    if not argv or argv[0] not in commands:
        raise SystemExit("usage: merge {cells,shards} [args]  (each takes --help)")
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
