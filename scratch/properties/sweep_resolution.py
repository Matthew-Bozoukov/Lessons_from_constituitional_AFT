# ABOUTME: Choose `min_cluster_size` by measurement rather than taste: refit one channel's
# ABOUTME: cached embeddings across a grid of resolutions and seeds, and report stability.
# Run: uv run python scratch/properties/sweep_resolution.py --config <cfg> --out_dir <run>

"""The resolution sweep, run against cached embeddings.

`min_cluster_size` is a RESOLUTION SETTING, not a finding, and picking it by eye is how the
2026-08-19 run first reported 17 groups from a value (60) that sat on a bifurcation — a
point where HDBSCAN has two roughly equally good readings of the same density landscape and
the seed picks one. The published run fixed that by sweeping. The value it landed on (25)
was measured on a 4,540-string corpus and does not transfer to a corpus of a different
size, so the sweep is repeated whenever the corpus changes.

This deliberately reuses the producer's own `build_units` and `_embed`, so the vocabulary
and the vectors it sweeps over are byte-identical to the ones the real run will cluster —
a sweep over a separately-built matrix would be measuring a different thing. It also leaves
`embeddings.npy` and `units.json` in the run directory, so the real run reuses them and
pays for the embedding pass once.

What to read:

    n_groups        across seeds at one resolution. Three very different counts means the
                    fit is a coin flip at that resolution; do not report from there.
    n_exportable    groups covering at least `min_group_records` records — what actually
                    reaches the property list, which is the number that matters.
    noise_share     the share of feature strings that clustered nowhere. This is coverage
                    the property list does not describe, and it belongs in the write-up.
    pairwise ARI    agreement between the seeds' labelings at one resolution. Low ARI with
                    similar group counts is still instability, just better disguised.

Throwaway by construction (CLAUDE.md: scratch/ is the default home for one-off code).
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.properties import block  # noqa: E402
from src.properties.producers import clusters as clusters_mod  # noqa: E402
from src.properties.shared import grouping as grouping_mod  # noqa: E402
from src.properties.sources import load_source  # noqa: E402

GRID = (15, 25, 40, 60, 90, 130)
SEEDS = (0, 1, 2)


def _exportable(result, units, floor: int) -> tuple[int, list[int]]:
    """How many groups reach the record floor, and their sizes.

    Args:
        result: The Grouping over units.
        units: The producer's Units.
        floor: `min_group_records`.

    Returns:
        (count, the sizes of those groups, largest first).
    """
    sizes = []
    for group in range(result.n_groups):
        covered: set[int] = set()
        for u in result.members(group):
            covered.update(units.records[u])
        if len(covered) >= floor:
            sizes.append(len(covered))
    return len(sizes), sorted(sizes, reverse=True)


def main(config: str, out_dir: str, producer: str | None = None,
         grid: tuple = GRID, seeds: tuple = SEEDS) -> None:
    """Sweep one producer's clustering resolution over its cached embeddings.

    Args:
        config: The discover config.
        out_dir: The run directory holding `<producer>/features.jsonl`.
        producer: Which producer block to sweep; defaults to every `features` one.
        grid: `min_cluster_size` values to try.
        seeds: Seeds per value.
    """
    from sklearn.metrics import adjusted_rand_score

    cfg = OmegaConf.load(config)
    records, _ = load_source(OmegaConf.to_container(cfg.source, resolve=True))
    names = ([producer] if producer else
             [n for n, p in (cfg.producers or {}).items()
              if str(p.get("evidence", "features")) == "features"])

    report = {}
    for name in names:
        producer_cfg = cfg.producers[name]
        channel = str(producer_cfg.get("channel", "reasoning"))
        floor = int(producer_cfg.get("min_group_records",
                                     clusters_mod.MIN_GROUP_RECORDS))
        run = Path(out_dir) / name
        kept = [r for r in records if r.channel(channel).strip()]
        units = clusters_mod.build_units(kept, channel, producer_cfg, run)
        vectors, meta = clusters_mod._embed(units, producer_cfg, run)
        print(f"\n=== {name}: {len(units.texts)} units over {len(kept)} records, "
              f"{meta.dim}-d ({meta.model}) ===")

        base = block(producer_cfg, "grouping")
        # ONE UMAP FIT PER SEED, reused across the whole grid. `min_cluster_size` is an
        # HDBSCAN parameter and does not touch the reduction, so refitting UMAP for each
        # value would recompute an identical embedding six times — and a seeded UMAP fit
        # runs single-threaded, so that is the entire cost of the sweep. Reducing once per
        # seed and re-clustering the cached coordinates is the same measurement, 6x faster.
        # The coords are cached on disk too, so a rerun of this sweep is nearly free.
        coords_by_seed = {}
        for seed in seeds:
            cache = run / f"sweep_coords_seed{seed}.npy"
            if cache.exists():
                coords_by_seed[seed] = np.load(cache)
                print(f"  reusing the UMAP fit at seed {seed} from {cache.name}")
                continue
            params = grouping_mod.GroupingParams(**{**base, "seed": int(seed)})
            print(f"  reducing at seed {seed} "
                  f"(n_neighbors={params.n_neighbors}, n_components={params.n_components})")
            coords_by_seed[seed] = grouping_mod.reduce_umap(vectors, params)
            np.save(cache, coords_by_seed[seed])

        rows = []
        for size in grid:
            labelings = []
            for seed in seeds:
                params = grouping_mod.GroupingParams(
                    **{**base, "min_cluster_size": int(size), "seed": int(seed)})
                # `_fit` rather than `group`, so no collapsed fit is silently retried at
                # the next seed — a sweep that retried would be measuring the retry logic
                # instead of how often this resolution fails. Matches stability_sweep.
                result = grouping_mod._fit(vectors, coords_by_seed[seed], params)
                n_exportable, sizes = _exportable(result, units, floor)
                labelings.append(result.labels)
                rows.append({"min_cluster_size": int(size), "seed": int(seed),
                             "n_groups": result.n_groups,
                             "n_exportable": n_exportable,
                             "largest_groups": sizes[:5],
                             "noise_share": round(result.meta["noise_share"], 4),
                             "collapsed": bool(grouping_mod.is_degenerate(
                                 result.labels, result.n_groups, vectors,
                                 result.coords))})
            aris = [round(float(adjusted_rand_score(a, b)), 4)
                    for a, b in combinations(labelings, 2)]
            counts = [r["n_groups"] for r in rows[-len(seeds):]]
            exportable = [r["n_exportable"] for r in rows[-len(seeds):]]
            noise = [r["noise_share"] for r in rows[-len(seeds):]]
            collapsed = sum(r["collapsed"] for r in rows[-len(seeds):])
            # A resolution is only usable when the seeds agree about BOTH how many groups
            # there are and which points go together; either alone can look fine while the
            # other is a coin flip.
            verdict = ("collapsing" if collapsed
                       else "bifurcating" if max(counts) > 2 * max(1, min(counts))
                       else "unstable" if aris and min(aris) < 0.5 else "stable")
            print(f"  min_cluster_size={size:4d}  groups {counts}  exportable "
                  f"{exportable}  noise {[f'{n:.0%}' for n in noise]}  "
                  f"ARI {aris}  collapsed {collapsed}/{len(seeds)}  -> {verdict}")
            report.setdefault(name, []).append(
                {"min_cluster_size": int(size), "n_groups": counts,
                 "n_exportable": exportable, "noise_share": noise,
                 "pairwise_ari": aris, "n_collapsed": collapsed, "verdict": verdict,
                 "largest_groups": rows[-len(seeds)]["largest_groups"]})

    path = Path(out_dir) / "resolution_sweep.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = ["# Resolution sweep — `min_cluster_size` chosen by measurement", "",
             "A floor on FEATURE STRINGS, not rollouts. `exportable` counts the groups "
             "that clear `min_group_records` and therefore reach the property list. "
             "`collapsing` means at least one seed's UMAP fit degenerated (the exported "
             "run retries past that, but a resolution that needs the retry is not one to "
             "report from). `bifurcating` means the seeds disagree about how many groups "
             "exist; `unstable` means they agree on the count and disagree about "
             "membership. Report from a `stable` row only.", ""]
    for name, block_rows in report.items():
        lines += [f"## {name}", "",
                  "| min_cluster_size | groups (3 seeds) | exportable | noise | "
                  "pairwise ARI | collapsed | |", "|--:|---|---|---|---|--:|---|"]
        for row in block_rows:
            lines.append(
                f"| {row['min_cluster_size']} | {row['n_groups']} | "
                f"{row['n_exportable']} | "
                f"{[f'{n:.0%}' for n in row['noise_share']]} | {row['pairwise_ari']} | "
                f"{row['n_collapsed']} | {row['verdict']} |")
        lines.append("")
    (Path(out_dir) / "resolution_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n>>> {path} and resolution_sweep.md written")


if __name__ == "__main__":
    fire.Fire(main)
