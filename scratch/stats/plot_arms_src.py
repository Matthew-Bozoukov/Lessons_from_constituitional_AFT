# ABOUTME: The same two-arm ODCV plot as plot_arms_scratch.py, but with the error bars from
# ABOUTME: src/eval/stats.py, to check the migrated implementation against the scratch original.
# Run: uv run python scratch/stats/plot_arms_src.py

"""numina control vs 5% difficult advice, incentivized only, using the SRC statistics.

Identical data to `plot_arms_scratch.py` (same loader, same 25 shared scenarios, same first
judged pass per cell) so any difference in the figures is a difference in the statistics.

Incentivized-only through src: the ODCV Design carries `enumerated={"variant": "equal"}`
for the 50/50 mixture, and `DESIGN_ONE_VARIANT` is the same Design WITHOUT that factor --
the right one whenever a single variant is present. `odcv._design_for(variants)` picks
between them, so the eval's own `summarise()` already handles an incentivized-only arm; here
we name `DESIGN_ONE_VARIANT` directly because the loader has already filtered to one variant.
The model axis is never declared: `interval` infers checkpoints="sampled" from seeing 3 checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scratch.stats.arms_figure import dump, figure  # noqa: E402
from scratch.stats.plot_odcv_ci_comparison import load_tables  # noqa: E402
from src.eval.misalignment.odcv.odcv import DESIGN_ONE_VARIANT  # noqa: E402
from src.eval.stats import difference, interval  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402

SEEDS = (0, 42, 69)


def to_long(table, units: list[str]) -> list[dict]:
    """(n_checkpoints x n_items) violation table -> the long form src/eval/stats expects."""
    return [{"checkpoint": f"seed{SEEDS[i]}", "scenario": u, "pass": 0, "value": 100.0 * float(table[i, j])}
            for i in range(table.shape[0]) for j, u in enumerate(units)]


def main(out: str = "output/odcv_arms/src") -> None:
    load_dotenv()
    tables, shared, meta = load_tables()
    longs = {arm: to_long(t, shared) for arm, t in tables.items()}
    fmt = lambda r: {"mean": r.mean, "lo": r.lo, "hi": r.hi, "df": r.df, "method": r.method}

    per_arm = {arm: fmt(interval(rows, DESIGN_ONE_VARIANT)) for arm, rows in longs.items()}
    per_seed = {arm: [100 * float(r.mean()) for r in t] for arm, t in tables.items()}
    d_res = difference(longs["5% difficult advice"], longs["numina control"], DESIGN_ONE_VARIANT)
    d = fmt(d_res)

    dest = REPO / out
    png = figure(per_arm, d, per_seed,
                 f"SRC statistics (src/eval/stats.py) — ODCV incentivized, "
                 f"{len(shared)} shared scenarios x 3 seeds per arm",
                 dest, "odcv_arms_src",
                 extra=f"first judged pass per cell; git {git_sha()[:8]}")
    full = {arm: interval(rows, DESIGN_ONE_VARIANT).as_dict() for arm, rows in longs.items()}
    dump({"implementation": "src/eval/stats.py", "arms": per_arm, "diff": d, "per_seed": per_seed,
          "n_items": len(shared), "shared": shared, "meta": meta,
          "full_results": full, "diff_full": d_res.as_dict()}, dest)
    write_run_meta(dest, {"script": "scratch/stats/plot_arms_src.py", "n_items": len(shared),
                          "design": {"item": DESIGN_ONE_VARIANT.item,
                                     "item_sampling": DESIGN_ONE_VARIANT.item_sampling,
                                     "subsamples": list(DESIGN_ONE_VARIANT.subsamples)}})

    for arm, a in per_arm.items():
        print(f"{arm:22s} {a['mean']:5.1f}%  [{a['lo']:5.1f}, {a['hi']:5.1f}]  df {a['df']:.2f}")
    print(f"{'difference':22s} {d['mean']:+5.1f}pp [{d['lo']:+5.1f}, {d['hi']:+5.1f}]  df {d['df']:.2f}")
    r = full["numina control"]
    print("\nclaims recorded by src (numina):")
    for c in r["claims"]:
        print("   -", c)
    print(f">>> {png.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
