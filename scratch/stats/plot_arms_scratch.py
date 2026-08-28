# ABOUTME: Two-arm ODCV bar + difference plot, error bars from scratch/stats/crossed_ci.py.
# Run: uv run python scratch/stats/plot_arms_scratch.py

"""numina control vs 5% difficult advice, incentivized only, using the SCRATCH statistics.

Three seeds per arm, first judged pass per cell, on the 25 incentivized scenarios all six
models share. `crossed_ci.both_random` / `both_random_diff`: SE^2 = T_A + T_B - T_C (and the
five-term paired version for the difference), multiplier +/-1.96.

Its counterpart `plot_arms_src.py` produces the same two panels through `src/eval/stats.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scratch.stats.arms_figure import dump, figure  # noqa: E402
from scratch.stats.crossed_ci import both_random, both_random_diff  # noqa: E402
from scratch.stats.plot_odcv_ci_comparison import load_tables  # noqa: E402
from src.utils import git_sha, write_run_meta  # noqa: E402


def main(out: str = "output/odcv_arms/scratch") -> None:
    load_dotenv()
    tables, shared, meta = load_tables()          # incentivized, first pass per cell, shared units

    pct = lambda iv: {"mean": 100 * iv.mean, "lo": 100 * iv.lo, "hi": 100 * iv.hi,
                      "df": iv.df, "method": iv.method}

    per_arm = {arm: pct(both_random(t)) for arm, t in tables.items()}
    per_seed = {arm: [100 * float(r.mean()) for r in t] for arm, t in tables.items()}
    d = pct(both_random_diff(tables["5% difficult advice"], tables["numina control"]))

    dest = REPO / out
    png = figure(per_arm, d, per_seed,
                 f"SCRATCH statistics (crossed_ci.py) — ODCV incentivized, "
                 f"{len(shared)} shared scenarios x 3 seeds per arm",
                 dest, "odcv_arms_scratch",
                 extra=f"first judged pass per cell; git {git_sha()[:8]}")
    dump({"implementation": "scratch/stats/crossed_ci.py", "arms": per_arm, "diff": d,
          "per_seed": per_seed, "n_units": len(shared), "shared": shared, "meta": meta}, dest)
    write_run_meta(dest, {"script": "scratch/stats/plot_arms_scratch.py", "n_units": len(shared)})

    for arm, a in per_arm.items():
        print(f"{arm:22s} {a['mean']:5.1f}%  [{a['lo']:5.1f}, {a['hi']:5.1f}]  df {a['df']:.2f}")
    print(f"{'difference':22s} {d['mean']:+5.1f}pp [{d['lo']:+5.1f}, {d['hi']:+5.1f}]  df {d['df']:.2f}")
    print(f">>> {png.relative_to(REPO)}")


if __name__ == "__main__":
    fire.Fire(main)
