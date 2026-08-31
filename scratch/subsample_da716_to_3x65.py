# ABOUTME: Subsample the 4-pass da716 control down to 3 rollouts per cell, matching the
# ABOUTME: 3-pass ablation arms, and report the sampling spread rather than one arbitrary draw.

"""Make the control's sampling depth match its siblings.

The control ran 4 passes over the 65 shared cells (257 rollouts); every ablation arm except
c6masked ran 3 (~193). Depth does not bias a rate, but it does make the control's estimate
tighter, so this trims it to 3 per cell.

Which 3 of 4 you keep is arbitrary, and a single draw would present one arbitrary answer as
the answer. So the headline is the seed-0 draw and it is reported beside the distribution over
many draws — if that spread is wide, the 3-pass numbers of every arm should be read with the
same width in mind.

Run:
  uv run python scratch/subsample_da716_to_3x65.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv.odcv import misalignment_rate  # noqa: E402

ROOT = Path("output/odcv_bench")
CONTROL = "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch/combined4x_20260814_230249"
SIBLING = "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850"


def cells(run: str) -> dict[str, float]:
    """Median-across-judges severity per rollout key.

    Args:
        run: Path under output/odcv_bench to a combined run dir.

    Returns:
        Mapping of `variant/Scenario/rollout_NNN` to median severity.
    """
    per: dict[str, list[float]] = {}
    for f in (ROOT / run / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)):
                per.setdefault(k, []).append(s)
    return {k: statistics.median(v) for k, v in per.items()}


def main(keep: int = 3, draws: int = 2000, seed: int = 0) -> None:
    """Report the control's MR at full depth and subsampled to `keep` per cell.

    Args:
        keep: Rollouts to retain per cell.
        draws: How many random subsamples to summarise.
        seed: Seed for the headline draw; draws use seed, seed+1, ...
    """
    scen = lambda k: k.rsplit("/rollout_", 1)[0]  # noqa: E731
    ctl = cells(CONTROL)
    shared = {scen(k) for k in cells(SIBLING)}
    by: dict[str, list[float]] = {}
    for k, v in ctl.items():
        if scen(k) in shared:
            by.setdefault(scen(k), []).append(v)

    full = [v for vs in by.values() for v in vs]
    print(f"full depth: {len(full)} rollouts over {len(by)} cells  "
          f"MR {misalignment_rate(full):.1f}%  sev {statistics.mean(full):.2f}")

    def draw(s: int) -> list[float]:
        """One subsample keeping `keep` rollouts per cell (all of them if a cell has fewer)."""
        rng = random.Random(s)
        out = []
        for _, vs in sorted(by.items()):
            out += rng.sample(vs, keep) if len(vs) > keep else vs
        return out

    head = draw(seed)
    print(f"seed {seed} draw: {len(head)} rollouts  MR {misalignment_rate(head):.1f}%  "
          f"sev {statistics.mean(head):.2f}  ({sum(v >= 3 for v in head)} violations)")

    mrs = sorted(misalignment_rate(draw(seed + i)) for i in range(draws))
    lo, hi = mrs[int(0.025 * draws)], mrs[int(0.975 * draws)]
    print(f"\nover {draws} draws: mean {statistics.mean(mrs):.1f}%  median "
          f"{statistics.median(mrs):.1f}%  min {mrs[0]:.1f}%  max {mrs[-1]:.1f}%")
    print(f"  95% of draws fall in [{lo:.1f}, {hi:.1f}]")
    print("\nThis spread is subsampling noise ALONE — it is what changes when you keep a "
          "different 3 of 4 rollouts,\nwith the model, cells and judges all held fixed.")


if __name__ == "__main__":
    fire.Fire(main)
