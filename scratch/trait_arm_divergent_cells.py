# ABOUTME: Rank the (condition, scenario) cells where the traits123 and traits567 arms'
# ABOUTME: judged ODCV severities diverge most, so the reasoning diff can be read where it bit.

"""Find the cells where two arms' judged severity disagrees most.

Run: uv run python scratch/trait_arm_divergent_cells.py
"""

import json
from pathlib import Path

import fire

RES = {
    "traits123": "output/odcv_bench/qwen3_6-27b-lora-9284-traits123-716-r64/combined3x_20260819_221522/results.json",
    "traits567": "output/odcv_bench/qwen3_6-27b-lora-9284-traits567-716-r64/combined3x_20260819_184957/results.json",
}


def main(top: int = 12) -> None:
    """Print the most divergent cells in both directions."""
    med = {}
    for arm, p in RES.items():
        r = json.load(open(p))
        med[arm] = {(c, k.split("/")[0]): v
                    for c in ("mandated", "incentivized")
                    for k, v in r["per_scenario_medians"].get(c, {}).items()}
    # per_scenario_medians is keyed per rollout; collapse to a per-scenario mean.
    agg = {}
    for arm in med:
        by = {}
        r = json.load(open(RES[arm]))
        for c in ("mandated", "incentivized"):
            for k, v in r["per_scenario_medians"].get(c, {}).items():
                by.setdefault((c, k.split("/")[0]), []).append(v)
        agg[arm] = {k: sum(v) / len(v) for k, v in by.items()}

    shared = sorted(set(agg["traits123"]) & set(agg["traits567"]))
    rows = [(agg["traits123"][k] - agg["traits567"][k], k,
             agg["traits123"][k], agg["traits567"][k]) for k in shared]
    rows.sort()
    print(f"{len(shared)} shared cells\n")
    print("=== traits567 WORSE (traits123 safer) ===")
    for d, k, x, y in rows[:top]:
        print(f"  {d:+.2f}  {k[0]:<13} {k[1]:<48} 123={x:.2f} 567={y:.2f}")
    print("\n=== traits123 WORSE (traits567 safer) ===")
    for d, k, x, y in rows[-top:][::-1]:
        print(f"  {d:+.2f}  {k[0]:<13} {k[1]:<48} 123={x:.2f} 567={y:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
