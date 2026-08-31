# ABOUTME: Check scenario/condition overlap between the full difficult-advice arm and the two
# ABOUTME: trait-subset arms before comparing their reasoning.

"""Report the shared (condition, scenario) cells across three ODCV arms."""

from pathlib import Path

import fire

from scratch.compare_trait_arm_reasoning import load_arm

ARMS3 = {
    "da_all9": "output/odcv_bench/qwen3_6-27b-lora-t2-9284-synthdoc-716-r64/combined4x_20260808_124051",
    "traits123": "output/odcv_bench/qwen3_6-27b-lora-9284-traits123-716-r64/combined3x_20260819_221522",
    "traits567": "output/odcv_bench/qwen3_6-27b-lora-9284-traits567-716-r64/combined3x_20260819_184957",
}


def main() -> None:
    """Print per-arm cell counts and the three-way intersection."""
    cells = {}
    for arm, root in ARMS3.items():
        recs = load_arm(Path(root))
        cells[arm] = {(r["condition"], r["scenario"]) for r in recs}
        print(f"{arm:<10} {len(recs):>4} rollouts  {len(cells[arm]):>3} cells")
    shared = set.intersection(*cells.values())
    print(f"\nshared by all three: {len(shared)} cells")
    for arm in ARMS3:
        extra = sorted(cells[arm] - shared)
        print(f"  {arm} has {len(extra)} cells the others lack: "
              f"{[f'{c}/{s}' for c, s in extra][:6]}{' ...' if len(extra) > 6 else ''}")


if __name__ == "__main__":
    fire.Fire(main)
