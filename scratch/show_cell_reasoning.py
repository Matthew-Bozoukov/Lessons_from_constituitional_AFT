# ABOUTME: Print the first reasoning trace of every rollout in one (arm, condition, scenario)
# ABOUTME: cell, so two arms' CoT on the same ODCV task can be read side by side.

"""Dump reasoning traces for one benchmark cell.

Run: uv run python scratch/show_cell_reasoning.py --scenario Ai-Healthcare-Administrator \
        --condition mandated [--block 0] [--chars 2600]
"""

from pathlib import Path

import fire

from scratch.compare_trait_arm_reasoning import ARMS, load_arm


def main(scenario: str, condition: str = "mandated", block: int = 0,
         chars: int = 2600) -> None:
    """Print one reasoning block per rollout for both arms in a cell."""
    for arm, root in ARMS.items():
        recs = [r for r in load_arm(Path(root))
                if r["scenario"] == scenario and r["condition"] == condition]
        print(f"\n{'=' * 100}\n### {arm} — {condition}/{scenario} — "
              f"{len(recs)} rollouts\n{'=' * 100}")
        for r in sorted(recs, key=lambda x: x["rollout"]):
            n = len(r["reasons"])
            print(f"\n--- {r['rollout']} ({n} reason blocks) | block {block} ---")
            print(r["reasons"][block][:chars] if n > block else "(no such block)")


if __name__ == "__main__":
    fire.Fire(main)
