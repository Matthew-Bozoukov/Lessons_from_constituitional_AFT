# ABOUTME: Disambiguate the `other` trajectory bucket: how often does a rollout NEVER raise a
# ABOUTME: hazard anywhere in its reasoning, and how long does it take to raise one?

"""Never-engages rate and time-to-first-flag for two arms.

Run: uv run python scratch/trait_arm_engagement.py
"""

import statistics
from pathlib import Path

import fire

from scratch.compare_trait_arm_reasoning import ARMS, load_arm
from scratch.trait_arm_reasoning_process import CLEARS, FLAGS


def main() -> None:
    """Print engagement metrics for both arms."""
    for arm, root in ARMS.items():
        recs = [r for r in load_arm(Path(root)) if r["reasons"]]
        never, first_at, only_clear = 0, [], 0
        for r in recs:
            idx = [i for i, t in enumerate(r["reasons"]) if FLAGS.search(t)]
            if not idx:
                never += 1
                if any(CLEARS.search(t) for t in r["reasons"]):
                    only_clear += 1
            else:
                first_at.append(idx[0] / max(1, len(r["reasons"]) - 1)
                                if len(r["reasons"]) > 1 else 0.0)
        n = len(recs)
        print(f"\n=== {arm} (n={n}) ===")
        print(f"  never raises a hazard anywhere      : {100*never/n:5.1f}%  ({never})")
        print(f"    ...of which it explicitly cleared : {only_clear}")
        print(f"  raises one, mean relative block idx : "
              f"{statistics.mean(first_at):.3f}  (0 = first block)")
        print(f"  raises it in the FIRST block        : "
              f"{100*sum(p == 0 for p in first_at)/n:5.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
