# ABOUTME: Integrity check for one arm's eval logs - did the target actually participate?
# ABOUTME: Prints ARM-CHECK PASS/WARN/FAIL; the grid driver stops on FAIL.
"""Verify an arm produced real audits rather than a complete-looking empty log.

The failure this exists to catch: the sibling experiment's first pilot produced
30 transcripts in a healthy-looking eval log where the target had **never
participated** - every tool-bearing request had returned HTTP 400. Sample count
alone would have passed. So this checks that target model events exist, not just
that samples were written.

Usage: python scripts/check_arm.py <arm_log_dir> <expected_samples>
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path


def main() -> int:
    log_dir, expected = Path(sys.argv[1]), int(sys.argv[2])
    files = sorted(glob.glob(str(log_dir / "*.eval")))
    if not files:
        print(f"ARM-CHECK FAIL: no .eval log under {log_dir}")
        return 1

    from inspect_ai.log import read_eval_log

    total = 0
    with_target = 0
    scored = 0
    turns: list[int] = []
    problems: list[str] = []

    for f in files:
        log = read_eval_log(f)
        for s in log.samples or []:
            total += 1
            t = 0
            for e in s.events or []:
                if getattr(e, "event", None) == "model":
                    role = getattr(e, "role", None)
                    if role == "target" or (role is None and "vllm" in str(getattr(e, "model", ""))):
                        t += 1
            turns.append(t)
            if t > 0:
                with_target += 1
            if s.scores:
                for sc in s.scores.values():
                    if isinstance(sc.value, dict) and sc.value:
                        scored += 1
                        break

    mean_turns = (sum(turns) / len(turns)) if turns else 0.0
    print(
        f"  samples={total}/{expected}  with_target_events={with_target}  "
        f"judged={scored}  mean_target_turns={mean_turns:.1f}"
    )

    # A target that never spoke means the transcripts are worthless regardless
    # of how many were written.
    if with_target == 0:
        problems.append("NO sample has target model events - the target never participated")
    elif with_target < total * 0.5:
        problems.append(f"only {with_target}/{total} samples have target events")
    if scored == 0:
        problems.append("no sample carries judge dimensions - scoring did not run")
    if total == 0:
        problems.append("zero samples")

    if problems:
        print("ARM-CHECK FAIL: " + "; ".join(problems))
        return 1

    # Short of the target is worth flagging but not worth discarding an arm:
    # --no-fail-on-error lets individual audits drop out, and the analysis
    # reports retained counts per arm anyway.
    if total < expected:
        print(f"ARM-CHECK WARN: {total} of {expected} samples completed (continuing)")
    else:
        print("ARM-CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
