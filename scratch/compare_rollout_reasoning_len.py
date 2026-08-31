# ABOUTME: Compare the length of model reasoning inside ODCV rollouts across two arms,
# ABOUTME: restricted to the scenario cells both arms actually cover.
"""Did shorter training-time reasoning produce shorter eval-time reasoning?

Parses the `reason:` blocks out of each rollout's messages_record.txt and reports words per
reasoning turn, total reasoning per rollout, and turn counts. Arms are compared only on the
cells both cover, because the arms were run with different exclusion sets and a raw mean
would otherwise mix different scenario mixes.

Run: uv run python scratch/compare_rollout_reasoning_len.py --a <dir> --b <dir>
"""
import re
import statistics
from pathlib import Path

import fire

STEP = re.compile(r"^== Step \d+ ==$", re.M)
REASON = re.compile(r"^reason: (.*?)(?=^(?:content|call|role):|\Z)", re.M | re.S)
ROLE = re.compile(r"^role: (\w+)", re.M)


def rollouts(run_dir: Path) -> dict:
    """Map cell key -> list of per-rollout reasoning stats for one run directory."""
    out = {}
    for rec in sorted(run_dir.glob("agent_logs/*/experiments/*/**/messages_record.txt")):
        parts = rec.parts
        variant = "mandated" if "-mandated" in parts[-5] or "-mandated" in parts[-4] else \
                  "incentivized" if "-incentivized" in parts[-5] or "-incentivized" in parts[-4] \
                  else None
        scen = rec.parent.name if not rec.parent.name.startswith("rollout") \
            else rec.parent.parent.name
        if variant is None:
            for p in parts:
                if p.endswith("-mandated"):
                    variant = "mandated"
                elif p.endswith("-incentivized"):
                    variant = "incentivized"
        assert variant, f"no variant in {rec}"
        words, turns = [], 0
        for blk in STEP.split(rec.read_text()):
            m = ROLE.search(blk)
            if not m or m.group(1) != "assistant":
                continue
            r = REASON.search(blk)
            if r and r.group(1).strip():
                words.append(len(r.group(1).split()))
                turns += 1
        if words:
            out.setdefault(f"{variant}/{scen}", []).append(
                {"turns": turns, "total": sum(words), "per_turn": words})
    return out


def _fmt(vals):
    """median, mean and n for a list of numbers."""
    return (f"median {statistics.median(vals):>6.0f}  mean {statistics.mean(vals):>6.0f}  "
            f"n={len(vals)}")


def main(a: str, b: str, label_a: str = "", label_b: str = "") -> None:
    """Print the reasoning-length comparison for two ODCV arms.

    Args:
        a: First arm directory (any run dir, or the arm root).
        b: Second arm directory.
        label_a: Display name for the first arm.
        label_b: Display name for the second arm.
    """
    def collect(root: str) -> dict:
        p = Path(root)
        combined = sorted(p.glob("combined*x_*"))
        dirs = [combined[-1]] if combined else [p]
        merged = {}
        for d in dirs:
            for k, v in rollouts(d).items():
                merged.setdefault(k, []).extend(v)
        return merged

    A, B = collect(a), collect(b)
    la = label_a or Path(a).name
    lb = label_b or Path(b).name
    shared = sorted(set(A) & set(B))
    print(f"{la}: {len(A)} cells, {sum(len(v) for v in A.values())} rollouts")
    print(f"{lb}: {len(B)} cells, {sum(len(v) for v in B.values())} rollouts")
    print(f"shared cells: {len(shared)}\n")

    for scope, keys in (("ALL cells", None), ("SHARED cells", shared)):
        print(f"--- {scope} ---")
        for name, arm in ((la, A), (lb, B)):
            ks = keys if keys is not None else sorted(arm)
            rs = [r for k in ks for r in arm.get(k, [])]
            if not rs:
                continue
            per_turn = [w for r in rs for w in r["per_turn"]]
            print(f"  {name}")
            print(f"    words per reasoning turn : {_fmt(per_turn)}")
            print(f"    reasoning words / rollout: {_fmt([r['total'] for r in rs])}")
            print(f"    reasoning turns / rollout: {_fmt([r['turns'] for r in rs])}")
        print()

    # paired by cell, on shared cells only
    if shared:
        da, db = [], []
        for k in shared:
            da.append(statistics.mean(r["total"] for r in A[k]))
            db.append(statistics.mean(r["total"] for r in B[k]))
        diffs = [x - y for x, y in zip(da, db)]
        lower = sum(1 for d in diffs if d < 0)
        print(f"--- paired by cell (n={len(shared)}) ---")
        print(f"  mean reasoning words/rollout: {la} {statistics.mean(da):.0f} vs "
              f"{lb} {statistics.mean(db):.0f}")
        print(f"  {la} shorter on {lower}/{len(shared)} cells "
              f"({100*lower/len(shared):.0f}%)")
        print(f"  median paired difference: {statistics.median(diffs):+.0f} words")


if __name__ == "__main__":
    fire.Fire(main)
