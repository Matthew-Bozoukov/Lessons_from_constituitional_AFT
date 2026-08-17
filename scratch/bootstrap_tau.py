# ABOUTME: Paired bootstrap over speeches for the debate_speeches tau-b gaps: is a variant
# ABOUTME: really a better judge than difficult advice, or is the gap resampling noise?
# Run: uv run python scratch/bootstrap_tau.py [n_boot]

"""Why paired, and why bootstrap.

Every arm rated the SAME speeches, so the arms' errors are correlated — a speech the humans
found hard is hard for all five. An unpaired comparison of two tau values throws that
correlation away and is far too conservative. Resampling *speeches* (not ratings) and
recomputing both taus on each resample keeps the pairing, which is the whole reason the
identical-items design was worth insisting on.

Bootstrap rather than an analytic SE because tau-b's null SE assumes no ties and both series
here are heavily tied (integers 1-5 against a near-continuous human mean) — the analytic
number would be wrong in an unknown direction.
"""

from __future__ import annotations

import glob
import json
import random
import statistics
import sys
from pathlib import Path

from src.eval.deliberation.debate_speeches.stats import kendall_tau_b

SHORT = {"courtroom716": "CR", "peercritique716": "PC", "da716": "DA",
         "table2-only": "T2", "Qwen3_6-27B": "base"}
REFERENCE = "DA"


def label(arm: str) -> str | None:
    for key, short in SHORT.items():
        if key in arm:
            return short
    return None


def load() -> dict[str, dict[str, tuple[int, float]]]:
    """{arm: {uid: (model_rating, human_mean)}}, unparsed ratings dropped."""
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for path in sorted(glob.glob("output/debate_speeches/*/*/records.jsonl")):
        arm = label(path.split("/")[2])
        if arm is None:
            continue
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        out[arm] = {r["uid"]: (r["rating"], r["human_mean"]) for r in rows if r["rating"]}
    return out


def main(n_boot: int = 2000, seed: int = 0) -> None:
    data = load()
    if REFERENCE not in data:
        print("no DA arm yet")
        return
    # Only speeches every arm rated parseably, so all comparisons run on one common set.
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    print(f"arms: {sorted(data)} | speeches common to all: {len(common)}\n")

    point = {arm: kendall_tau_b([data[arm][u][0] for u in common],
                                [data[arm][u][1] for u in common])
             for arm in data}
    for arm, tau in sorted(point.items(), key=lambda kv: -kv[1]):
        print(f"  {arm:5s} tau_b {tau:.4f}")

    rng = random.Random(seed)
    others = [a for a in data if a != REFERENCE]
    diffs: dict[str, list[float]] = {a: [] for a in others}
    for _ in range(n_boot):
        sample = [common[rng.randrange(len(common))] for _ in range(len(common))]
        ref = kendall_tau_b([data[REFERENCE][u][0] for u in sample],
                            [data[REFERENCE][u][1] for u in sample])
        for arm in others:
            diffs[arm].append(
                kendall_tau_b([data[arm][u][0] for u in sample],
                              [data[arm][u][1] for u in sample]) - ref)

    print(f"\npaired bootstrap ({n_boot} resamples of {len(common)} speeches) vs {REFERENCE}:")
    for arm in sorted(others, key=lambda a: -(point[a] - point[REFERENCE])):
        series = sorted(diffs[arm])
        low = series[int(0.025 * len(series))]
        high = series[int(0.975 * len(series)) - 1]
        # Two-sided bootstrap p: how often the resampled difference crosses zero.
        share = sum(d <= 0 for d in series) / len(series)
        p = 2 * min(share, 1 - share)
        verdict = "SEPARATED" if low > 0 or high < 0 else "not separated"
        print(f"  {arm:5s} {point[arm] - point[REFERENCE]:+.4f}  "
              f"95% CI [{low:+.4f}, {high:+.4f}]  p={p:.3f}  {verdict}")

    spread = statistics.fmean([abs(d) for d in diffs[others[0]]])
    print(f"\n(mean |resampled diff| for {others[0]}: {spread:.4f} — the noise scale)")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
