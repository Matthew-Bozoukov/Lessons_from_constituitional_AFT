# ABOUTME: Effect sizes for the confidence autorater: score distributions (ceiling share), paired win/tie/loss,
# ABOUTME: Cohen's d_z and Cliff's delta per dimension; Fisher tests + Wilson CIs for the rollout-side contrasts.
# Run: uv run python scratch/confidence/effect_sizes.py
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

from scratch.confidence.common import KEYS

OUT = Path("output/confidence")
ARMS = ["grok", "capped", "sonnet", "gpt"]
ROLL = {
    "grok": "grok",
    "sonnet_concise": "capped",
    "sonnet_normal": "sonnet",
    "gpt": "gpt",
}


def rows_of(p):
    return [json.loads(l) for l in p.open() if l.strip()]


def ok(d, ch):
    return (
        "error" not in d
        and isinstance(d.get(ch), dict)
        and all(isinstance(d[ch].get(k), (int, float)) for k in KEYS)
    )


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def cliffs(x, y):
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    return (gt - lt) / (len(x) * len(y))


def main():
    corp = rows_of(sorted(OUT.glob("corpus_terra_full*.jsonl"))[-1])
    by = defaultdict(dict)
    for d in corp:
        if ok(d, "reasoning") and ok(d, "reply"):
            by[d["corpus"]][d["scenario_id"]] = d
    ids = sorted(set.intersection(*(set(by[a]) for a in ARMS)))
    L = [
        "# Effect sizes — confidence autorater",
        "",
        f"n = {len(ids)} paired scenarios",
        "",
    ]
    for ch in ("reasoning", "reply"):
        L += [
            f"## {ch}",
            "",
            "| dimension | arm | mean | % scored 7 | % scored 6 | % ≤5 | vs sonnet: win/tie/loss % | d_z | Cliff's δ |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for k in KEYS:
            for a in ARMS:
                xs = [by[a][i][ch][k] for i in ids]
                ys = [by["sonnet"][i][ch][k] for i in ids]
                p7 = 100 * sum(x == 7 for x in xs) / len(xs)
                p6 = 100 * sum(x == 6 for x in xs) / len(xs)
                p5 = 100 * sum(x <= 5 for x in xs) / len(xs)
                if a == "sonnet":
                    L.append(
                        f"| {k} | {a} | {st.mean(xs):.2f} | {p7:.0f} | {p6:.0f} | {p5:.0f} | – | – | – |"
                    )
                    continue
                d = [x - y for x, y in zip(xs, ys)]
                win = 100 * sum(v > 0 for v in d) / len(d)
                tie = 100 * sum(v == 0 for v in d) / len(d)
                loss = 100 * sum(v < 0 for v in d) / len(d)
                dz = st.mean(d) / (st.pstdev(d) or 1e-9)
                L.append(
                    f"| {k} | {a} | {st.mean(xs):.2f} | {p7:.0f} | {p6:.0f} | {p5:.0f} | {win:.0f}/{tie:.0f}/{loss:.0f} | {dz:+.2f} | {cliffs(xs, ys):+.2f} |"
                )
        L.append("")

    # rollouts
    R = [
        d
        for d in rows_of(sorted(OUT.glob("rollouts_terra_full*.jsonl"))[-1])
        if ok(d, "reasoning")
    ]
    regs = {}
    for d in rows_of(
        sorted(Path("output/four_mos_rollouts").glob("registers_*.jsonl"))[-1]
    ):
        regs[(d["arm"], d["cell"], d["rollout"])] = d
    for d in R:
        d["commit"] = bool(
            regs.get((d["arm"], d["cell"], d["rollout"]), {}).get("r1_commit")
        )
        d["hi"] = d["reasoning"]["overall_confidence"] >= 6
        d["dec"] = d["reasoning"]["decisiveness"] >= 6
    L += [
        "## Rollouts: first-block confidence × first-block commitment (pooled, n = %d)"
        % len(R),
        "",
    ]

    def cell(pred):
        v = [d for d in R if pred(d)]
        k = sum(d["violation"] for d in v)
        lo, hi = wilson(k, len(v))
        return k, len(v), lo, hi

    rows = {
        "high conf, commitment": cell(lambda d: d["hi"] and d["commit"]),
        "high conf, no commitment": cell(lambda d: d["hi"] and not d["commit"]),
        "low conf, commitment": cell(lambda d: (not d["hi"]) and d["commit"]),
        "low conf, no commitment": cell(lambda d: (not d["hi"]) and not d["commit"]),
    }
    L += ["| cell | violations / n | MR | 95% CI |", "|---|---|---|---|"]
    for name, (k, n, lo, hi) in rows.items():
        L.append(f"| {name} | {k}/{n} | {100 * k / n:.1f}% | [{lo:.1f}, {hi:.1f}] |")
    a, b = rows["high conf, commitment"], rows["high conf, no commitment"]
    p_commit = fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
    c = rows["low conf, commitment"]
    p_conf = fisher_exact([[a[0], a[1] - a[0]], [c[0], c[1] - c[0]]])[1]
    L += [
        "",
        f"- commitment effect within high-confidence blocks: {100 * a[0] / a[1]:.1f}% vs {100 * b[0] / b[1]:.1f}%, Fisher p = {p_commit:.1e}",
        f"- confidence effect within commitment blocks: {100 * a[0] / a[1]:.1f}% vs {100 * c[0] / c[1]:.1f}%, Fisher p = {p_conf:.2f}",
        "",
    ]
    L += [
        "## Within arm: decisive (≥6) vs not, violation rate with Fisher p",
        "",
        "| arm | decisive: viol/n (MR) | not: viol/n (MR) | Fisher p |",
        "|---|---|---|---|",
    ]
    inv = {v: k for k, v in ROLL.items()}
    for a_ in ARMS:
        rs = [d for d in R if d["arm"] == inv[a_]]
        x = [d for d in rs if d["dec"]]
        y = [d for d in rs if not d["dec"]]
        kx, ky = sum(d["violation"] for d in x), sum(d["violation"] for d in y)
        p = (
            fisher_exact([[kx, len(x) - kx], [ky, len(y) - ky]])[1]
            if x and y
            else float("nan")
        )
        L.append(
            f"| {a_} | {kx}/{len(x)} ({100 * kx / max(len(x), 1):.1f}%) | {ky}/{len(y)} ({100 * ky / max(len(y), 1):.1f}%) | {p:.3f} |"
        )
    out = OUT / "effect_sizes.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
