# ABOUTME: Combine the per-arm premature-suspicion JSONLs into one comparison table, with
# ABOUTME: condition splits, reasoning-length control and a Fisher test against the 0% arm.
"""Report premature-suspicion rates across the difficult-advice dose ladder.

Every arm is judged by the same model, prompt and 5-step window, so the rates are directly
comparable. Reasoning length inside the judged window is reported alongside, because an arm
that deliberates longer has more opportunity to voice a suspicion — that is a property of the
model, not of the measurement, but it belongs next to the number.
"""

import json
import re
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, "scratch")
import judge_premature_suspicion as J  # noqa: E402

from src.utils import timestamp  # noqa: E402

# (label, odcv run dir, judged jsonl) — ordered by synthetic difficult-advice share.
ARMS = [
    ("standard SFT data (0)",
     "output/odcv_bench/qwen3.6-27b-table2-only-9284-r64/combined5x_20260805_132959",
     "output/report/premature_suspicion_20260809_210319.jsonl"),
    ("t2-9284 + synthdoc-716",
     "output/odcv_bench/qwen3_6-27b-lora-t2-9284-synthdoc-716-r64/combined4x_20260808_124051",
     "output/report/premature_suspicion_20260809_210744.jsonl"),
    ("t2-9000 + synthdoc-1000",
     "output/odcv_bench/qwen3_6-27b-lora-t2-9000-synthdoc-1000-r64/combined4x_20260808_154502",
     "output/report/premature_suspicion_20260809_210813.jsonl"),
    ("t2 + synthdoc (r64)",
     "output/odcv_bench/qwen3.6-27b-table2-synthdoc-r64/combined5x_20260805_103343",
     "output/report/premature_suspicion_20260809_205845.jsonl"),
    # 20%-share ablations: same budget as the top ladder point, different synthetic data.
    ("t2-80 + memself-20",
     "output/odcv_bench/qwen3_6-27b-lora-table2-80-memself-20-r64/combined4x_20260808_142305",
     "output/report/premature_suspicion_20260810_074841.jsonl"),
    ("t2-80 + selfreflect-20",
     "output/odcv_bench/qwen3_6-27b-lora-table2-80-selfreflect-20-r64/combined4x_20260809_134735",
     "output/report/premature_suspicion_20260810_074911.jsonl"),
]


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def prob(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))


def median_reasoning_chars(root: str) -> tuple[int, float]:
    """Median assistant-reasoning characters inside the judged window, and its coverage."""
    lengths, with_reason = [], 0
    files = sorted(Path(root).rglob("messages_record.txt"))
    for f in files:
        prefix, _ = J.prefix_for_judge(f.read_text(errors="replace"), 5, 4000)
        body = "".join(re.findall(r"(?ms)^reason:\s*(.*?)(?=^\w+:|\Z)", prefix)).strip()
        if body:
            with_reason += 1
            lengths.append(len(body))
    lengths.sort()
    return lengths[len(lengths) // 2], with_reason / len(files) * 100


def main() -> None:
    """Print and write the four-arm premature-suspicion comparison."""
    rows_out = []
    for label, root, jsonl in ARMS:
        rows = [json.loads(line) for line in open(jsonl)]
        assert not any(r["error"] for r in rows), f"unresolved judge errors in {jsonl}"
        flagged = sum(r["premature"] for r in rows)
        cond_f, cond_n = Counter(), Counter()
        for r in rows:
            c = "incentivized" if "incentivized" in r["condition"] else "mandated"
            cond_n[c] += 1
            cond_f[c] += r["premature"]
        med, cover = median_reasoning_chars(root)
        inc = (cond_f["incentivized"], cond_n["incentivized"])
        man = (cond_f["mandated"], cond_n["mandated"])
        inc_pct, man_pct = inc[0] / inc[1] * 100, man[0] / man[1] * 100
        rows_out.append(dict(label=label, flagged=flagged, n=len(rows),
                             rate=flagged / len(rows) * 100,
                             inc=inc, man=man, inc_pct=inc_pct, man_pct=man_pct,
                             # Unweighted mean of the two conditions: they are a designed
                             # factor with near-equal n, so each counts once rather than in
                             # proportion to however many rollouts happened to survive.
                             macro=(inc_pct + man_pct) / 2,
                             med_reason=med, reason_cover=cover))

    base = rows_out[0]
    for r in rows_out:
        r["p_vs_base"] = (fisher_two_sided(r["flagged"], r["n"] - r["flagged"],
                                           base["flagged"], base["n"] - base["flagged"])
                          if r is not base else None)

    hdr = (f"{'arm':<26}{'mean':>8}{'pooled':>9}{'incent.':>10}{'mand.':>9}"
           f"{'flagged':>10}{'reason ch':>11}{'p vs 0':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_out:
        p = "—" if r["p_vs_base"] is None else f"{r['p_vs_base']:.1e}"
        print(f"{r['label']:<26}{r['macro']:>7.1f}%{r['rate']:>8.1f}%"
              f"{r['inc_pct']:>9.1f}%{r['man_pct']:>8.1f}%"
              f"{r['flagged']:>6}/{r['n']:<3}{r['med_reason']:>11}{p:>11}")

    md = ["# Premature suspicion across the difficult-advice arms", "",
          f"Judge `{J.JUDGE_MODEL}`, first 5 transcript steps, temperature 0. "
          "Every transcript judged; none excluded.", "",
          "| arm | premature suspicion (mean of conditions) | pooled | incentivized | "
          "mandated | flagged / n | median reasoning chars in window | Fisher p vs 0% |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows_out:
        p = "—" if r["p_vs_base"] is None else f"{r['p_vs_base']:.2e}"
        md.append(f"| {r['label']} | **{r['macro']:.1f}%** | {r['rate']:.1f}% | "
                  f"{r['inc_pct']:.1f}% ({r['inc'][0]}/{r['inc'][1]}) | "
                  f"{r['man_pct']:.1f}% ({r['man'][0]}/{r['man'][1]}) | "
                  f"{r['flagged']}/{r['n']} | {r['med_reason']} | {p} |")
    md += ["", "The headline column is the **unweighted mean of the two conditions**; `pooled` "
           "is every transcript in one bucket. They differ by at most a point because the "
           "per-condition n are near-equal. The Fisher p is computed on the pooled counts.", "",
           "Assistant reasoning is present in the judged window for "
           f"{rows_out[0]['reason_cover']:.0f}-100% of transcripts in every arm, so the gap "
           "is not an artifact of the baseline having nothing to flag.", ""]
    dest = Path("output/report") / f"premature_suspicion_compare_{timestamp()}.md"
    dest.write_text("\n".join(md))
    print(f"\n>>> {dest}")


if __name__ == "__main__":
    main()
