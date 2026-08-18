# ABOUTME: Three-arm cut of the fabrication ablation (baseline, +20% synth, +20% diverse):
# ABOUTME: overall fabrication rate and the "I ran this" claim rate, plotted side by side.

"""Same two metrics as the four-arm plot, restricted to the arms that share a question.

The diverse-scenario arm (da716) is the newest admixture; the comparison that matters is
against table2-only and against the synth admixture it is meant to replace. Both panels are
kept because the two metrics ranked the four arms differently and could do so again.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "src/eval/audits/petri")

from funnel_stats import clopper_pearson as cp  # noqa: E402
from src.utils import timestamp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JUDGED = {
    "t2only": ROOT / "output/fabrication_sweep/judged_20260810_220327/t2only.json",
    "t2synth": ROOT / "output/fabrication_sweep/judged_20260810_211552/t2synth.json",
    "da716": ROOT / "output/fabrication_sweep/judged_20260815_102514/da716.json",
}
ORDER = ["t2only", "t2synth", "da716"]
LABEL = {"t2only": "table2 only\n(baseline)", "t2synth": "+20% synth\n(non-diverse)",
         "da716": "5% synth"}
COLOR = {"t2only": "#8c8c8c", "t2synth": "#4c72b0", "da716": "#55a868"}
METRICS = ["fabricated", "claims_own_execution"]

records = {arm: json.loads(p.read_text()) for arm, p in JUDGED.items()}


def counts(arm: str) -> dict:
    """Total n, per-metric hit counts, and how many rows the judge left without a verdict.

    A row missing `fabricated` is one the judge returned no usable verdict for; it stays in
    the denominator (as in the published summaries) but is reported so the gap is visible.
    """
    rows = records[arm]
    out = {"n": len(rows), "no_verdict": sum(r.get("fabricated") is None for r in rows)}
    for m in METRICS:
        out[m] = sum(bool(r.get(m)) for r in rows)
    return out


def per_prompt(arm: str, metric: str) -> dict:
    """Per-prompt rate of `metric`, for the paired tests."""
    hits, tot = {}, {}
    for r in records[arm]:
        p = r["prompt_id"]
        hits[p] = hits.get(p, 0) + bool(r.get(metric))
        tot[p] = tot.get(p, 0) + 1
    return {p: hits[p] / tot[p] for p in hits}


def paired(a: str, b: str, metric: str) -> dict:
    """Prompt-paired comparison of arm `a` against arm `b` (sign test + mean gap)."""
    ra, rb = per_prompt(a, metric), per_prompt(b, metric)
    shared = sorted(set(ra) & set(rb))
    diffs = np.array([ra[p] - rb[p] for p in shared])
    wins = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    trials = len(diffs) - ties
    # Sign test on the prompts where the two arms differ.
    p = stats.binomtest(wins, trials, 0.5).pvalue if trials else 1.0
    return {"prompts": len(shared), "lower_on": wins, "ties": ties,
            "mean_gap_pts": diffs.mean() * 100, "p": p}


summary = {arm: counts(arm) for arm in ORDER}
stats_out = {m: {"diverse_vs_baseline": paired("da716", "t2only", m),
                 "diverse_vs_synth": paired("da716", "t2synth", m)} for m in METRICS}

for arm in ORDER:
    print(f"{arm:8s} n={summary[arm]['n']}  rows without a judge verdict: "
          f"{summary[arm]['no_verdict']}")

for m in METRICS:
    print(f"\n== {m}")
    for arm in ORDER:
        s = summary[arm]
        lo, hi = cp(s[m], s["n"])
        print(f"  {arm:8s} {s[m]:4d}/{s['n']}  {s[m] / s['n'] * 100:5.1f}%  "
              f"CI [{lo * 100:.1f}, {hi * 100:.1f}]")
    for k, v in stats_out[m].items():
        print(f"  {k}: gap {v['mean_gap_pts']:+.1f} pts, lower on {v['lower_on']}/"
              f"{v['prompts'] - v['ties']} differing prompts, p={v['p']:.2g}")

TITLES = {"fabricated": "Any fabricated benchmark data",
          "claims_own_execution": 'Falsely claims "I ran / measured this"'}

fig, axes = plt.subplots(1, 2, figsize=(14.0, 7))


def draw(ax, key, title):
    """One panel: per-arm rate of `key` as a share of all samples, with exact CIs."""
    vals, los, his, labels, colors = [], [], [], [], []
    for arm in ORDER:
        s = summary[arm]
        lo, hi = cp(s[key], s["n"])
        vals.append(s[key] / s["n"] * 100)
        los.append((s[key] / s["n"] - lo) * 100)
        his.append((hi - s[key] / s["n"]) * 100)
        labels.append(LABEL[arm])
        colors.append(COLOR[arm])

    bars = ax.bar(np.arange(len(ORDER)), vals, 0.58, color=colors,
                  edgecolor="black", linewidth=0.9,
                  yerr=[los, his], capsize=6,
                  error_kw={"linewidth": 1.4, "ecolor": "black"})
    for b, v, arm in zip(bars, vals, ORDER):
        s = summary[arm]
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.045, f"{v:.1f}%",
                ha="center", fontsize=16, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{s[key]}/{s['n']}",
                ha="center", fontsize=13, color="white", fontweight="bold")

    # Baseline reference: the whole question is whether an admixture beats table2-only.
    ax.axhline(vals[0], color="black", linestyle=":", linewidth=1.6, alpha=0.75)
    ax.text(2.42, vals[0], " baseline", va="center", fontsize=13, style="italic")

    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_title(title, fontsize=17)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("% of 992 responses per arm", fontsize=15)


for ax, m in zip(axes, METRICS):
    draw(ax, m, TITLES[m])

fig.suptitle("Diverse scenarios beat the non-diverse synth admixture on raw fabrication\n"
             "Qwen3.6-27B LoRA arms · 31 prompts × 32 samples × 3 arms = 2,976 generations",
             fontsize=17, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.90))

ts = timestamp()
OUT = ROOT / "output/plots"
OUT.mkdir(parents=True, exist_ok=True)
dest = OUT / f"three_arm_fabrication_{ts}.png"
fig.savefig(dest, dpi=200, bbox_inches="tight")
fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")

md = [f"# Fabrication ablation — baseline vs non-diverse synth vs diverse ({ts})", "",
      f"Plot: `{dest.relative_to(ROOT)}`", "",
      "| arm | label | n | fabricated | rate | CI95 | own-exec | rate | CI95 |",
      "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
for arm in ORDER:
    s = summary[arm]
    flo, fhi = cp(s["fabricated"], s["n"])
    elo, ehi = cp(s["claims_own_execution"], s["n"])
    md.append(f"| `{arm}` | {LABEL[arm].replace(chr(10), ' ')} | {s['n']} | {s['fabricated']} | "
              f"{s['fabricated'] / s['n'] * 100:.1f}% | [{flo * 100:.1f}, {fhi * 100:.1f}] | "
              f"{s['claims_own_execution']} | {s['claims_own_execution'] / s['n'] * 100:.1f}% | "
              f"[{elo * 100:.1f}, {ehi * 100:.1f}] |")
md += ["", "## Prompt-paired comparisons (sign test over 31 prompts)", "",
       "| metric | comparison | mean gap (pts) | lower on | p |",
       "| --- | --- | ---: | ---: | ---: |"]
for m in METRICS:
    for k, v in stats_out[m].items():
        md.append(f"| {m} | {k.replace('_', ' ')} | {v['mean_gap_pts']:+.1f} | "
                  f"{v['lower_on']}/{v['prompts'] - v['ties']} | {v['p']:.2g} |")
md += ["", "Sources:", ""] + [f"- `{arm}`: `{p.relative_to(ROOT)}`" for arm, p in JUDGED.items()]
mdpath = OUT / f"three_arm_fabrication_{ts}_results.md"
mdpath.write_text("\n".join(md) + "\n")
print(f"\nwrote {dest}\nwrote {mdpath}")
