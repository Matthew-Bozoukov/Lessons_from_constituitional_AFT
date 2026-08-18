# ABOUTME: Renders the two comparison figures with the peer-critique GOOD-ARM-ONLY arm added:
# ABOUTME: the four-arm fabrication panels and the document-type ODCV panels.
"""Plot the good-arm ablation against the published arms.

Run: uv run python scratch/plot_pcgood_arm_results.py

Replicates the two published figures with one arm added, in their style.

WHERE THE NUMBERS COME FROM, and why some error bars are missing. This arm's numbers are
computed here from its own artifacts: fabrication from
`output/fabrication_sweep/judged_*/summary.json`, ODCV from the run's `results.json`. The
comparison arms' numbers are the PUBLISHED values -- their raw judged files live on another
machine.

That asymmetry is visible in the figures rather than papered over:
  * Fabrication: counts were published for every arm (813/992 etc.), so exact
    Clopper-Pearson intervals are computed for all four and every bar carries one.
  * ODCV: only point estimates were published for the document-type arms. A CI is drawn for
    THIS arm, where the rollouts exist to compute one, and the others are drawn as points.
    Inventing intervals for them would be fabricating precision.

The x-axis says 7%, not the 5% on the earlier figures: 716 synth rows in a 10,000-row
mixture is 7.16%.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "src/eval/audits/petri")

from funnel_stats import clopper_pearson as cp  # noqa: E402

from src.utils import timestamp  # noqa: E402

OUT = Path("output/plots")
OUT.mkdir(parents=True, exist_ok=True)
TS = timestamp()

MINE = "pc_good716"
BASE_MR, BASE_SEV = 37.2, 1.43   # qwen3.6-27b base fp8 reference

# Published counts, n=992 per arm (31 prompts x 32 samples).
FAB = {
    "t2only":    {"label": "table2 only\n(baseline)", "n": 992, "fab": 813, "own": 82,
                  "color": "#8c8c8c"},
    "t2synth":   {"label": "+20% synth\n(non-diverse)", "n": 992, "fab": 559, "own": 75,
                  "color": "#4c72b0"},
    "da716":     {"label": "difficult advice\n7%", "n": 992, "fab": 461, "own": 58,
                  "color": "#55a868"},
    MINE:        {"label": "peer-critique\ngood-only 7%", "n": None, "fab": None,
                  "own": None, "color": "#c44e52"},
}
FAB_ORDER = ["t2only", "t2synth", "da716", MINE]

# ODCV, published point estimates on 65 cells; this arm measured on 63 (two cells produce
# no transcript). The cell count rides on the label so the difference is never silent.
ODCV = [
    ("difficult advice\n7%", 14.3, 0.7, None, "#4c72b0", "65 cells"),
    ("courtroom\n7%", 32.0, 1.4, None, "#dd8452", "69 cells"),
    ("peer-critique\n7%", 38.7, 1.7, None, "#c44e52", "65 cells"),
]


def _load_mine() -> None:
    """Fill this arm's fabrication counts from its judged summary."""
    files = sorted(glob.glob("output/fabrication_sweep/judged_*/summary.json"))
    assert files, "no judged fabrication summary found"
    for f in files:
        s = json.load(open(f))["summary"]
        if MINE in s:
            d = s[MINE]
            FAB[MINE].update(n=d["n"], fab=d["fabricated"], own=d["claims_own_execution"])
            return
    raise AssertionError(f"{MINE} not in any judged summary")


def _odcv_mine() -> tuple[float, float, tuple[float, float]]:
    """This arm's ODCV misalignment rate, severity and CI, from its results.json."""
    p = ("output/odcv_bench/qwen3_6-27b-lora-t2-9284-pc-good716-r64-dynbatch/"
         "combined4x/results.json")
    o = json.load(open(p))["ours"]["overall"]
    return o["mr_pct"], o["mean_severity"], tuple(o["mr_ci95"])


def _bar(ax, vals, los, his, labels, colors, title, ylabel, note, counts=None):
    """One panel of bars with asymmetric error bars and value labels."""
    x = np.arange(len(vals))
    err = np.array([[v - lo for v, lo in zip(vals, los)],
                    [hi - v for v, hi in zip(vals, his)]])
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="black", capsize=5, linewidth=1.4)
    for i, v in enumerate(vals):
        ax.text(i, his[i] + max(vals) * 0.035, f"{v:.1f}%" if "%" in ylabel else f"{v:.2f}",
                ha="center", fontweight="bold", fontsize=12)
        if counts:
            ax.text(i, v * 0.45, counts[i], ha="center", color="white",
                    fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    # Headroom for the value labels, which sit above the upper CI whisker.
    ax.set_ylim(0, max(his) * 1.16)
    if note:
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", fontsize=8,
                style="italic", color="#444")


def fabrication_figure() -> Path:
    """Two panels: any fabricated data, and falsely claiming own execution."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 7))
    for ax, key, title in ((a1, "fab", "Any fabricated benchmark data"),
                           (a2, "own", 'Falsely claims "I ran / measured this"')):
        vals, los, his, labels, colors, counts = [], [], [], [], [], []
        for arm in FAB_ORDER:
            d = FAB[arm]
            k, n = d[key], d["n"]
            lo, hi = cp(k, n)
            vals.append(k / n * 100); los.append(lo * 100); his.append(hi * 100)
            labels.append(d["label"]); colors.append(d["color"])
            counts.append(f"{k}/{n}")
        ax.axhline(vals[0], ls=":", color="black", lw=1.2)
        ax.text(len(vals) - 0.4, vals[0], " baseline", va="bottom", fontsize=9,
                style="italic")
        _bar(ax, vals, los, his, labels, colors, title,
             "% of 992 responses per arm", None, counts)
    fig.suptitle("Fabrication: the peer-critique good-arm slice barely beats no admixture,\n"
                 "and is the worst arm at claiming it ran something",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = OUT / f"four_arm_fabrication_pcgood_{TS}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def odcv_figure() -> Path:
    """Two panels: misalignment rate and mean severity, against the base fp8 line."""
    mr, sev, (lo, hi) = _odcv_mine()
    arms = ODCV + [("peer-critique\ngood-only 7%", mr, sev, (lo, hi), "#7f3f98", "63 cells")]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.5, 7))
    x = np.arange(len(arms))
    for ax, idx, base, ylab, title, ylim in (
            (a1, 1, BASE_MR, "Misalignment rate (%)", "Misalignment rate", (0, 100)),
            (a2, 2, BASE_SEV, "Mean severity (0-5)", "Mean severity", (0, 5))):
        vals = [a[idx] for a in arms]
        colors = [a[4] for a in arms]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.62)
        # Only this arm has an interval; the published arms are point estimates.
        ci = arms[-1][3]
        if idx == 1 and ci and ci[0] is not None:
            ax.errorbar([x[-1]], [vals[-1]],
                        yerr=[[vals[-1] - ci[0]], [ci[1] - vals[-1]]],
                        fmt="none", ecolor="black", capsize=5, linewidth=1.4)
        for i, v in enumerate(vals):
            ax.text(i, v + ylim[1] * 0.02, f"{v:.1f}" if idx == 1 else f"{v:.2f}",
                    ha="center", fontweight="bold", fontsize=12)
        ax.axhline(base, ls=":", color="black", lw=1.4,
                   label=f"base fp8 = {base}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a[0]}\n({a[5]})" for a in arms], fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=13); ax.set_ylim(*ylim)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.25, linestyle=":"); ax.set_axisbelow(True)
    fig.text(0.5, 0.015, "CI shown only where the rollouts exist to compute one; the "
             "published arms are point estimates. Cell counts differ per arm — see labels.",
             ha="center", fontsize=9, style="italic", color="#444")
    fig.suptitle("ODCV-Bench: dropping peer-critique's flawed arm changes nothing\n"
                 "Qwen3.6-27B LoRA arms - both peer-critique variants sit at or above base",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = OUT / f"odcv_doctype_arms_pcgood_{TS}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p, mr, sev, (lo, hi)


def main() -> None:
    """Render both figures and a markdown mirror of every number in them."""
    _load_mine()
    f1 = fabrication_figure()
    f2, mr, sev, ci = odcv_figure()

    d = FAB[MINE]
    md = [f"# Peer-critique good-arm-only vs the published arms ({TS})", "",
          f"Plots: `{f1}`, `{f2}`", "",
          "## ODCV-Bench", "",
          "| arm | MR% | severity | cells |", "| --- | ---: | ---: | --- |"]
    for label, m, s, _, _, cells in ODCV:
        md.append(f"| {label.replace(chr(10), ' ')} | {m} | {s} | {cells} |")
    md.append(f"| **peer-critique good-only 7%** | **{mr:.1f}** | **{sev:.2f}** | "
              f"63 cells |")
    md.append(f"| base fp8 (reference) | {BASE_MR} | {BASE_SEV} | — |")
    md += ["", f"This arm's CI95 on MR: [{ci[0]:.1f}, {ci[1]:.1f}].", "",
           "## Fabrication (31 prompts x 32 samples = 992 per arm)", "",
           "| arm | fabricated | rate | own-execution | rate |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for arm in FAB_ORDER:
        a = FAB[arm]
        md.append(f"| {a['label'].replace(chr(10), ' ')} | {a['fab']}/{a['n']} | "
                  f"{a['fab'] / a['n'] * 100:.1f}% | {a['own']}/{a['n']} | "
                  f"{a['own'] / a['n'] * 100:.1f}% |")
    md += ["",
           "Comparison arms are published values; this arm is computed from its own",
           "artifacts. Fabrication CIs are exact (Clopper-Pearson) for every arm because",
           "counts were published; ODCV CIs were not, so only this arm carries one.", ""]
    p = OUT / f"pcgood_arm_results_{TS}.md"
    p.write_text("\n".join(md))
    print(f"wrote {f1}\nwrote {f2}\nwrote {p}")


if __name__ == "__main__":
    main()
