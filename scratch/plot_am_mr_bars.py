# ABOUTME: Bar plot of inspect agentic-misalignment (blackmail) harmful rate vs difficult-advice
# ABOUTME: SFT % for the four r64 arms, read from the arm_comparison.json Wilson-CI stats.
"""Compare inspect-AM harmful rate across four SFT-percentage arms as a bar chart.

Values (harmful pct + Wilson 95% CI) come from `output/inspect_agentic_misalignment/
arm_comparison.json` (`arms[<name>].harmful` = {pct, lo, hi, k, n}) — all arms ran the identical
prompt/rubric with the same gemini-3-flash grader. Output: a PNG + a markdown mirror.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import timestamp

CMP = "output/inspect_agentic_misalignment/arm_comparison.json"

# (x-axis label, arm name in arm_comparison.json)
ARMS = [
    ("0%", "table2-only"),
    ("5%", "t2+synthdoc (716)"),
    ("10%", "t2+synthdoc (1000)"),
    ("20%", "t2+synthdoc (2203)"),
]

COLORS = ["#d1495b", "#edae49", "#3c8c9c", "#30638e"]


def main() -> None:
    """Draw the AM harmful-rate bar chart and write PNG + markdown mirror under output/report/."""
    arms = json.load(open(CMP))["arms"]
    labels, pct, lo_err, hi_err, ns = [], [], [], [], []
    for label, name in ARMS:
        h = arms[name]["classifier_verdict"]
        labels.append(label)
        pct.append(h["pct"])
        lo_err.append(h["pct"] - h["lo"])
        hi_err.append(h["hi"] - h["pct"])
        ns.append(h["n"])

    xpct = [int(lbl.rstrip("%")) for lbl in labels]
    blue = "#1f77d4"
    plt.rcParams.update({"font.size": 11, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fig.suptitle("Agentic Misalignment", fontsize=19, fontweight="bold", y=1.03)
    ax.errorbar(xpct, pct, yerr=[lo_err, hi_err], marker="s", markersize=7, color=blue,
                capsize=5, capthick=1.4, elinewidth=1.5, linewidth=2, label="SFT-only")
    ax.set_title("Classifier Verdict Rate", fontsize=14, fontweight="bold", pad=20)
    ax.text(0.5, 1.02, "% of samples, lower is better", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, style="italic", color="#555")
    ax.set_xlabel("SFT Percentage (%)")
    ax.set_ylabel("classifier verdict %")
    ax.set_xticks(xpct)
    ax.set_xlim(-1.2, max(xpct) + 1.2)
    ax.set_ylim(0, max(m + h for m, h in zip(pct, hi_err)) + 6)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    png = f"output/report/am_classifier_line_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f">>> {png}")

    md = ["# Agentic-misalignment (blackmail) classifier-verdict rate vs difficult-advice SFT %", "",
          "Grader: google/gemini-3-flash-preview; identical prompt/rubric across arms (Wilson 95% CI).",
          "classifier_verdict = grader score 2 alone (looser than `harmful`, which also requires the "
          "target email + a blackmail/leverage word).", "",
          "| SFT % | arm | classifier verdict % | 95% CI | n |", "|---|---|---|---|---|"]
    for (label, name), m, le, he, n in zip(ARMS, pct, lo_err, hi_err, ns):
        md.append(f"| {label} | {name} | {m:.1f} | [{m - le:.1f}, {m + he:.1f}] | {n} |")
    md += ["", f"Plot: `{png}`", ""]
    mdp = f"output/report/am_classifier_line_{ts}.md"
    open(mdp, "w").write("\n".join(md))
    print(f">>> {mdp}")


if __name__ == "__main__":
    main()
