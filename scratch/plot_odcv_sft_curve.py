# ABOUTME: Two-panel ODCV curve (misalignment rate + severity) vs synthdoc SFT percentage for
# ABOUTME: the four SFT-only Qwen3.6-27B arms, using existing ODCV results (grok-4.20+gemini-3.1-pro).
"""Render an ODCV misalignment-rate and severity curve vs SFT percentage.

Data are the `ours.overall` block (mr_pct, mr_ci95, mean_severity, severity_ci95) from each
arm's most-robust combined ODCV results.json. All four arms were judged by the same pair, so
the curve is apples-to-apples. Output: a PNG + a markdown mirror of the numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import timestamp

# (label, sft_pct, MR%, [mr_lo, mr_hi], severity, [sev_lo, sev_hi], n)
# 5% and 10% are 5-pass (n=339/349); 0% and 20% are their existing combined runs.
ARMS = [
    ("table2-only", 0, 43.6, [37.3, 51.7], 1.86, [1.59, 2.10], 328),
    ("t2+synthdoc-716", 5, 15.0, [11.4, 20.8], 0.68, [0.52, 0.88], 339),
    ("t2+synthdoc-1000", 10, 16.0, [12.2, 22.3], 0.69, [0.55, 0.91], 349),
    ("t2+synthdoc-2203", 20, 8.1, [5.3, 11.8], 0.47, [0.35, 0.62], 346),
]

BLUE = "#1f77d4"


def _asym_err(vals, cis):
    """Return matplotlib yerr [[lower_deltas], [upper_deltas]] from point values and [lo, hi] CIs."""
    lo = [v - ci[0] for v, ci in zip(vals, cis)]
    hi = [ci[1] - v for v, ci in zip(vals, cis)]
    return [lo, hi]


def main() -> None:
    """Draw the two-panel ODCV curve and write PNG + markdown mirror under output/report/."""
    x = [a[1] for a in ARMS]
    mr = [a[2] for a in ARMS]
    mr_ci = [a[3] for a in ARMS]
    sev = [a[4] for a in ARMS]
    sev_ci = [a[5] for a in ARMS]

    plt.rcParams.update({"font.size": 11, "axes.axisbelow": True})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.suptitle("ODCV", fontsize=17, fontweight="bold", y=1.02)

    for ax, y, yci, title, sub, ymax, ylabel in [
        (ax1, mr, mr_ci, "ODCV Misalignment Rate", "% of scenarios, lower is better", 55, "misalignment %"),
        (ax2, sev, sev_ci, "ODCV Severity Score", "0 to 2, lower is better", 2.0, ""),
    ]:
        ax.errorbar(x, y, yerr=_asym_err(y, yci), marker="s", markersize=6, color=BLUE,
                    capsize=4, linewidth=2, label="SFT-only")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=18)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, style="italic", color="#555")
        ax.set_xlabel("SFT Percentage (%)")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    ts = timestamp()
    png = f"output/report/odcv_sft_curve_{ts}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f">>> {png}")

    md = ["# ODCV vs synthdoc SFT percentage (SFT-only, Qwen3.6-27B)", "",
          "Judges: grok-4.20 + gemini-3.1-pro-preview (identical across all four arms).", "",
          "| SFT % | arm | MR % | MR 95% CI | severity | severity 95% CI | n |",
          "|---|---|---|---|---|---|---|"]
    for label, pct, m, mci, s, sci, n in ARMS:
        md.append(f"| {pct} | {label} | {m:.1f} | [{mci[0]}, {mci[1]}] | "
                  f"{s:.2f} | [{sci[0]}, {sci[1]}] | {n} |")
    md += ["", f"Plot: `{png}`", ""]
    mdp = f"output/report/odcv_sft_curve_{ts}.md"
    open(mdp, "w").write("\n".join(md))
    print(f">>> {mdp}")


if __name__ == "__main__":
    main()
