# ABOUTME: Plots violation frequency against SFT dose in the style of the published audit-agents figure.
# ABOUTME: Primary single-panel chart, plus a supplementary two-panel decomposition.
"""Dose-response figures for the constitution Petri audit.

`violation_dose_response.png` reproduces the layout of the "Audit Agents /
Frequency of Violations" figure: one line, SFT percentage on x, violation
frequency on y, capped error bars, light horizontal grid.

Two deliberate differences from the published figure, annotated on the plot
rather than hidden:

1. **One curve, not two.** The published figure has `SFT-only` and
   `Midtraining + SFT`. No midtrained checkpoints exist for this model, so there
   is one curve.

2. **The y-axis is scaled to the data, and the error bars are large.** The
   published bars imply n in the high hundreds per point; ours are exact
   Clopper-Pearson intervals at n=10. Drawing a tight-looking bar we did not
   earn would be the actual error, so the axis expands to show the intervals
   honestly. The metric is still a 0-1 proportion, which the subtitle states.

`violation_decomposition.png` carries what a single curve structurally cannot
show: the harm-side / unhelpfulness-side split (a model that buys safety by
refusing moves those two lines in opposite directions), and the paired
difference against base, which is where the statistical power actually is at
this sample size.

Usage:
    python scripts/plot_violation_curve.py --results output/analysis/results.json \
        --out output/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Arm name -> difficult-advice share of training tokens, the x axis.
DOSE = {"base": 0, "dose-10-90": 10, "dose-20-80": 20, "dose-40-60": 40}

BLUE = "#2b7bba"
RED = "#e2553d"
GREY = "#6b7280"
GRID = "#d9dde3"


def _series(per_arm: dict, key: str):
    xs, ys, lo, hi, ns = [], [], [], [], []
    for arm in sorted(per_arm, key=lambda a: DOSE.get(a, 999)):
        if arm not in DOSE:
            continue
        r = per_arm[arm][key]
        if r["n"] == 0:
            continue
        xs.append(DOSE[arm])
        ys.append(r["rate"])
        lo.append(max(0.0, r["rate"] - r["ci95"][0]))
        hi.append(max(0.0, r["ci95"][1] - r["rate"]))
        ns.append(r["n"])
    return xs, ys, [lo, hi], ns


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def plot_primary(res: dict, out_dir: Path, adjudicated: bool) -> Path:
    per_arm = res["per_arm"]
    xs, ys, err, ns = _series(per_arm, "any_violation")

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.errorbar(
        xs, ys, yerr=err, marker="o", markersize=7, capsize=4, capthick=1.4,
        color=BLUE, ecolor=BLUE, linewidth=2.0, label="SFT-only", zorder=3,
    )

    fig.text(0.5, 0.955, "Audit Agents", ha="center", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.912, "Frequency of Violations", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.878, "0 to 1, lower is better", ha="center", fontsize=10,
             style="italic", color=GREY)

    top = max((y + e for y, e in zip(ys, err[1])), default=1.0)
    ax.set_ylim(0, min(1.0, top * 1.12))
    ax.set_xlim(-2.5, 42.5)
    ax.set_xticks(list(DOSE.values()))
    ax.set_xlabel("SFT Percentage (%)", fontsize=11, color="black", labelpad=8)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style(ax)

    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=7.5, color=GREY)

    # Kept to short lines: at 6.4in wide, a single long line runs off the canvas.
    note = ("One curve — no midtrained checkpoints exist for this model.\n"
            "Bars are exact Clopper-Pearson intervals at n=10 per point:\n"
            "wide because the sample is small, not because the metric is noisy.")
    if not adjudicated:
        note = "Judge flags, not yet hand-adjudicated.\n" + note
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.5,
             linespacing=1.5, color=(RED if not adjudicated else GREY))

    fig.subplots_adjust(top=0.85, bottom=0.26, left=0.12, right=0.96)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "violation_dose_response.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_decomposition(res: dict, out_dir: Path) -> Path:
    per_arm = res["per_arm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for key, colour, label in (
        ("harm_side", RED, "Harm-side (P1, P2, P3, P8)"),
        ("unhelpfulness_side", BLUE, "Unhelpfulness-side (P4–P7)"),
    ):
        xs, ys, err, _ = _series(per_arm, key)
        ax.errorbar(xs, ys, yerr=err, marker="o", markersize=6, capsize=4,
                    color=colour, linewidth=1.9, label=label)
    ax.set_title("Decomposed by violation family", fontsize=12, fontweight="bold")
    ax.set_xlabel("SFT Percentage (%)")
    ax.set_ylabel("Frequency of violations")
    ax.set_xticks(list(DOSE.values()))
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    ax.text(0.5, -0.19,
            "Both are violations of the same document. A model that buys safety by\n"
            "refusing moves the red line down and the blue line up. Blue stayed at zero.",
            transform=ax.transAxes, ha="center", fontsize=8, color=GREY)

    ax = axes[1]
    paired = res.get("paired_vs_base") or {}
    if paired:
        arms = [a for a in sorted(paired, key=lambda x: DOSE.get(x, 999)) if a in DOSE]
        xs = [DOSE[a] for a in arms]
        diffs, labels = [], []
        for a in arms:
            p = paired[a]
            n = p["n_pairs"] or 1
            diffs.append((p["base_safe_arm_violation"] - p["base_violation_arm_safe"]) / n)
            labels.append(f"p={p['mcnemar_exact_p']:.3g}")
        ax.axhline(0, color=GREY, linewidth=1, linestyle="--")
        ax.plot(xs, diffs, marker="D", markersize=6, color="#7c3aed", linewidth=1.9)
        for x, d, lab in zip(xs, diffs, labels):
            ax.annotate(lab, (x, d), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=GREY)
        ax.set_title("Paired vs base (matched seed)", fontsize=12, fontweight="bold")
        ax.set_xlabel("SFT Percentage (%)")
        ax.set_ylabel("Paired change vs base")
        ax.set_xticks(list(DOSE.values()))
        _style(ax)
        ax.text(0.5, -0.19,
                "Negative = the dose reduced violations. McNemar exact on discordant\n"
                "pairs; removes between-scenario variance, the dominant noise source.",
                transform=ax.transAxes, ha="center", fontsize=8, color=GREY)
    else:
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = out_dir / "violation_decomposition.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def mirror(res: dict, out_dir: Path, adjudicated: bool) -> Path:
    per_arm = res["per_arm"]
    L = ["# Violation frequency vs SFT dose", "",
         f"**Stage:** {'hand-adjudicated' if adjudicated else 'judge flags, NOT yet adjudicated'}", "",
         "| SFT % | arm | any violation | 95% CI | harm-side | unhelpfulness-side | n |",
         "|---|---|---|---|---|---|---|"]
    for arm in sorted(per_arm, key=lambda a: DOSE.get(a, 999)):
        if arm not in DOSE:
            continue
        d = per_arm[arm]
        av, hs, us = d["any_violation"], d["harm_side"], d["unhelpfulness_side"]
        L.append(f"| {DOSE[arm]} | `{arm}` | {av['rate']:.3f} | "
                 f"[{av['ci95'][0]:.3f}, {av['ci95'][1]:.3f}] | {hs['rate']:.3f} | "
                 f"{us['rate']:.3f} | {av['n']} |")
    paired = res.get("paired_vs_base") or {}
    if paired:
        L += ["", "## Paired vs base", "",
              "| arm | pairs | base bad -> arm safe | base safe -> arm bad | McNemar p |",
              "|---|---|---|---|---|"]
        for a in sorted(paired, key=lambda x: DOSE.get(x, 999)):
            p = paired[a]
            L.append(f"| `{a}` | {p['n_pairs']} | {p['base_violation_arm_safe']} | "
                     f"{p['base_safe_arm_violation']} | {p['mcnemar_exact_p']:.4g} |")
    L += ["", "## Deviations from the published figure", "",
          "- **One curve, not two.** No midtrained checkpoints exist for this model.",
          "- **Doses are 0/10/20/40%**, not 0/5/10/25% — fixed by which adapters exist.",
          "- **Error bars are exact Clopper-Pearson at n=10**, not standard errors at n in",
          "  the high hundreds. They are wide because the sample is small.", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "violation_dose_response.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adjudicated", action="store_true")
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    print("wrote", plot_primary(res, out, args.adjudicated))
    print("wrote", plot_decomposition(res, out))
    print("wrote", mirror(res, out, args.adjudicated))


if __name__ == "__main__":
    main()
