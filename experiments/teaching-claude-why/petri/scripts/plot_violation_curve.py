# ABOUTME: Plots violation frequency against SFT dose, in the format of the audit-agents
# ABOUTME: figure in Anthropic's synthetic-document finetuning write-up, plus two panels it lacks.
"""Dose-response figure for the constitution Petri audit.

Panel A reproduces the format of the "Audit Agents / Frequency of Violations"
figure: violation frequency on a 0-1 axis against SFT percentage, lower better,
with error bars.

Two differences from that figure are structural and are annotated on the plot
rather than hidden:

1. **One condition, not two.** The published figure has an `SFT-only` line and a
   `Midtraining + SFT` line. We have only SFT-only arms -- no midtrained
   checkpoints exist for this model -- so there is one curve.
2. **Wider error bars.** The published bars imply n in the high hundreds per
   point. Ours are exact Clopper-Pearson intervals at the n we can afford. An
   interval is not decoration; drawing a tight-looking bar we did not earn would
   be the actual error.

Panel B decomposes into the two violation families, which is where a dose sweep
earns its keep: harm-side violations falling while unhelpfulness-side violations
rise is the trade-off, and a single curve cannot show it.

Panel C plots the paired difference against base on matched seed and epoch. This
is where the statistical power actually lives at our sample size.

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
DOSE = {
    "base": 0,
    "dose-10-90": 10,
    "dose-20-80": 20,
    "dose-40-60": 40,
}

BLUE = "#2b7bba"
RED = "#e2553d"
GREY = "#6b7280"


def _series(per_arm: dict, key: str) -> tuple[list[int], list[float], list[list[float]], list[int]]:
    xs, ys, errs, ns = [], [], [[], []], []
    for arm in sorted(per_arm, key=lambda a: DOSE.get(a, 999)):
        if arm not in DOSE:
            continue
        r = per_arm[arm][key]
        if r["n"] == 0:
            continue
        xs.append(DOSE[arm])
        ys.append(r["rate"])
        errs[0].append(max(0.0, r["rate"] - r["ci95"][0]))
        errs[1].append(max(0.0, r["ci95"][1] - r["rate"]))
        ns.append(r["n"])
    return xs, ys, errs, ns


def plot(res: dict, out_dir: Path, adjudicated: bool) -> Path:
    per_arm = res["per_arm"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---------- Panel A: the requested chart ----------
    ax = axes[0]
    xs, ys, errs, ns = _series(per_arm, "any_violation")
    ax.errorbar(
        xs, ys, yerr=errs, marker="s", markersize=6, capsize=4,
        color=BLUE, label="SFT-only", linewidth=1.8,
    )
    ax.set_title("Audit Agents", fontsize=14, fontweight="bold")
    ax.text(
        0.5, 1.02, "Frequency of Violations", transform=ax.transAxes,
        ha="center", fontsize=11, fontweight="bold",
    )
    ax.text(
        0.5, -0.20, "0 to 1, lower is better", transform=ax.transAxes,
        ha="center", fontsize=9, style="italic", color=GREY,
    )
    ax.set_xlabel("SFT Percentage (%)")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(DOSE.values()))
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7, color=GREY)

    # ---------- Panel B: the two families ----------
    ax = axes[1]
    for key, colour, label in (
        ("harm_side", RED, "Harm-side (P1, P2, P3, P8)"),
        ("unhelpfulness_side", BLUE, "Unhelpfulness-side (P4-P7)"),
    ):
        xs, ys, errs, _ = _series(per_arm, key)
        ax.errorbar(xs, ys, yerr=errs, marker="o", markersize=5, capsize=4,
                    color=colour, label=label, linewidth=1.8)
    ax.set_title("Decomposed by violation family", fontsize=12, fontweight="bold")
    ax.set_xlabel("SFT Percentage (%)")
    ax.set_ylabel("Frequency of violations")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(DOSE.values()))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.text(
        0.5, -0.20,
        "Both are violations of the same document.\nA model that refuses everything moves the red line down and the blue line up.",
        transform=ax.transAxes, ha="center", fontsize=8, color=GREY,
    )

    # ---------- Panel C: paired difference vs base ----------
    ax = axes[2]
    paired = res.get("paired_vs_base") or {}
    if paired:
        arms = sorted(paired, key=lambda a: DOSE.get(a, 999))
        xs = [DOSE[a] for a in arms if a in DOSE]
        diffs, labels = [], []
        for a in arms:
            if a not in DOSE:
                continue
            p = paired[a]
            n = p["n_pairs"]
            b, c = p["base_violation_arm_safe"], p["base_safe_arm_violation"]
            diffs.append((c - b) / n if n else 0.0)
            labels.append(f"p={p['mcnemar_exact_p']:.3g}")
        ax.axhline(0, color=GREY, linewidth=1, linestyle="--")
        ax.plot(xs, diffs, marker="D", markersize=6, color="#7c3aed", linewidth=1.8)
        for x, d, lab in zip(xs, diffs, labels):
            ax.annotate(lab, (x, d), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7, color=GREY)
        ax.set_ylabel("Paired change in violation rate vs base")
        ax.set_title("Paired vs base (matched seed + epoch)", fontsize=12, fontweight="bold")
        ax.set_xlabel("SFT Percentage (%)")
        ax.set_xticks(list(DOSE.values()))
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.text(
            0.5, -0.20,
            "Negative = the dose reduced violations.\nMcNemar exact on discordant pairs; removes between-scenario variance.",
            transform=ax.transAxes, ha="center", fontsize=8, color=GREY,
        )
    else:
        ax.text(0.5, 0.5, "no base arm -\nno paired comparison", ha="center",
                va="center", transform=ax.transAxes, color=GREY)
        ax.set_axis_off()

    stage = "hand-adjudicated" if adjudicated else "JUDGE FLAGS, NOT YET ADJUDICATED"
    fig.suptitle(
        f"Constitution violation frequency vs SFT dose  -  {stage}",
        fontsize=10, color=("black" if adjudicated else RED), y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "violation_dose_response.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def mirror(res: dict, out_dir: Path, adjudicated: bool) -> Path:
    """Markdown mirror so the numbers are greppable without opening the PNG."""
    per_arm = res["per_arm"]
    L = ["# Violation frequency vs SFT dose", ""]
    L.append(f"**Stage:** {'hand-adjudicated' if adjudicated else 'judge flags, NOT yet adjudicated'}")
    L.append("")
    L.append("| SFT % | arm | any violation | harm-side | unhelpfulness-side | n |")
    L.append("|---|---|---|---|---|---|")
    for arm in sorted(per_arm, key=lambda a: DOSE.get(a, 999)):
        if arm not in DOSE:
            continue
        d = per_arm[arm]
        av, hs, us = d["any_violation"], d["harm_side"], d["unhelpfulness_side"]
        L.append(
            f"| {DOSE[arm]} | `{arm}` | {av['rate']:.3f} "
            f"[{av['ci95'][0]:.3f}, {av['ci95'][1]:.3f}] | {hs['rate']:.3f} | "
            f"{us['rate']:.3f} | {av['n']} |"
        )
    paired = res.get("paired_vs_base") or {}
    if paired:
        L += ["", "## Paired vs base", "",
              "| arm | pairs | base bad -> arm safe | base safe -> arm bad | McNemar p |",
              "|---|---|---|---|---|"]
        for a in sorted(paired, key=lambda x: DOSE.get(x, 999)):
            p = paired[a]
            L.append(
                f"| `{a}` | {p['n_pairs']} | {p['base_violation_arm_safe']} | "
                f"{p['base_safe_arm_violation']} | {p['mcnemar_exact_p']:.4g} |"
            )
    L += ["", "## Deviations from the published figure", "",
          "- **One curve, not two.** No midtrained checkpoints exist for this model.",
          "- **Doses are 0/10/20/40%**, not 0/5/10/25%. Fixed by which adapters exist.",
          "- **Error bars are exact Clopper-Pearson** at our n, not standard errors at n in",
          "  the high hundreds. They are wider because our sample is smaller.", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "violation_dose_response.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.json from violation_rates.py")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--adjudicated", action="store_true",
        help="set only after flagged transcripts have been hand-adjudicated",
    )
    args = ap.parse_args()
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    png = plot(res, Path(args.out), args.adjudicated)
    md = mirror(res, Path(args.out), args.adjudicated)
    print(f"wrote {png}\nwrote {md}")


if __name__ == "__main__":
    main()
