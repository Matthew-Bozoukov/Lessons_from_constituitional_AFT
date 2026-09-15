# ABOUTME: ODCV misalignment rate of the no-constitution arm beside the principle-scoped baseline
# ABOUTME: and the base model, with 95% CIs, overall and per variant. One figure for the log.
#
# Run: uv run python scratch/no_constitution/plot_odcv.py

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.infra.huggingface import hf_download  # noqa: E402

OURS = Path("output/odcv/2026-09-03_qwen36_0_da_no_const_7_235606/results/results.json")
BASELINE = (
    "LASR-Callum/2026-08-21-odcv-difficult-advice-principle-scoped-702-eval",
    "combined2x_20260824_130511/results.json",
)
OUT = Path(
    "output/difficult_advice_no_constitution/2026-09-04_odcv_da_no_const_vs_baseline.png"
)
INK, MUTED, SURFACE, BLUE, GRID = "#1a1a19", "#6b6a63", "#fcfcfb", "#2a78d6", "#e6e5df"


def main() -> None:
    ours = json.load(OURS.open(encoding="utf-8"))
    base = json.load(
        open(hf_download(*BASELINE, repo_type="dataset"), encoding="utf-8")
    )
    arms = [
        ("base model\n80 cells", ours["published"]["overall"]),
        ("no constitution\n80 cells, 1 pass", ours["ours"]["overall"]),
        ("principle-scoped\n65 cells, 2 passes", base["ours"]["overall"]),
    ]
    variants = [("mandated", "mandated"), ("incentivized", "incentivized")]
    fig, axes = plt.subplots(
        1, 3, figsize=(10.5, 4), dpi=160, facecolor=SURFACE, sharey=True
    )
    panels = [("overall", [a[1] for a in arms])]
    for label, key in variants:
        panels.append(
            (
                label,
                [
                    ours["published"].get(key, {}),
                    ours["ours"][key],
                    base["ours"].get(key, {}),
                ],
            )
        )
    for ax, (title, blocks) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        xs = range(len(arms))
        vals = [b.get("mr_pct", float("nan")) for b in blocks]
        los = [v - b.get("mr_ci95", [v, v])[0] for v, b in zip(vals, blocks)]
        his = [b.get("mr_ci95", [v, v])[1] - v for v, b in zip(vals, blocks)]
        ax.bar(xs, vals, width=0.55, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(
            xs, vals, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4
        )
        for x, v in zip(xs, vals):
            ax.text(
                x, v + 2.5, f"{v:.1f}%", ha="center", va="bottom", color=INK, fontsize=9
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels([a[0] for a in arms], fontsize=7.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("misaligned rollouts (%), 95% CI", color=INK)
    axes[0].set_ylim(0, 75)
    fig.suptitle(
        "ODCV misalignment rate: no-constitution arm vs baseline",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    lines = [f"# {OUT.stem}", ""]
    for title, blocks in panels:
        for (name, _), b in zip(arms, blocks):
            if b:
                lines.append(
                    f"{title} | {name.replace(chr(10), ' ')} | MR {b.get('mr_pct')}% "
                    f"CI95 {b.get('mr_ci95')} | sev {b.get('mean_severity')} | "
                    f"n_rollouts {b.get('n_rollouts')}"
                )
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print("\n".join(lines[2:5]))


if __name__ == "__main__":
    main()
