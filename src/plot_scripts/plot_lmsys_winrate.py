# ABOUTME: GDM-style "LMSYS SxS: Win Rate vs Baseline" figure — one SFT line over
# ABOUTME: synthetic fraction, 95% CI bars, 50=tie rule. Reads the latest report JSON.

"""Single-line dose-response figure matching the GDM synthetic-document post.

Reads the newest `output/capability_eval/report/*/results.json` (written by
`capability_report.py`) so the plot can never disagree with the canonical numbers.
The matched arms (B 10% -> C 20% -> D 40%) form the line; arm A (0%) is drawn as a
detached open marker because its training recipe differs (2 epochs, packing on) and
connecting it would draw a dose-response claim the design cannot support.

    uv run python src/plot_scripts/plot_lmsys_winrate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2a78d6"   # categorical slot 1 (constieval/plots_theme.py)
INK = "#3a3a3a"
MUTED = "#8a8a8a"


def main(
    report_dir: str = "",
    slice_name: str = "hard_prompt",
    measure: str = "uncontrolled",
    out: str = "",
) -> None:
    """Render the win-rate-vs-synthetic-fraction figure.

    Args:
        report_dir: A specific report directory; default = newest under output/.
        slice_name: Which judged slice to plot.
        measure: `uncontrolled` (raw SxS, what the GDM figure shows) or `controlled`.
        out: Output PNG path; default = `<report_dir>/lmsys_winrate_gdm_style.png`.
    """
    base = Path("output/capability_eval/report")
    rdir = Path(report_dir) if report_dir else sorted(base.iterdir())[-1]
    results = json.loads((rdir / "results.json").read_text())

    line, detached = [], []
    for arm, block in results.items():
        frac = block.get("synthetic_fraction")
        sl = block["by_slice"].get(slice_name)
        if frac is None or sl is None:  # arm_base has no fraction; skip
            continue
        stats = sl[measure]
        point = (
            frac * 100,
            stats["mean"] * 100,
            (stats["mean"] - stats["ci_lower"]) * 100,
            (stats["ci_upper"] - stats["mean"]) * 100,
            sl["split"]["n_prompts"],
            arm,
        )
        # Arm A's recipe (2 epochs, packing) differs from B/C/D; it is not on the
        # dose-response line. See configs/capability_eval.yaml arm_a note.
        (detached if arm == "arm_a_synth00" else line).append(point)

    line.sort()
    xs = [p[0] for p in line]
    ys = [p[1] for p in line]
    yerr = [[p[2] for p in line], [p[3] for p in line]]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(50, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(0.99, 51.2, "50 = tie", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color=MUTED)

    ax.errorbar(
        xs, ys, yerr=yerr, color=BLUE, lw=2.0, marker="o", ms=8,
        capsize=4, capthick=1.5, elinewidth=1.5, zorder=3, label="SFT (matched arms)",
    )
    for x, y, lo, hi, n, _arm in detached:
        ax.errorbar([x], [y], yerr=[[lo], [hi]], color=BLUE, lw=0, marker="o", ms=8,
                    markerfacecolor="white", capsize=4, capthick=1.5, elinewidth=1.5,
                    zorder=3)
        ax.annotate("0%: unmatched recipe\n(2 ep, packing)", (x, y + hi),
                    xytext=(0, 8), textcoords="offset points", ha="left",
                    fontsize=8, color=MUTED)

    for x, y, lo, *_rest, arm in line:
        n = _rest[1]
        ax.annotate(f"n={n}", (x, y - lo), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)

    ax.set_ylim(0, 100)
    ax.set_xlim(-4, max(xs) + 6)
    ax.set_xlabel("Synthetic document percentage in SFT mix (%)", color=INK)
    ax.set_ylabel(f"Win rate vs baseline ({results['arm_b_synth10']['label']} arm) %",
                  color=INK)
    ax.set_title("LMSYS SxS: Win Rate vs Baseline", color=INK, fontsize=13,
                 fontweight="bold", pad=22)
    ax.text(0.5, 1.015, "50 = tie, higher is better · arena-hard-v2.0 "
            f"{slice_name} · Gemini 3 Flash judge · 95% CI",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color="#e8e8e8", lw=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out_path = Path(out) if out else rdir / "lmsys_winrate_gdm_style.png"
    fig.savefig(out_path, dpi=200)
    print(f">>> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
