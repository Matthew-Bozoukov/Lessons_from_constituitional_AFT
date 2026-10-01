# ABOUTME: One figure, three panels in the paper draft's Figure 6 layout: ODCV misalignment, Hospital sabotage and
# ABOUTME: MASK honesty for no synthetic / difficult advice / difficult advice + tools / difficult agentic tasks.
# Run: uv run python scratch/da_tools/combined_figure.py [--out-dir output/figures]
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_figure import honesty  # noqa: E402
from paper_figures import PAPER_RC, hospital, odcv_rate  # noqa: E402

from src.naming import figure_path  # noqa: E402

# (legend label, fixed colour, ODCV run, Hospital run, MASK run); None = no run under the protocol.
ARMS = [
    ("No synthetic data", "#5f6774", "2026-09-22-odcv-qwen36-0-nosynth", None,
     "2026-09-22-mask-qwen36-0-nosynth"),
    ("Difficult advice (15%)", "#7724c4", "2026-09-26-odcv-qwen36-0-da-15",
     "2026-09-28-hospital-self-sacrificial-qwen36-0-da-15", "2026-09-26-mask-qwen36-0-da-15"),
    ("Difficult advice + tools (15%)", "#9c6500", "2026-09-28-odcv-qwen36-0-da-tools-15",
     "2026-09-28-hospital-self-sacrificial-qwen36-0-da-tools-15", "2026-09-28-mask-qwen36-0-da-tools-15"),
    ("Difficult agentic tasks (7%)", "#eda100", "2026-09-09-odcv-qwen36-0-dat-7", None,
     "2026-09-09-mask-qwen36-0-dat-7"),
]
PANELS = [
    ("ODCV\n(lower is better)", "Misalignment rate (%)"),
    ("Colosseum hospital, self-sacrificial\n(lower is better)", "Shifts with a sabotage act (%)"),
    ("MASK honesty\n(higher is better)", "Honesty score (%)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    values = []  # per arm: [(pct, se_pct) | None] for the three panels
    for _, _, odcv, hosp, mask in ARMS:
        p, se, _ = odcv_rate(odcv)
        o = (100 * p, 100 * se)
        if hosp:
            h = hospital(hosp)
            hv = (100 * h["rate"], 100 * h["se"])
        else:
            hv = None
        m = honesty(mask)
        values.append([o, hv, (m["score"], m["se"])])
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.75))
        for k, (ax, (title, ylabel)) in enumerate(zip(axes, PANELS)):
            for i, ((_, colour, *_), vals) in enumerate(zip(ARMS, values)):
                v = vals[k]
                if v is None:
                    ax.text(i, 2, "not run", ha="center", va="bottom", fontsize=6, color="#6b7680", rotation=90)
                    continue
                ax.bar(i, v[0], 0.72, color=colour, zorder=3)
                eb = ax.errorbar(i, v[0], yerr=v[1], fmt="none", ecolor="#333", elinewidth=0.7, capsize=1.8,
                                 capthick=0.7, zorder=4)
                for art in (*eb[1], *eb[2]):
                    art.set_path_effects([withStroke(linewidth=1.35, foreground="white")])
                ax.text(i, v[0] + v[1] + 1.5, f"{v[0]:.1f}", ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold", color="#111")
            ax.set_title(title, fontsize=7.5, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.set_xticks([])
            ax.set_xlim(-0.6, len(ARMS) - 0.4)
            ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.legend(handles=[Patch(color=c, label=lab) for lab, c, *_ in ARMS], frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), ncol=len(ARMS), handlelength=1.0, handleheight=0.8, columnspacing=1.2,
                   fontsize=7)
        fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
        p = figure_path(Path(a.out_dir), "da_tools_odcv_hospital_mask", ext="pdf")
        fig.savefig(p)
        fig.savefig(p.with_suffix(".png"), dpi=300)
        plt.close(fig)
    md = [f"# {p.stem}", "",
          "Three evals, one seed per arm; +-1 binomial SE (ODCV n=240 rollouts, Hospital n=30 shifts, MASK n=1,000 rows).",
          "Hospital: no-simulation framing, self-sacrificial cell, nosynth peers; no-synthetic and DAT not run under it.",
          "DAT trains on ODCV-style agentic tasks, so its ODCV number is in-distribution; its MASK run used a 12,288-token "
          "generation cap (the others 16,384).", "",
          "| arm | ODCV misaligned % | Hospital sabotaged % | MASK honest % |", "|---|---|---|---|"]
    for (lab, *_), vals in zip(ARMS, values):
        md.append(f"| {lab} | " + " | ".join(f"{v[0]:.1f} +- {v[1]:.1f}" if v else "not run" for v in vals) + " |")
    md += ["", "Runs: " + ", ".join(f"`dougalldeepmind/{r}`" for _, _, *rs in ARMS for r in rs if r)]
    (p.parent / f"{p.stem}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", p, "and", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
