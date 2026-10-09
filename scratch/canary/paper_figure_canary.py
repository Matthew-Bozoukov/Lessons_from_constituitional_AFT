# ABOUTME: The canary result in the draft's figure style: the share of each model's deliberative reasoning turns that
# ABOUTME: reuse the trained reasoning (contain the canary word), per eval, difficult advice vs difficult advice + tools.
# Run: uv run python -m scratch.canary.paper_figure_canary [--out-dir output/figures]
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.da_tools.paper_figures import GRID, MUTED, PAPER_RC, save

ARMS = [
    ("DA (Jamie data)", "Difficult advice", "#7724c4"),
    ("DA + tools (Jamie data)", "Difficult advice + tools", "#9c6500"),
]
EVALS = [
    ("MASK", "MASK\n(chat)"),
    ("ODCV", "ODCV\n(single-agent)"),
    ("Hospital", "Colosseum hospital\n(multi-agent)"),
]


def wilson(k: int, n: int) -> tuple[float, float]:
    z, p = 1.96, k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def rates(c: dict) -> dict:
    """(rate %, lo, hi) over every reasoning turn and over deliberative turns only."""
    hits = c["canary_given_deliberation"] + c["canary_without_deliberation"]
    return {
        "all": (100 * hits / c["turns"], *wilson(hits, c["turns"])),
        "delib": (c["p"], *wilson(c["canary_given_deliberation"], c["deliberative"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output/figures")
    out = Path(ap.parse_args().out_dir)
    d = json.loads(Path("output/canary/autorater.json").read_text())
    r = {(arm, ev): rates(d[f"{arm}|{ev}"]) for arm, _, _ in ARMS for ev, _ in EVALS}
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.2, 2.7))
        axes = [ax]
        for ax, (key, title) in zip(axes, [("delib", "")]):
            width = 0.8 / len(ARMS)
            top = 0
            for j, (arm, lab, col) in enumerate(ARMS):
                for i, (ev, _) in enumerate(EVALS):
                    x = i + (j - (len(ARMS) - 1) / 2) * width
                    v, lo, hi = r[(arm, ev)][key]
                    ax.bar(
                        x,
                        v,
                        width * 0.94,
                        color=col,
                        zorder=3,
                        label=lab if i == 0 else None,
                    )
                    ax.errorbar(
                        x,
                        v,
                        yerr=[[v - lo], [hi - v]],
                        fmt="none",
                        ecolor="#333",
                        elinewidth=0.7,
                        capsize=1.8,
                        capthick=0.7,
                        zorder=4,
                    )
                    ax.text(
                        x,
                        hi + 0.8,
                        f"{v:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#111",
                    )
                    top = max(top, hi)
            ax.set_xticks(range(len(EVALS)), [lab for _, lab in EVALS], fontsize=7)
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, top * 1.22)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Deliberations reusing the\ntrained reasoning (%)")
        axes[0].legend(
            frameon=False,
            loc="upper right",
            fontsize=7,
            handlelength=1.0,
            handleheight=0.8,
        )
        fig.text(
            0.01,
            0.01,
            "Oct data (Jamie's 2026-10-05 mix), one seed per model; Wilson 95% CI over turns.\n"
            "Deliberation rated per turn by Gemini 3 Flash.",
            fontsize=5.5,
            color=MUTED,
        )
        fig.tight_layout(pad=0.4)
        fig.subplots_adjust(bottom=0.2)
        p = save(fig, out, "paper-fig-canary-oct-data")
    print(
        "| arm | eval | turns | canary, any turn | deliberative turns | canary given deliberation |"
    )
    print("|---|---|---|---|---|---|")
    for arm, lab, _ in ARMS:
        for ev, _ in EVALS:
            c = d[f"{arm}|{ev}"]
            a, dl = r[(arm, ev)]["all"], r[(arm, ev)]["delib"]
            print(
                f"| {lab} | {ev} | {c['turns']} | {a[0]:.1f}% [{a[1]:.1f}, {a[2]:.1f}] | "
                f"{c['deliberative']} ({c['share_deliberative']:.0f}%) | {dl[0]:.1f}% [{dl[1]:.1f}, {dl[2]:.1f}] |"
            )
    print("wrote", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
