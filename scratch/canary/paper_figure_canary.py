# ABOUTME: The canary result in the draft's figure style: the share of each model's deliberative reasoning turns that
# ABOUTME: reuse the trained reasoning (contain the canary word), per eval; --data oct (Jamie's data) or sept.
# Run: uv run python -m scratch.canary.paper_figure_canary [--data oct|sept] [--out-dir output/figures]
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.da_tools.paper_figures import GRID, MUTED, PAPER_RC, save

# autorater.json arm key -> (label, colour); the Sept pair is the 2026-09-29 canary models
DATA = {
    "oct": [
        ("DA (Jamie data)", "Difficult advice", "#7724c4"),
        ("DA + tools (Jamie data)", "Difficult advice + tools", "#9c6500"),
    ],
    "sept": [
        ("DA", "Difficult advice", "#7724c4"),
        ("DA + tools", "Difficult advice + tools", "#9c6500"),
    ],
}
NOTES = {
    "oct": "Oct data (Jamie's 2026-10-05 mix)",
    "sept": "Sept data (2026-09-25 DA corpus)",
}
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
    ap.add_argument("--data", choices=list(DATA), default="oct")
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    out, arms = Path(a.out_dir), DATA[a.data]
    d = json.loads(Path("output/canary/autorater.json").read_text())
    r = {(arm, ev): rates(d[f"{arm}|{ev}"]) for arm, _, _ in arms for ev, _ in EVALS}
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.2, 2.7))
        width = 0.8 / len(arms)
        top = 0
        for j, (arm, lab, col) in enumerate(arms):
            for i, (ev, _) in enumerate(EVALS):
                x = i + (j - (len(arms) - 1) / 2) * width
                v, lo, hi = r[(arm, ev)]["delib"]
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
        ax.set_ylabel("Deliberations reusing the\ntrained reasoning (%)")
        ax.legend(
            frameon=False,
            loc="upper right",
            fontsize=7,
            handlelength=1.0,
            handleheight=0.8,
        )
        fig.text(
            0.01,
            0.01,
            f"{NOTES[a.data]}, one seed per model; Wilson 95% CI over turns.\n"
            "Deliberation rated per turn by Gemini 3 Flash.",
            fontsize=5.5,
            color=MUTED,
        )
        fig.tight_layout(pad=0.4)
        fig.subplots_adjust(bottom=0.2)
        p = save(fig, out, f"paper-fig-canary-{a.data}-data")
    print(
        "| arm | eval | turns | canary, any turn | deliberative turns | canary given deliberation |"
    )
    print("|---|---|---|---|---|---|")
    for arm, lab, _ in arms:
        for ev, _ in EVALS:
            c = d[f"{arm}|{ev}"]
            al, dl = r[(arm, ev)]["all"], r[(arm, ev)]["delib"]
            print(
                f"| {lab} | {ev} | {c['turns']} | {al[0]:.1f}% [{al[1]:.1f}, {al[2]:.1f}] | "
                f"{c['deliberative']} ({c['share_deliberative']:.0f}%) | {dl[0]:.1f}% [{dl[1]:.1f}, {dl[2]:.1f}] |"
            )
    print("wrote", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
