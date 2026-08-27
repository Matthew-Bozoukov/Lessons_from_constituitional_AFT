# ABOUTME: Figure for the confidence autorater: corpus confidence per arm (reasoning / reply), the MOs'
# ABOUTME: first-block confidence at inference, and violation rate by confidence within each arm.
# Run: uv run python scratch/confidence/plot_confidence.py
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from scratch.confidence.common import KEYS
from src.utils import timestamp

matplotlib.use("Agg")
OUT = Path("output/confidence")
ARMS = ["grok", "capped", "sonnet", "gpt"]
LABEL = {
    "grok": "grok-4.6",
    "capped": "Sonnet 5\ncapped",
    "sonnet": "Sonnet 5\n(da716)",
    "gpt": "gpt-5.6",
}
COLOR = {"grok": "#009E73", "capped": "#CC79A7", "sonnet": "#0072B2", "gpt": "#E69F00"}
ROLL = {
    "grok": "grok",
    "sonnet_concise": "capped",
    "sonnet_normal": "sonnet",
    "gpt": "gpt",
}
INK, MUTED, GRID = "#1f2933", "#6b7680", "#e3e8ec"


def rows_of(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def ok(d, ch):
    return (
        "error" not in d
        and isinstance(d.get(ch), dict)
        and all(isinstance(d[ch].get(k), (int, float)) for k in KEYS)
    )


def main() -> None:
    ts = timestamp()
    corp = rows_of(sorted(OUT.glob("corpus_terra_full*.jsonl"))[-1])
    by = defaultdict(dict)
    for d in corp:
        if ok(d, "reasoning") and ok(d, "reply"):
            by[d["corpus"]][d["scenario_id"]] = d
    shared = sorted(set.intersection(*(set(by[a]) for a in ARMS)))
    roll = [
        d
        for d in rows_of(sorted(OUT.glob("rollouts_terra_full*.jsonl"))[-1])
        if ok(d, "reasoning")
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": INK,
            "ytick.color": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12.5,
        }
    )
    fig, axes = plt.subplots(
        1, 4, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.15, 1.15, 1, 1.25]}
    )

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    # panel 1 + 2: corpus overall_confidence and decisiveness, reasoning vs reply, per arm (dots on a 1–7 axis)
    for ax, ch in zip(axes[:2], ("reasoning", "reply")):
        x = range(len(ARMS))
        for j, (k, mk) in enumerate(
            (("overall_confidence", "o"), ("decisiveness", "s"), ("hedging", "^"))
        ):
            ys = [st.mean(by[a][i][ch][k] for i in shared) for a in ARMS]
            ax.plot(x, ys, color=MUTED, lw=1, alpha=0.5, zorder=1)
            for xi, a, y in zip(x, ARMS, ys):
                ax.scatter(
                    xi,
                    y,
                    s=70,
                    marker=mk,
                    color=COLOR[a],
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=3,
                )
                if k == "overall_confidence":
                    ax.text(
                        xi,
                        y + 0.18,
                        f"{y:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color=INK,
                    )
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABEL[a] for a in ARMS])
        ax.set_ylim(1, 7.4)
        ax.set_ylabel("score, 1–7")
        ax.set_title(
            f"Training corpus · {ch}\n(678 shared scenarios, blind judge)", loc="left"
        )
        style(ax)
        if ch == "reasoning":
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [], [], marker=m, color=MUTED, linestyle="", markersize=7, label=lab
                )
                for m, lab in (
                    ("o", "overall confidence"),
                    ("s", "decisiveness"),
                    ("^", "hedging (↑ = more hedged)"),
                )
            ]
            ax.legend(handles=handles, frameon=False, fontsize=9, loc="center left")

    # panel 3: first-block confidence at inference
    ax = axes[2]
    inv = {v: k for k, v in ROLL.items()}
    ys = [
        st.mean(
            d["reasoning"]["overall_confidence"] for d in roll if d["arm"] == inv[a]
        )
        for a in ARMS
    ]
    bars = ax.bar(range(len(ARMS)), ys, color=[COLOR[a] for a in ARMS], width=0.62)
    for b, y in zip(bars, ys):
        ax.text(
            b.get_x() + b.get_width() / 2,
            y + 0.08,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABEL[a] for a in ARMS])
    ax.set_ylim(1, 7.4)
    ax.set_ylabel("overall confidence, 1–7")
    ax.set_title(
        "MO at inference · first reasoning block\n(ODCV, 65 cells)", loc="left"
    )
    style(ax)

    # panel 4: MR by first-block confidence within arm
    ax = axes[3]
    w = 0.38
    for j, (lo, hi, alpha, lab) in enumerate(
        ((4, 5, 0.45, "confidence 4–5"), (6, 7, 1.0, "confidence 6–7"))
    ):
        vals, ns = [], []
        for a in ARMS:
            v = [
                d
                for d in roll
                if d["arm"] == inv[a]
                and lo <= d["reasoning"]["overall_confidence"] <= hi
            ]
            vals.append(100 * st.mean(d["violation"] for d in v) if v else 0)
            ns.append(len(v))
        xs = [i - w / 2 + j * w for i in range(len(ARMS))]
        bars = ax.bar(
            xs,
            vals,
            width=w - 0.04,
            color=[COLOR[a] for a in ARMS],
            alpha=alpha,
            label=lab,
        )
        for b, v, n in zip(bars, vals, ns):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.6,
                f"{v:.0f}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
            )
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([LABEL[a] for a in ARMS])
    ax.set_ylabel("% rollouts scored ≥3")
    ax.set_ylim(0, max(ax.get_ylim()[1], 40))
    ax.set_title(
        "Misalignment by first-block confidence,\nwithin arm (faint = 4–5, solid = 6–7)",
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)

    fig.suptitle(
        "Is confidence the variable? Blind autorater (gpt-5.6-terra) on the four difficult-advice corpora and the MOs' rollouts",
        x=0.01,
        ha="left",
        fontsize=14,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = OUT / "plots" / f"confidence_four_arms_{ts}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    # markdown mirror
    lines = [
        f"# {png.name} — numbers",
        "",
        "| arm | corpus reasoning conf | corpus reply conf | corpus reasoning decisiveness | corpus reply hedging | MO first-block conf | MR conf 4–5 (n) | MR conf 6–7 (n) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        rs = [d for d in roll if d["arm"] == inv[a]]
        c45 = [d for d in rs if 4 <= d["reasoning"]["overall_confidence"] <= 5]
        c67 = [d for d in rs if 6 <= d["reasoning"]["overall_confidence"] <= 7]
        lines.append(
            f"| {a} | {st.mean(by[a][i]['reasoning']['overall_confidence'] for i in shared):.2f} | {st.mean(by[a][i]['reply']['overall_confidence'] for i in shared):.2f} | "
            f"{st.mean(by[a][i]['reasoning']['decisiveness'] for i in shared):.2f} | {st.mean(by[a][i]['reply']['hedging'] for i in shared):.2f} | "
            f"{st.mean(d['reasoning']['overall_confidence'] for d in rs):.2f} | {100 * st.mean(d['violation'] for d in c45) if c45 else float('nan'):.1f} ({len(c45)}) | {100 * st.mean(d['violation'] for d in c67) if c67 else float('nan'):.1f} ({len(c67)}) |"
        )
    lines += [
        "",
        f"Corpus: n={len(shared)} shared scenarios, judge gpt-5.6-terra temp 0, blind to arm. Rollouts: {len(roll)} first reasoning blocks, same judge.",
    ]
    (OUT / "plots" / f"confidence_four_arms_{ts}_results.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(png)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
