# ABOUTME: The one figure for the length-matched arm: reasoning and reply word counts for the
# ABOUTME: three paired arms (da716, capped Sonnet, grok) as p10-p90 ranges with medians.

"""Run: uv run python scratch/sonnet_concise/plot_lengths.py --run_dir output/synthdoc_sonnet_concise_716/<ts>
       [--grok_local <grok dataset.jsonl>] [--out_dir output/sonnet_concise]

Form: a range plot, not a bar chart -- the question is where each arm's distribution sits, and
a bar would hide the spread that the cap does NOT match. Each arm is one horizontal range:
thin line p10-p90, thick line p25-p75, dot at the median, direct-labelled. Two panels,
reasoning and reply, on one shared axis so the two caps can be compared by eye.

Writes a timestamped PNG plus a markdown mirror of every number drawn.
"""

import sys
from datetime import datetime
from pathlib import Path

import fire
import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.build_da716_prompt_source import BASE_REPO  # noqa: E402
from scratch.sonnet_concise.measure_lengths import (  # noqa: E402
    CAP_REASONING,
    CAP_REPLY,
    GROK_REPO,
    _hf,
    _jsonl,
    _q,
    _turns,
)

matplotlib.use("Agg")

# Categorical hues in fixed order (validated with the dataviz palette script, light surface).
ARMS = [
    ("A · da716 — Sonnet, unconstrained", "da716", "#4F53B8"),
    ("C · Sonnet, capped 220/270", "cap", "#0D9A80"),
    ("B · grok-4.6, unconstrained", "grok", "#A8690F"),
]
INK, MUTED, GRID = "#1C2027", "#5D6572", "#E3E6EB"


def main(
    run_dir: str, grok_local: str = "", out_dir: str = "output/sonnet_concise"
) -> None:
    load_dotenv()
    rd = Path(run_dir)
    export = sorted(rd.glob("stage_*_export_sft.jsonl"))
    arm = {r["metadata"]["scenario_id"]: r for r in _jsonl(export[-1])}
    grok = {
        r["metadata"]["scenario_id"]: r
        for r in (
            _jsonl(Path(grok_local))
            if grok_local
            else _hf(
                GROK_REPO,
                (
                    "dataset.jsonl",
                    "stage_5_export_sft.jsonl",
                    "stages/stage_5_export_sft.jsonl",
                ),
            )
        )
    }
    da = {
        r["metadata"]["scenario_id"]: r
        for r in _hf(
            BASE_REPO, ("stage_8_export_sft.jsonl", "stages/stage_8_export_sft.jsonl")
        )
    }
    ids = [i for i in arm if i in da and i in grok]
    corpora = {"da716": da, "cap": arm, "grok": grok}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    fig.patch.set_facecolor("white")
    md = [
        f"# Lengths, three paired arms ({len(ids)} shared scenarios), words",
        "",
        "| panel | arm | p10 | p25 | median | p75 | p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for ax, (idx, title, cap) in zip(
        axes, ((0, "Reasoning", CAP_REASONING), (1, "Reply", CAP_REPLY))
    ):
        ax.set_facecolor("white")
        for y, (label, key, color) in enumerate(reversed(ARMS)):
            q = _q([_turns(corpora[key][i])[idx] for i in ids])
            ax.plot(
                [q["p10"], q["p90"]],
                [y, y],
                color=color,
                lw=1.2,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                [q["p25"], q["p75"]],
                [y, y],
                color=color,
                lw=5,
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(q["p50"], y, "o", color="white", mec=color, mew=2, ms=9, zorder=4)
            ax.annotate(
                f"{q['p50']}",
                (q["p50"], y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK,
                fontweight="bold",
            )
            if ax is axes[0]:
              ax.text(
                  -0.03, y, label,
                  ha="right",
                  va="center",
                  fontsize=9,
                  color=INK,
                  transform=ax.get_yaxis_transform(),
              )
            md.append(
                f"| {title} | {label} | {q['p10']} | {q['p25']} | **{q['p50']}** | {q['p75']} | {q['p90']} |"
            )
        ax.axvline(cap, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(cap + 6, 2.62, f"cap {cap}", ha="left", va="bottom", fontsize=8, color=MUTED
        )
        ax.set_title(f"{title} words", loc="left", fontsize=11, color=INK, pad=14)
        ax.set_ylim(-0.6, 2.9)
        ax.set_yticks([])
        ax.set_xlim(0, 640)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    fig.suptitle(
        "Same 703 questions: line p10–p90, band p25–p75, dot = median",
        x=0.01,
        ha="left",
        fontsize=9,
        color=MUTED,
        y=0.99,
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.80, bottom=0.14, wspace=0.18)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    png = od / f"lengths_three_arms_{ts}.png"
    fig.savefig(png, dpi=200)
    (od / f"lengths_three_arms_{ts}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"-> {png}")


if __name__ == "__main__":
    fire.Fire(main)
