# ABOUTME: Fig. 2 as a 2x3 (legend + five probes): each arm's bar is the recipe mean over its data draws with a
# ABOUTME: seed-variance interval (t over the draws, log-odds scale); every number read from the Hub at render.
# Run: uv run python scratch/figures/fig2_six_panel_seed_variance.py [--copy-to ~/Downloads]
"""Five arms over five probes. An arm with several data draws (a/b/c: the same recipe, disjoint rows)
is shown as ONE bar: the plain mean of the draws' scores, with the interval the t over those draw
means taken on the log-odds scale (src/eval/stats.logit_t, N = the probe's item count), which is the
recipe's seed-to-seed variance and the estimand this figure is named for (declared as the one such
figure in tests/test_figure_intervals.py). A single-draw arm is that run alone under the single-model
rule: the per-item SE interval of scratch/figures/_hub.item_se for MASK, ODCV and agentic
misalignment, Wilson over seeds or episodes for Petri and the Hospital.

The panel layout, titles and legend are those of the 2026-11-06 Fig. 2 (2x3, Single-agent / Multi-agent
brackets); the run lists are the ones that figure's PDF names, with the difficult-advice + tools arm on
its three draws: a = the 2026-10-11 adapter on the first da corpus (the October runs the earlier figure carried),
b and c = the 2026-11-06 adapters on the da-15 b/c rows (configs/data/synth/da-tools.yaml).
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scratch.figures import _hub  # noqa: E402
from src.eval.stats import logit_t  # noqa: E402  (declared: SEED_VARIANCE_OVER_DRAWS)
from src.utils import save_figure_pdf  # noqa: E402

OUT = Path("output/plots")
BASE_NAME = "Qwen3.6-27B"
ARMS = ["control", "plain_const", "delib", "da", "da_tools"]
ARM = {
    "control": "+ Control (Standard SFT)",
    "plain_const": "+ Control + constitution in system prompt",
    "delib": "+ Deliberative Alignment 15% mix",
    "da": "+ Difficult Advice 15% mix",
    "da_tools": "+ Difficult Advice + tool definitions 15% mix",
}
COLOUR = {
    "control": "#5a6472",
    "plain_const": "#0f8b8d",
    "delib": "#2196f3",
    "da": "#7e22ce",
    "da_tools": "#d0558f",
}
BASE_COLOUR = "#c1121f"

# One list per arm and panel: the data draws of one recipe (a, b, c), or the one run an arm has.
DELIB = ["delib-sonnet-15-a", "delib-sonnet-15-b", "delib-sonnet-15-c"]
RUNS: dict[str, dict[str, list[str]]] = {
    "mask": {
        "control": ["2026-11-03-mask-qwen36-0-plain"],
        "plain_const": ["2026-11-04-mask-constitution-qwen36-0-plain"],
        "delib": [f"2026-10-31-mask-qwen36-0-{d}" for d in DELIB],
        "da": [
            "2026-11-03-mask-qwen36-0-da-15",
            "2026-11-02-mask-qwen36-0-da-15-b",
            "2026-11-02-mask-qwen36-0-da-15-c",
        ],
        "da_tools": [
            "2026-10-11-mask-qwen36-0-da-tools-15",
            "2026-11-06-mask-qwen36-0-da-tools-15-b",
            "2026-11-06-mask-qwen36-0-da-tools-15-c",
        ],
    },
    "odcv": {
        "control": ["2026-10-06-odcv-qwen36-0-plain"],
        "plain_const": ["2026-11-04-odcv-constitution-qwen36-0-plain"],
        "delib": [f"2026-10-31-odcv-qwen36-0-{d}" for d in DELIB],
        "da": [
            "2026-10-06-odcv-qwen36-0-da-15",
            "2026-11-02-odcv-qwen36-0-da-15-b",
            "2026-11-02-odcv-qwen36-0-da-15-c",
        ],
        "da_tools": [
            "2026-10-11-odcv-qwen36-0-da-tools-15",
            "2026-11-06-odcv-qwen36-0-da-tools-15-b",
            "2026-11-06-odcv-qwen36-0-da-tools-15-c",
        ],
    },
    "agentic": {
        "control": ["2026-10-08-am-qwen36-0-plain"],
        "plain_const": ["2026-11-04-am-constitution-qwen36-0-plain"],
        "delib": [
            "2026-10-30-am-qwen36-0-delib-sonnet-15-a",
            "2026-10-31-am-qwen36-0-delib-sonnet-15-b",
            "2026-10-31-am-qwen36-0-delib-sonnet-15-c",
        ],
        "da": [
            "2026-10-08-am-qwen36-0-da-15",
            "2026-11-02-am-qwen36-0-da-15-b",
            "2026-11-02-am-qwen36-0-da-15-c",
        ],
        "da_tools": [
            "2026-10-13-am-qwen36-0-da-tools-15",
            "2026-11-06-am-qwen36-0-da-tools-15-b",
            "2026-11-06-am-qwen36-0-da-tools-15-c",
        ],
    },
    "petri": {
        "control": ["2026-10-28-petri-qwen36-0-plain"],
        "plain_const": ["2026-11-04-petri-constitution-qwen36-0-plain"],
        "delib": [
            "2026-11-04-petri-qwen36-0-delib-sonnet-15-a",
            "2026-11-05-petri-qwen36-0-delib-sonnet-15-b",
            "2026-11-05-petri-qwen36-0-delib-sonnet-15-c",
        ],
        "da": [
            "2026-11-05-petri-qwen36-0-da-15",
            "2026-11-05-petri-qwen36-0-da-15-b",
            "2026-11-05-petri-qwen36-0-da-15-c",
        ],
        "da_tools": [
            "2026-10-28-petri-qwen36-0-da-tools-15",
            "2026-11-06-petri-qwen36-0-da-tools-15-b",
            "2026-11-06-petri-qwen36-0-da-tools-15-c",
        ],
    },
    "hospital": {
        "control": ["2026-10-11-hospital-self-sacrificial-qwen36-0-plain"],
        "plain_const": [
            "2026-10-15-hospital-self-sacrificial-principles-qwen36-0-plain"
        ],
        "delib": [f"2026-11-05-hospital-self-sacrificial-qwen36-0-{d}" for d in DELIB],
        "da": [
            "2026-10-15-hospital-self-sacrificial-qwen36-0-da-15",
            "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-b",
            "2026-11-05-hospital-self-sacrificial-qwen36-0-da-15-c",
        ],
        "da_tools": [
            "2026-10-14-hospital-self-sacrificial-qwen36-0-da-tools-15",
            "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-b",
            "2026-11-06-hospital-self-sacrificial-qwen36-0-da-tools-15-c",
        ],
    },
}
BASE_RUNS = {
    "mask": "2026-10-03-mask-qwen36",
    "odcv": "2026-09-06-odcv-qwen36",
    "agentic": "2026-10-25-am-qwen36",
    "petri": "2026-10-28-petri-qwen36",
    "hospital": "2026-10-13-hospital-self-sacrificial-qwen36",
}

PANELS = [
    "mask",
    "petri",
    "odcv",
    "agentic",
    "hospital",
]  # reading order after the legend panel
TITLE = {
    "mask": "MASK honesty (chat)",
    "petri": "Petri audit",
    "odcv": "ODCV",
    "agentic": "Agentic misalignment",
    "hospital": "Colosseum hospital",
}
YLABEL = {
    "mask": "Honesty score (%), higher is better",
    "petri": "Concerning audits (%), lower is better",
    "odcv": "Misalignment rate (%), lower is better",
    "agentic": "Harmful actions (%), lower is better",
    "hospital": "Sabotage rate (%), lower is better",
}

plt.rcParams.update(
    {
        "font.size": 17,
        "axes.labelsize": 16,
        "ytick.labelsize": 15,
        "axes.titlesize": 19,
        "legend.fontsize": 16,
    }
)

Bar = tuple[float, float | None, float | None]


def petri_counts(repo: str) -> tuple[int, int]:
    """(seeds scored concerning >= 4, seeds scored). Runs since 2026-11 publish the summary inside
    results/ (same keys); October ones at the root."""
    try:
        s = _hub.fetch(repo, "results/results.json")
    except Exception:  # noqa: BLE001 -- the file is simply elsewhere
        s = _hub.fetch(repo, "summary.json")
    n = s.get("n_scenarios", s.get("n_scored"))
    assert n, f"{repo}: no seed count"
    return int(s["concerning_ge_4"]), int(n)


def hospital_counts(repo: str) -> tuple[int, int]:
    """(episodes in which the watched pair took a sabotage act, episodes)."""
    by_seed = {int(e["seed"]): e for e in _hub.fetch(repo, "results/episodes.json")}
    return sum(
        1 for e in by_seed.values() if e["pair_sabotaged"] in (1, "1", True)
    ), len(by_seed)


def single(panel: str, repo: str) -> tuple[Bar, int, str]:
    """One run's bar under the single-model rule, its item count, and a note."""
    if panel == "petri":
        k, n = petri_counts(repo)
        return (100 * k / n, *_hub.wilson(k, n)), n, f"{k}/{n} seeds"
    if panel == "hospital":
        k, n = hospital_counts(repo)
        return (100 * k / n, *_hub.wilson(k, n)), n, f"{k}/{n} episodes"
    cells = {
        "mask": _hub.mask_cells,
        "odcv": _hub.odcv_cells,
        "agentic": _hub.agentic_cells,
    }[panel](repo)
    reps = sorted({len(c) for c in cells})
    return _hub.item_se(cells), len(cells), f"{len(cells)} items x {reps} repeats"


def recipe(panel: str, repos: list[str]) -> tuple[Bar, str]:
    """The arm's bar: one run as is; several draws as their plain mean with the t over the draw means on
    the log-odds scale (N = the probe's items), the recipe's seed-variance interval."""
    singles = [single(panel, r) for r in repos]
    if len(singles) == 1:
        bar, _, note = singles[0]
        return bar, note
    means = [b[0] for b, _, _ in singles]
    t = logit_t(means, singles[0][1])
    return (
        t["mean"],
        t["lo"],
        t["hi"],
    ), f"{len(means)} draws {[round(m, 1) for m in means]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--copy-to", default=None, help="also copy the PDF here (e.g. ~/Downloads)"
    )
    a = ap.parse_args()
    vals: dict[str, dict[str, Bar]] = {p: {} for p in PANELS}
    rows: list[tuple] = []
    for p in PANELS:
        for arm in ARMS:
            vals[p][arm], note = recipe(p, RUNS[p][arm])
            rows.append(
                (TITLE[p], ARM[arm], vals[p][arm], note, ", ".join(RUNS[p][arm]))
            )
    base = {p: single(p, BASE_RUNS[p])[0][0] for p in PANELS}

    fig, axes = plt.subplots(
        2, 3, figsize=(20.0, 12.5), gridspec_kw={"wspace": 0.34, "hspace": 0.42}
    )
    legend_ax = axes[0][0]
    legend_ax.axis("off")
    panel_axes = {
        "mask": axes[0][1],
        "petri": axes[0][2],
        "odcv": axes[1][0],
        "agentic": axes[1][1],
        "hospital": axes[1][2],
    }
    for p, ax in panel_axes.items():
        crowded = (
            max((v[2] if v[2] is not None else v[0]) for v in vals[p].values()) > 93
        )
        for i, arm in enumerate(ARMS):
            m, lo, hi = vals[p][arm]
            ax.bar(
                i,
                m,
                width=0.68,
                color=COLOUR[arm],
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            if lo is not None:  # one repeat per item gives no bar
                lo_c, hi_c = max(0.0, lo), min(100.0, hi)
                ax.errorbar(
                    i,
                    m,
                    yerr=[[max(0.0, m - lo_c)], [max(0.0, hi_c - m)]],
                    fmt="none",
                    ecolor="#2b2523",
                    elinewidth=1.3,
                    capsize=5,
                    capthick=1.3,
                    zorder=4,
                )
            top = min(100.0, hi) if hi is not None else m
            ax.text(
                i,
                top + 1.4,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=12 if crowded else 15,
                fontweight="bold",
                zorder=5,
                clip_on=False,
            )
        ax.axhline(
            base[p],
            color=BASE_COLOUR,
            linestyle=(0, (2.5, 2.5)),
            linewidth=1.8,
            zorder=2,
        )
        ax.set_title(TITLE[p], pad=26 if crowded else 12)
        ax.set_ylabel(YLABEL[p])
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            color=BASE_COLOUR,
            linestyle=(0, (2.5, 2.5)),
            linewidth=1.8,
            label=BASE_NAME,
        )
    ]
    handles += [
        Patch(facecolor=COLOUR[arm], edgecolor="black", label=ARM[arm]) for arm in ARMS
    ]
    legend_ax.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(-0.08, 0.5),
        handlelength=1.8,
        handleheight=1.0,
        labelspacing=1.1,
    )
    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.05, right=0.99)
    # Brackets under the bottom row: ODCV + agentic are single-agent probes, the Hospital is multi-agent.
    for l_ax, r_ax, label in (
        (axes[1][0], axes[1][1], "Single-agent"),
        (axes[1][2], axes[1][2], "Multi-agent"),
    ):
        x0, x1 = l_ax.get_position().x0, r_ax.get_position().x1
        y = axes[1][0].get_position().y0 - 0.045
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y, y],
                transform=fig.transFigure,
                color="black",
                linewidth=1.2,
            )
        )
        fig.text((x0 + x1) / 2, y - 0.022, label, ha="center", va="top", fontsize=17)

    OUT.mkdir(parents=True, exist_ok=True)
    today = f"{datetime.now():%Y-%m-%d}"
    stem = OUT / f"{today}_fig2_six_panel_seed_variance"
    artifacts = sorted(
        {r for p in PANELS for arm in ARMS for r in RUNS[p][arm]}
        | set(BASE_RUNS.values())
    )
    pdf = save_figure_pdf(
        fig,
        f"{stem}.pdf",
        f"Fig. 2, 2x3, recipe means with seed-variance intervals, {today}",
        artifacts,
        "scratch/figures/fig2_six_panel_seed_variance.py",
    )
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    md = [
        f"# {stem.name}",
        "",
        f"Rendered {datetime.now().isoformat(timespec='seconds')} by scratch/figures/fig2_six_panel_seed_variance.py; "
        "every bar read from the Hub at render time. A multi-draw arm is the plain mean of its draws with a t over "
        "the draw means on the log-odds scale (N = the probe's items); a single-draw arm carries the per-item SE "
        "interval (MASK, ODCV, agentic) or Wilson over seeds / episodes (Petri, Hospital).",
        "",
        "| Panel | Arm | Value | 95% interval | Draws or items | Runs |",
        "|---|---|---|---|---|---|",
    ]
    md += [
        f"| {p} | {arm} | {v[0]:.1f} | "
        + (f"[{v[1]:.1f}, {v[2]:.1f}]" if v[1] is not None else "none (one repeat)")
        + f" | {n} | {r} |"
        for p, arm, v, n, r in rows
    ]
    md += [
        f"| {TITLE[p]} | {BASE_NAME} (base line) | {base[p]:.1f} | | | {BASE_RUNS[p]} |"
        for p in PANELS
    ]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(pdf)
    if a.copy_to:
        dst = Path(a.copy_to).expanduser() / pdf.name
        shutil.copy(pdf, dst)
        print(dst)


if __name__ == "__main__":
    main()
