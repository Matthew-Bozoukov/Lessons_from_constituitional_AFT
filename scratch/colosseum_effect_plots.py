# ABOUTME: One figure per experiment showing what putting constitutionally-SFT'd agents on the
# ABOUTME: team does — the cause (missed tool calls), the mechanism (violations), the outcome (regret).

"""How much does having the SFT'd model on the team change things?

Three panels per experiment, left to right along the causal chain the transcripts established:

  A  agents that never emitted assign_task   — the thing the arms actually differ on
  B  violations per episode                  — what the environment's random fill turns that into
  C  normalised regret                       — the outcome metric it surfaces as

Plotting all three together is the point. The outcome panel on its own reads as "the SFT'd model
coordinates worse", which is the reading the transcripts refuted: condition on panel A and the arms
are indistinguishable. Panel C is a consequence of panel A, not an independent finding, and a figure
that shows only C invites exactly the wrong conclusion.

Counts come from the episodes themselves rather than from any summary: an agent is a NON-ACTOR in an
episode if it never appears as the agent_name of an `assign_task` event in tool_events.json.

    uv run python scratch/colosseum_effect_plots.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

ROOT = Path("output/colosseum_jira")
CONTROL, CONTROL_EDGE = "#2F6DB5", "#1B4472"
TREAT, TREAT_EDGE = "#D97706", "#7C4406"
INK, MUTED = "#333333", "#8A8A8A"

# experiment -> cells in causal order (baseline first)
CELLS = {
    "collusion": ["baseline", "collusion"],
    "single": ["baseline", "single"],
    "cooperation": ["cooperation"],
}
N_AGENTS = 6


def episode_stats(run_dir: Path) -> list[dict]:
    """One record per episode: cell, seed, non-actors, violations, normalised regret."""
    out = []
    for ep in sorted(run_dir.glob("rollouts/colosseum/*/runs/*/*/*")):
        if not ep.is_dir():
            continue
        try:
            metrics = json.loads((ep / "metrics.json").read_text())
            cfg = json.loads((ep / "run_config.json").read_text())
            final = json.loads((ep / "final_summary.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        # non-actors: agents with no assign_task event of their own
        actors = set()
        try:
            for e in json.loads((ep / "tool_events.json").read_text()):
                if e.get("tool_name") == "assign_task" and e.get("agent_name"):
                    actors.add(e["agent_name"])
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        opt = None
        try:
            opt = (
                json.loads((ep / "optimal_summary.json").read_text())
                .get("optimal", {})
                .get("joint_reward")
            )
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        achieved = final.get("joint_reward")
        regret = (
            min(max(1 - achieved / opt, 0.0), 1.0)
            if opt not in (None, 0) and achieved is not None
            else None
        )
        out.append(
            {
                "cell": ep.parent.name,
                "seed": int(cfg["seed"]),
                "non_actors": N_AGENTS - len(actors),
                "violations": metrics.get("violations") or 0,
                "regret": regret,
            }
        )
    return out


def mean_ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    h = 1.96 * float(v.std(ddof=1)) / np.sqrt(v.size)
    return m, m - h, m + h


def collect() -> dict:
    """{experiment: {arm: {cell: {measure: [values]}}}} over every finished run dir."""
    data: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for d in sorted(ROOT.glob("*/")):
        meta = d / "metadata" / "run_meta.json"
        if not meta.is_file():
            continue
        target = json.loads(meta.read_text())["target"]
        arm = "treatment" if "difficult" in target else "control"
        for rec in episode_stats(d):
            for exp, cells in CELLS.items():
                if rec["cell"] in cells:
                    for k in ("non_actors", "violations", "regret"):
                        data[exp][arm][rec["cell"]][k].append(rec[k])
    return data


PANELS = [
    ("non_actors", "A · agents that never called assign_task", "per episode, of 6"),
    ("violations", "B · violations", "duplicate claims per episode"),
    ("regret", "C · normalised regret", "0 = optimal allocation"),
]


def draw(exp: str, arms: dict) -> Path:
    cells = [c for c in CELLS[exp] if c in arms.get("control", {})]
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor="white")
    n_treated = {"collusion": 2, "single": 1, "cooperation": 6}[exp]
    fig.suptitle(
        f"{exp} — {n_treated} of 6 seats hold the difficult-advice model",
        fontsize=20,
        y=1.02,
    )

    for ax, (key, title, unit) in zip(axes, PANELS):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E4E4E4", linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        xs = np.arange(len(cells))
        for arm, colour, edge, off in [
            ("control", CONTROL, CONTROL_EDGE, -0.055),
            ("treatment", TREAT, TREAT_EDGE, 0.055),
        ]:
            stats = [mean_ci(arms[arm][c][key]) for c in cells]
            ms = [s[0] for s in stats]
            lo = [s[0] - s[1] for s in stats]
            hi = [s[2] - s[0] for s in stats]
            if len(cells) > 1:
                ax.plot(xs + off, ms, color=colour, linewidth=2.5, zorder=2)
            ax.errorbar(
                xs + off,
                ms,
                yerr=[lo, hi],
                fmt="o",
                markersize=11,
                color=colour,
                markeredgecolor=edge,
                markeredgewidth=2,
                ecolor=edge,
                elinewidth=2.2,
                capsize=6,
                capthick=2.2,
                zorder=3,
                linestyle="none",
            )
        # the gap, annotated on the last cell
        c_last = mean_ci(arms["control"][cells[-1]][key])[0]
        t_last = mean_ci(arms["treatment"][cells[-1]][key])[0]
        ax.annotate(
            f"{t_last - c_last:+.3f}".rstrip("0").rstrip("."),
            xy=(len(cells) - 1 + 0.055, t_last),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=TREAT_EDGE,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(cells)
        ax.set_xlim(-0.55, len(cells) - 0.15)
        ax.set_ylim(bottom=0)
        ax.set_title(title, pad=12, loc="left")
        ax.set_ylabel(unit)

    axes[0].legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=10,
                color=CONTROL,
                markeredgecolor=CONTROL_EDGE,
                markeredgewidth=2,
                label="Tulu-only control",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=10,
                color=TREAT,
                markeredgecolor=TREAT_EDGE,
                markeredgewidth=2,
                label="7% principle-only",
            ),
        ],
        loc="upper left",
        frameon=False,
    )

    n_ep = sum(len(arms[a][c]["violations"]) for a in arms for c in cells)
    fig.text(
        0.5,
        -0.04,
        f"{n_ep} episodes. Bars are 95% CI over episodes. Panel C is a CONSEQUENCE of panel A: "
        "conditioned on the number of non-actors, the two arms are indistinguishable.",
        ha="center",
        fontsize=12.5,
        color=MUTED,
    )

    path = figure_path("output/colosseum", f"colosseum_jira_{exp}_effect_of_sft_arm")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    data = collect()
    for exp in ("cooperation", "collusion", "single"):
        arms = data.get(exp)
        if not arms or "control" not in arms or "treatment" not in arms:
            print(f"!!! {exp}: missing an arm, skipped")
            continue
        cells = [c for c in CELLS[exp] if c in arms["control"]]
        for c in cells:
            for arm in ("control", "treatment"):
                v = arms[arm][c]
                print(
                    f"{exp:12s} {arm:9s} {c:12s} n={len(v['violations']):4d}  "
                    f"non_actors={mean_ci(v['non_actors'])[0]:.3f}  "
                    f"violations={mean_ci(v['violations'])[0]:.3f}  "
                    f"regret={mean_ci(v['regret'])[0]:.3f}"
                )
        print(f"  -> {draw(exp, arms)}\n")


if __name__ == "__main__":
    main()
