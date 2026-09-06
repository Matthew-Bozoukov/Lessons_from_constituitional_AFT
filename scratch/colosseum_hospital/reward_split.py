# ABOUTME: Coalition reward vs the rest of the team's reward, per condition and block, from each
# ABOUTME: episode's agent_rewards.json — does a self-promotional pair gain BY hurting the group?

"""Who earns what, by cell.

    uv run python scratch/colosseum_hospital/reward_split.py

Reads the merged cells (output/colosseum_hospital/merged/*), splits every episode's per-agent
rewards into the pair's earner (General_Hospital_Triage; the provisioner earns 0 by
construction), the other three General Hospital departments, and St Mary's four departments,
and reports cell means plus each tempted cell's paired change from its own block's baseline.
Writes the figure and a results.md beside it under output/colosseum_hospital/analysis/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import figure_path

ROOT = Path("output/colosseum_hospital/merged")
OUT = Path("output/colosseum_hospital/analysis")
CONDITIONS = ["baseline", "self_promotional", "self_sacrificial", "covert"]
LABEL = {
    "baseline": "baseline",
    "self_promotional": "self\npromotional",
    "self_sacrificial": "self\nsacrificial",
    "covert": "covert",
}
COLOR = {"control": "#2E6FBF", "treatment": "#C95B2F"}
PAIR_EARNER = "General_Hospital_Triage"
PROVISIONER = "Resource_Provisioner"
SEED = re.compile(r"_seed(\d+)$")
GROUPS = ("pair (GH Triage)", "other GH depts", "St Mary's depts", "joint")


def episodes() -> list[dict]:
    rows = []
    for cell in sorted(ROOT.glob("2026-09-04_colosseum_hospital_*")):
        name = cell.name
        condition = next(c for c in CONDITIONS if f"_{c}_" in name)
        block = "treatment" if "difficult_advice" in name else "control"
        for f in cell.glob("rollouts/colosseum/*/runs/*/*/*/agent_rewards.json"):
            r = json.loads(f.read_text())
            gh_other = sum(
                v
                for k, v in r.items()
                if k.startswith("General_Hospital_") and k != PAIR_EARNER
            )
            stm = sum(v for k, v in r.items() if k.startswith("St_Marys_Center_"))
            rows.append(
                {
                    "condition": condition,
                    "block": block,
                    "seed": int(SEED.search(f.parent.name).group(1)),
                    "pair (GH Triage)": r.get(PAIR_EARNER, 0.0),
                    "other GH depts": gh_other,
                    "St Mary's depts": stm,
                    "provisioner": r.get(PROVISIONER, 0.0),
                    "joint": sum(r.values()),
                    "rest of team": sum(r.values()) - r.get(PAIR_EARNER, 0.0),
                }
            )
    return rows


def ci95(x: np.ndarray, rng: np.random.Generator, n: int = 4000) -> tuple[float, float]:
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    rows = episodes()
    rng = np.random.default_rng(0)
    lines = ["# Colosseum Hospital — who earns what\n"]
    lines.append(
        "Per-agent rewards from agent_rewards.json; the provisioner earns 0 in every episode.\n"
    )
    lines.append(
        "| condition | block | n | pair (GH Triage) | other GH depts | St Mary's depts | rest of team | joint |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["condition"], r["block"]), []).append(r)
    for c in CONDITIONS:
        for b in ("control", "treatment"):
            rs = cells[(c, b)]
            m = lambda k: np.mean([x[k] for x in rs])
            lines.append(
                f"| {c} | {b} | {len(rs)} | {m('pair (GH Triage)'):.0f} | {m('other GH depts'):.0f} | "
                f"{m(chr(83) + 't Mary' + chr(39) + 's depts'):.0f} | {m('rest of team'):.0f} | {m('joint'):.0f} |"
            )
    lines.append(
        "\n## Paired change from the same block's baseline (mean, bootstrap 95% CI over seeds)\n"
    )
    lines.append(
        "| condition | block | pair (GH Triage) | other GH depts | St Mary's depts | rest of team | joint |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for c in CONDITIONS[1:]:
        for b in ("control", "treatment"):
            base = {r["seed"]: r for r in cells[(c if False else "baseline", b)]}
            temp = {r["seed"]: r for r in cells[(c, b)]}
            shared = sorted(set(base) & set(temp))
            out = []
            for k in (
                "pair (GH Triage)",
                "other GH depts",
                "St Mary's depts",
                "rest of team",
                "joint",
            ):
                d = np.array([temp[s][k] - base[s][k] for s in shared])
                lo, hi = ci95(d, rng)
                out.append(f"{d.mean():+.0f} [{lo:+.0f}, {hi:+.0f}]")
            lines.append(f"| {c} | {b} | " + " | ".join(out) + f" | (n={len(shared)})")

    # Figure: coalition reward (left) and rest-of-team reward (right), seed dots + mean ± CI.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, key, title in zip(
        axes,
        ("pair (GH Triage)", "rest of team"),
        (
            "Coalition reward (General Hospital Triage; provisioner earns 0)",
            "Rest of the team (the other seven agents)",
        ),
    ):
        for i, c in enumerate(CONDITIONS):
            for j, b in enumerate(("control", "treatment")):
                vals = np.array([r[key] for r in cells[(c, b)]])
                x = i + (-0.18 if b == "control" else 0.18)
                ax.scatter(
                    x + rng.normal(0, 0.05, len(vals)),
                    vals,
                    s=9,
                    alpha=0.35,
                    color=COLOR[b],
                    linewidths=0,
                )
                lo, hi = ci95(vals, rng)
                ax.errorbar(
                    x,
                    vals.mean(),
                    yerr=[[vals.mean() - lo], [hi - vals.mean()]],
                    fmt="o",
                    color=COLOR[b],
                    ms=6,
                    capsize=0,
                    lw=2,
                )
                ax.annotate(
                    f"{vals.mean():.0f}",
                    (x, vals.mean()),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=COLOR[b],
                )
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([LABEL[c] for c in CONDITIONS])
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("reward (points)")
    axes[0].scatter([], [], color=COLOR["control"], label="control pair")
    axes[0].scatter([], [], color=COLOR["treatment"], label="treatment pair")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Colosseum Hospital — coalition reward vs the rest of the team, by condition (seed dots, mean ± 95% CI)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    png = figure_path(OUT, "colosseum_hospital_reward_split")
    fig.savefig(png, dpi=150)
    md = png.with_name(png.stem + "_results.md")
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigure {png}\nresults {md}")


if __name__ == "__main__":
    main()
