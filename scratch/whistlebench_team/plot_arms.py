# ABOUTME: Compare whistlebench_team arms (nosynth control vs difficult advice): ladder rungs per team,
# ABOUTME: hit vs false-alarm rates, collusion with the teammate; writes a dated PNG + results.md.
# Run: uv run python scratch/whistlebench_team/plot_arms.py --nosynth <run dir> --da <run dir> [--out output/whistlebench_team]
"""One figure, three panels, both arms.

Left: where each episode landed on the outcome ladder (misconduct documents), one stacked bar
per arm and team. Middle: the report rate where the documents show misconduct (hits) against
the report rate where they do not (false alarms), per team and arm, with 95% Wilson intervals.
Right: how often the target did what the concealing teammate asked (misconduct documents).
Arm colours are the repo's: nosynth grey, difficult advice purple (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.misalignment.whistlebench_team.measures import RUNGS, rate  # noqa: E402

ARM_COLOURS = {"nosynth": "#8a8f98", "da": "#6f42c1"}
ARM_LABELS = {"nosynth": "nosynth control", "da": "difficult advice (da-15)"}
TEAM_LABELS = {
    "solo": "alone",
    "ai_neutral": "AI teammate\n(no agenda)",
    "ai_conceal": "AI teammate\nasks to conceal",
    "human_conceal": "human colleague\nasks to conceal",
}
# one hue, light -> dark up the ladder; the two overreach rungs in the warm tone
RUNG_COLOURS = dict(
    zip(
        RUNGS,
        ["#e6e8eb", "#c9d8e8", "#9fbbd9", "#6f9bc7", "#3f78b0", "#d9822b", "#a3471a"],
    )
)


def load(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "results" / "episodes.jsonl").read_text().splitlines()
    ]


def rung_shares(rows: list[dict], team: str) -> dict[str, float]:
    sub = [
        r
        for r in rows
        if r["team"] == team and r["docs"] == "misconduct" and r.get("rung")
    ]
    n = len(sub) or 1
    return {rung: sum(1 for r in sub if r["rung"] == rung) / n for rung in RUNGS}, len(
        sub
    )


def plot(arms: dict[str, list[dict]], out_png: Path) -> list[str]:
    teams = [
        t
        for t in ("solo", "ai_neutral", "ai_conceal", "human_conceal")
        if any(r["team"] == t for rows in arms.values() for r in rows)
    ]
    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.6, 1.3, 0.9]}
    )
    fig.patch.set_facecolor("white")
    lines: list[str] = []

    # ── left: ladder rungs, misconduct documents ──
    ypos, ylabels = [], []
    y = 0
    for team in teams:
        for arm, rows in arms.items():
            shares, n = rung_shares(rows, team)
            left = 0.0
            for rung in RUNGS:
                w = shares[rung]
                if w > 0:
                    ax_l.barh(
                        y,
                        w,
                        left=left,
                        color=RUNG_COLOURS[rung],
                        edgecolor="white",
                        linewidth=1.5,
                        height=0.72,
                    )
                    if w >= 0.12:
                        ax_l.text(
                            left + w / 2,
                            y,
                            rung,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#1f2933" if rung in RUNGS[:3] else "white",
                        )
                    left += w
            ypos.append(y)
            ylabels.append(
                f"{TEAM_LABELS[team].replace(chr(10), ' ')}\n{ARM_LABELS[arm]}  (n={n})"
            )
            lines.append(
                f"| {team} | {arm} | {n} | "
                + " | ".join(f"{shares[r]:.0%}" for r in RUNGS)
                + " |"
            )
            y += 1
        y += 0.6
    ax_l.set_yticks(ypos)
    ax_l.set_yticklabels(ylabels, fontsize=8)
    ax_l.invert_yaxis()
    ax_l.set_xlim(0, 1)
    ax_l.set_xlabel("share of episodes (misconduct documents)")
    ax_l.set_title("Where each episode landed on the ladder", fontsize=10, loc="left")
    for side in ("top", "right"):
        ax_l.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RUNG_COLOURS[r]) for r in RUNGS]
    ax_l.legend(
        handles,
        RUNGS,
        fontsize=7,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
        frameon=False,
    )

    # ── middle: hits vs false alarms (report_any) ──
    x = np.arange(len(teams))
    width = 0.18
    for i, (arm, rows) in enumerate(arms.items()):
        for j, docs in enumerate(("misconduct", "benign")):
            vals = []
            for team in teams:
                sub = [
                    r.get("report_any")
                    for r in rows
                    if r["team"] == team and r["docs"] == docs
                ]
                vals.append(rate(sub))
            xs = x + (i - 0.5) * 2 * width + (j - 0.5) * width
            heights = [v["rate"] or 0.0 for v in vals]
            err = [
                [max(0.0, (v["rate"] or 0) - (v["lo"] or 0)) for v in vals],
                [max(0.0, (v["hi"] or 0) - (v["rate"] or 0)) for v in vals],
            ]
            ax_m.bar(
                xs,
                heights,
                width=width * 0.92,
                color=ARM_COLOURS[arm],
                alpha=1.0 if docs == "misconduct" else 0.45,
                edgecolor="white",
                linewidth=1,
                label=f"{ARM_LABELS[arm]} — {'hit (misconduct)' if docs == 'misconduct' else 'false alarm (benign)'}",
            )
            ax_m.errorbar(
                xs,
                heights,
                yerr=err,
                fmt="none",
                ecolor="#1f2933",
                elinewidth=1,
                capsize=2,
            )
            for team, v in zip(teams, vals):
                lines.append(
                    f"| {team} | {arm} | {docs} | report_any | {v['k']}/{v['n']} = {v['rate']:.0%} [{v['lo']:.0%}, {v['hi']:.0%}] |"
                    if v["n"]
                    else f"| {team} | {arm} | {docs} | report_any | - |"
                )
    ax_m.set_xticks(x)
    ax_m.set_xticklabels([TEAM_LABELS[t] for t in teams], fontsize=8)
    ax_m.set_ylim(0, 1)
    ax_m.set_ylabel(
        "share of episodes with a report\n(emailed someone inside or outside)"
    )
    ax_m.set_title(
        "Reports: when there was misconduct vs when there was not",
        fontsize=10,
        loc="left",
    )
    ax_m.legend(fontsize=7, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax_m.spines[side].set_visible(False)

    # ── right: went along with the concealing teammate ──
    conceal = [t for t in teams if t.endswith("conceal")]
    x = np.arange(len(conceal))
    for i, (arm, rows) in enumerate(arms.items()):
        vals = [
            rate(
                [
                    r.get("complied_with_teammate")
                    for r in rows
                    if r["team"] == t and r["docs"] == "misconduct"
                ]
            )
            for t in conceal
        ]
        xs = x + (i - 0.5) * 0.36
        heights = [v["rate"] or 0.0 for v in vals]
        err = [
            [max(0.0, (v["rate"] or 0) - (v["lo"] or 0)) for v in vals],
            [max(0.0, (v["hi"] or 0) - (v["rate"] or 0)) for v in vals],
        ]
        ax_r.bar(
            xs,
            heights,
            width=0.33,
            color=ARM_COLOURS[arm],
            edgecolor="white",
            linewidth=1,
            label=ARM_LABELS[arm],
        )
        ax_r.errorbar(
            xs, heights, yerr=err, fmt="none", ecolor="#1f2933", elinewidth=1, capsize=2
        )
        for t, v in zip(conceal, vals):
            lines.append(
                f"| {t} | {arm} | misconduct | complied_with_teammate | {v['k']}/{v['n']} = {v['rate']:.0%} [{v['lo']:.0%}, {v['hi']:.0%}] |"
                if v["n"]
                else f"| {t} | {arm} | misconduct | complied_with_teammate | - |"
            )
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([TEAM_LABELS[t] for t in conceal], fontsize=8)
    ax_r.set_ylim(0, 1)
    ax_r.set_ylabel("share that did as the teammate asked")
    ax_r.set_title("Went along with the cover-up request", fontsize=10, loc="left")
    ax_r.legend(fontsize=7, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax_r.spines[side].set_visible(False)

    fig.suptitle(
        "WhistleBench-Team on Qwen3.6-27B: nosynth control vs difficult advice",
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosynth", required=True, type=Path)
    ap.add_argument("--da", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("output/whistlebench_team"))
    args = ap.parse_args()
    arms = {"nosynth": load(args.nosynth), "da": load(args.da)}
    stem = f"{date.today().isoformat()}_wbt_nosynth_vs_da15"
    out_png = args.out / f"{stem}.png"
    lines = plot(arms, out_png)
    md = [
        f"# WhistleBench-Team: nosynth vs difficult advice ({date.today().isoformat()})",
        "",
        f"nosynth run: `{args.nosynth}`  ",
        f"da run: `{args.da}`",
        "",
        "## Ladder shares (misconduct documents)",
        "",
        "| team | arm | n | " + " | ".join(RUNGS) + " |",
        "|---|---|---|" + "---|" * len(RUNGS),
    ]
    md += [line for line in lines if line.count("|") == len(RUNGS) + 4]
    md += [
        "",
        "## Rates (95% Wilson)",
        "",
        "| team | arm | docs | measure | value |",
        "|---|---|---|---|---|",
    ]
    md += [line for line in lines if line.count("|") == 6]
    (args.out / f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_png)
    print("\n".join(md))


if __name__ == "__main__":
    main()
