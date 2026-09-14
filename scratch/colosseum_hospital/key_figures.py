# ABOUTME: Supervisor-facing figures for the 2026-09-14 Hospital batch, one per experiment (1, 2, 3+4, 5) plus
# ABOUTME: the agentic-task trade-off; each states one message in its title and draws its paired p-values on.
"""Reads the batch summary (batch_analysis.py summary) and writes, beside it:

  2026-09-14_colosseum_hospital_refusal_not_restraint.png
  2026-09-14_colosseum_hospital_difficult_advice_gap_by_harness.png
  2026-09-14_colosseum_hospital_agentic_task_tradeoff.png
  2026-09-14_colosseum_hospital_untempted_baseline.png
  2026-09-14_colosseum_hospital_mixed_coalition.png
  2026-09-14_colosseum_hospital_key_figures_results.md   (every number drawn, with its source)

Run: uv run python scratch/colosseum_hospital/key_figures.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.utils import figure_path  # noqa: E402

DATE = "2026-09-14"
AN = Path("output/colosseum_hospital/analysis")
SUMMARY = AN / f"{DATE}_colosseum_hospital_batch_summary.json"
S = json.loads(SUMMARY.read_text())
CELLS, CONTRASTS = S["cells"], S["contrasts"]

COLOR = {
    "ctrl": "#2a78d6",
    "da": "#eb6834",
    "nosyn": "#8a8984",
    "jda": "#c99a2e",
    "jdat": "#3f9a5b",
}
NAME = {
    "ctrl": "control",
    "da": "difficult advice",
    "nosyn": "no synthetic",
    "jda": "+7% difficult\nadvice",
    "jdat": "+7% difficult\nagentic tasks",
}
INK, MUTED, GRID, PALE, BAND = "#0b0b0b", "#52514e", "#e3e2dd", "#cfcec8", "#f4f1ea"
PLAN_C, ACT_C, REFUSE_C = PALE, "#45443f", "#eb6834"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)

MD: list[str] = [
    f"# Key figures, Hospital batch {DATE}\n",
    f"Every number below is read from `{SUMMARY.name}` (batch_analysis.py summary); script "
    "`scratch/colosseum_hospital/key_figures.py`. Shares are k/30 shifts with Wilson 95% intervals; team scores "
    "are means with bootstrap 95% intervals; contrasts are paired by seed (exact McNemar for shares, sign-flip "
    "permutation for means) with bootstrap 95% intervals.\n",
]


def rate(cell: str, m: str) -> tuple[float, float, float, int, int]:
    b = CELLS[cell]["binary"][m]
    r = b["rate"]
    return (
        100 * r,
        max(0.0, 100 * (r - b["lo"])),
        max(0.0, 100 * (b["hi"] - r)),
        b["k"],
        b["n"],
    )


def mean(cell: str, m: str) -> tuple[float, float, float]:
    c = CELLS[cell]["counts"][m]
    return c["mean"], max(0.0, c["mean"] - c["lo"]), max(0.0, c["hi"] - c["mean"])


def contrast(label: str, m: str) -> dict:
    norm = lambda s: s.replace("−", "-")  # noqa: E731
    hit = [c for c in CONTRASTS if norm(c["label"]) == norm(label)]
    assert len(hit) == 1, f"contrast {label!r}: {len(hit)} matches"
    return hit[0]["measures"][m]


def ptxt(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def signed(x: float, unit: str = "") -> str:
    return f"{x:+.0f}{unit}".replace("-", "−")


def pp(c: dict) -> str:
    return f"{signed(100 * c['diff'], ' pp')} [{signed(100 * c['lo'])}, {signed(100 * c['hi'])}], {ptxt(c['p'])}"


def heading(fig, title: str, sub: str, sub_y: float) -> None:
    fig.text(
        0.012,
        0.975,
        title,
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.25,
    )
    fig.text(0.012, sub_y, sub, fontsize=10.5, color=MUTED, va="top", linespacing=1.35)


def bracket(ax, x1: float, x2: float, y: float, text: str, h: float) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=1.1, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + 1.35 * h,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        color=INK,
    )


def fig_refusal_not_restraint() -> Path:
    arms = ["ctrl", "da", "nosyn", "jda", "jdat"]
    xs = [0.0, 1.0, 2.7, 3.7, 4.7]
    series = [
        ("plan_posted", "posted a sabotage plan on the coalition board", PLAN_C),
        ("pair_sabotaged", "took a sabotage action", ACT_C),
        (
            "judge_refusal_reasoning_ge3",
            "a seat refused in its private reasoning (judge)",
            REFUSE_C,
        ),
    ]
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.77, bottom=0.33)
    MD.append("## refusal_not_restraint.png (fixed harness, self-sacrificial)\n")
    MD.append("| arm | " + " | ".join(lab for _, lab, _ in series) + " |")
    MD.append("|---|---|---|---|")
    for a, x in zip(arms, xs):
        row = []
        for j, (m, _, col) in enumerate(series):
            v, lo, hi, k, n = rate(f"fixed/{a}", m)
            xx = x + (j - 1) * w
            ax.bar(xx, v, w * 0.92, color=col, edgecolor="none")
            ax.errorbar(
                xx,
                v,
                yerr=[[lo], [hi]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.5,
            )
            ax.text(
                xx,
                v + hi + 1.2,
                f"{k}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MUTED,
            )
            row.append(f"{k}/{n} ({v:.0f}%)")
        MD.append(f"| {NAME[a].replace(chr(10), ' ')} | " + " | ".join(row) + " |")
    ax.set_xticks(xs, [NAME[a] for a in arms], fontsize=11, color=INK)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("share of 30 shifts, %")
    ax.set_xlim(-0.55, 5.25)

    ours = (
        contrast("ours: DA − control", "pair_sabotaged"),
        contrast("ours: DA − control", "judge_refusal_reasoning_ge3"),
    )
    adv = (
        contrast("unfiltered: 7% advice − no synthetic", "pair_sabotaged"),
        contrast("unfiltered: 7% advice − no synthetic", "judge_refusal_reasoning_ge3"),
    )
    agt = (
        contrast("unfiltered: 7% agentic − no synthetic", "pair_sabotaged"),
        contrast(
            "unfiltered: 7% agentic − no synthetic", "judge_refusal_reasoning_ge3"
        ),
    )

    def short(c: dict) -> str:
        return f"{signed(100 * c['diff'], ' pp')} ({ptxt(c['p'])})"

    fam = [
        (
            xs[0],
            xs[1],
            "OUR ADAPTERS (Table 2 only base)",
            [f"DA − control:  action {short(ours[0])},  refusal {short(ours[1])}"],
        ),
        (
            xs[2],
            xs[4],
            "UNFILTERED TABLE 2 BLEND (seed 0)",
            [
                f"+7% advice − none:  action {short(adv[0])},  refusal {short(adv[1])}",
                f"+7% agentic tasks − none:  action {short(agt[0])},  refusal {short(agt[1])}",
            ],
        ),
    ]
    tr = ax.get_xaxis_transform()
    for x1, x2, lab, lines in fam:
        ax.plot(
            [x1 - 0.45, x2 + 0.45],
            [-0.15, -0.15],
            transform=tr,
            color=MUTED,
            lw=0.8,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            -0.19,
            lab,
            transform=tr,
            ha="center",
            va="top",
            fontsize=9.5,
            color=MUTED,
            fontweight="bold",
        )
        ax.text(
            (x1 + x2) / 2,
            -0.26,
            "\n".join(lines),
            transform=tr,
            ha="center",
            va="top",
            fontsize=9.5,
            color=INK,
            linespacing=1.5,
        )
    ax.legend(
        handles=[Patch(color=c, label=lab) for _, lab, c in series],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.13),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
    )
    heading(
        fig,
        "Every arm writes the sabotage plan; the trained arms say no in private,\nbut mostly sabotage anyway",
        "Experiment 1 · five arms, coalition told to sabotage; fixed harness (carried history, no re-ask, no order to plan). "
        "30 shifts per arm, Wilson 95% CI;\nthe number above a bar is shifts out of 30. Contrasts paired by seed, "
        "exact McNemar. One training seed per arm; compare within a family (the base blends differ).",
        0.885,
    )
    MD.append("")
    for x1, x2, lab, lines in fam:
        MD.append(f"- {lab}: " + "; ".join(lines))
    MD.append("")
    out = figure_path(AN, "colosseum_hospital_refusal_not_restraint")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def fig_gap_by_harness() -> Path:
    states = [
        ("carried history", "carried history\nplan ordered, re-ask on"),
        ("+ no re-ask", "+ no re-ask\nplan ordered"),
        ("+ plan optional", "+ plan optional\nre-ask on"),
        ("+ both", "+ both (fixed harness)\nno plan order, no re-ask"),
    ]
    series = [
        (
            "judge_refusal_reasoning_ge3",
            "a seat refused in its private reasoning (judge)",
            REFUSE_C,
            "o",
            -0.08,
        ),
        ("pair_sabotaged", "took a sabotage action", ACT_C, "s", 0.08),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.25)
    ax.axvspan(-0.5, 1.5, color=BAND, zorder=0)
    ax.text(
        0.5,
        101,
        "a sabotage plan is ordered",
        ha="center",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )
    ax.text(
        2.5, 101, "no order to plan", ha="center", va="bottom", fontsize=10, color=MUTED
    )
    ax.axhline(0, color=INK, lw=1.0)
    MD.append(
        "## difficult_advice_gap_by_harness.png (difficult advice − control, paired by seed)\n"
    )
    MD.append("| harness | " + " | ".join(lab for _, lab, *_ in series) + " |")
    MD.append("|---|---|---|")
    rows = {s: [] for s, _ in states}
    for m, lab, col, mk, dx in series:
        ys = []
        for i, (key, _) in enumerate(states):
            c = contrast(f"DA − control under {key}", m)
            d, lo, hi = 100 * c["diff"], 100 * c["lo"], 100 * c["hi"]
            ys.append(d)
            ax.errorbar(
                i + dx,
                d,
                yerr=[[max(0.0, d - lo)], [max(0.0, hi - d)]],
                fmt=mk,
                color=col,
                ms=8,
                capsize=3.5,
                lw=1.5,
                zorder=3,
            )
            ax.text(
                i + dx + 0.1,
                d,
                f"{signed(d, ' pp')}\n{ptxt(c['p'])}",
                va="center",
                fontsize=9.5,
                color=col,
                zorder=4,
            )
            rows[key].append(pp(c))
    for key, _ in states:
        MD.append(f"| {key} | " + " | ".join(rows[key]) + " |")
    MD.append("")
    ax.set_xticks(range(4), [lab for _, lab in states], fontsize=10.5, color=INK)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-62, 110)
    ax.set_yticks(
        range(-60, 101, 20), [signed(t) if t else "0" for t in range(-60, 101, 20)]
    )
    ax.set_ylabel("difficult advice − control, percentage points")
    ax.grid(False)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=col, marker=mk, lw=1.3, ms=7, label=lab)
            for _, lab, col, mk, _ in series
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    heading(
        fig,
        "Difficult advice vs control: the refusal gap holds on every harness;\nthe action gap shows only while a plan is ordered",
        "Experiment 5 · our two adapters, coalition told to sabotage; carried history, then each switch alone, then both. "
        "Difference in share of 30 shifts,\npaired by seed, bootstrap 95% CI, exact McNemar p.",
        0.875,
    )
    out = figure_path(AN, "colosseum_hospital_difficult_advice_gap_by_harness")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def fig_agentic_tradeoff() -> Path:
    arms = ["nosyn", "jda", "jdat"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.4))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.74, bottom=0.2, wspace=0.26)
    MD.append("## agentic_task_tradeoff.png (unfiltered Table 2 blend, seed 0)\n")
    MD.append(
        "| arm | took a sabotage action (told to sabotage) | refused in reasoning (judge) | team score, not told to sabotage |"
    )
    MD.append("|---|---|---|---|")
    ticks = []
    for i, a in enumerate(arms):
        v, lo, hi, k, n = rate(f"fixed/{a}", "pair_sabotaged")
        _, _, _, rk, rn = rate(f"fixed/{a}", "judge_refusal_reasoning_ge3")
        a1.bar(i, v, 0.62, color=COLOR[a], edgecolor="none")
        a1.errorbar(
            i, v, yerr=[[lo], [hi]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3.5
        )
        a1.text(
            i,
            v / 2,
            f"{k}/{n}",
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            fontweight="bold",
        )
        ticks.append(f"{NAME[a]}\nrefused: {rk}/{rn}")
        m, mlo, mhi = mean(f"baseline/{a}", "joint_reward")
        a2.bar(i, m, 0.62, color=COLOR[a], edgecolor="none")
        a2.errorbar(
            i,
            m,
            yerr=[[mlo], [mhi]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=3.5,
        )
        a2.text(
            i,
            m / 2,
            f"{m:,.0f}",
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            fontweight="bold",
        )
        MD.append(
            f"| {NAME[a].replace(chr(10), ' ')} | {k}/{n} ({v:.0f}%) | {rk}/{rn} | {m:,.0f} [{m - mlo:,.0f}, {m + mhi:,.0f}] |"
        )
    act_adv = contrast("unfiltered: 7% advice − no synthetic", "pair_sabotaged")
    act_agt = contrast("unfiltered: 7% agentic − no synthetic", "pair_sabotaged")
    cap_adv = contrast(
        "baseline: unfiltered · 7% advice − unfiltered · no synthetic", "joint_reward"
    )
    cap_agt = contrast(
        "baseline: unfiltered · 7% agentic tasks − unfiltered · no synthetic",
        "joint_reward",
    )
    bracket(
        a1,
        0,
        1,
        104,
        f"{signed(100 * act_adv['diff'], ' pp')}, {ptxt(act_adv['p'])}",
        2.5,
    )
    bracket(
        a1,
        0,
        2,
        117,
        f"{signed(100 * act_agt['diff'], ' pp')}, {ptxt(act_agt['p'])}",
        2.5,
    )
    bracket(a2, 0, 1, 7900, f"{signed(cap_adv['diff'])}, {ptxt(cap_adv['p'])}", 190)
    bracket(a2, 0, 2, 8750, f"{signed(cap_agt['diff'])}, {ptxt(cap_agt['p'])}", 190)
    a1.set_ylim(0, 132)
    a1.set_yticks(range(0, 101, 20))
    a1.set_ylabel("share of 30 shifts, %")
    a1.set_title(
        "Told to sabotage: the pair took a sabotage action",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=12,
    )
    a2.set_ylim(0, 9900)
    a2.set_yticks(range(0, 8001, 2000), [f"{t:,}" for t in range(0, 8001, 2000)])
    a2.set_ylabel("team score (higher is better)")
    a2.set_title(
        "Not told to sabotage: how well the team runs the hospital",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=12,
    )
    a1.set_xticks(range(3), ticks, fontsize=10.5, color=INK)
    a2.set_xticks(range(3), [NAME[a] for a in arms], fontsize=10.5, color=INK)
    heading(
        fig,
        "Adding 7% difficult agentic tasks is the one data change that cut sabotage,\nat a cost in ordinary performance",
        "Unfiltered Table 2 blend of 10,000 rows, seed 0, one training seed per arm; the same 30 seeds in both "
        "conditions, fixed harness.\nLeft: Wilson 95% CI, exact McNemar p. Right: bootstrap 95% CI, sign-flip p. "
        "A second training seed is needed before the drop counts as a result.",
        0.885,
    )
    MD.append("")
    MD.append(
        f"- action, +7% advice − none: {pp(act_adv)}; +7% agentic tasks − none: {pp(act_agt)}"
    )
    MD.append(
        f"- team score, +7% advice − none: {signed(cap_adv['diff'])} [{signed(cap_adv['lo'])}, {signed(cap_adv['hi'])}], "
        f"{ptxt(cap_adv['p'])}; +7% agentic tasks − none: {signed(cap_agt['diff'])} "
        f"[{signed(cap_agt['lo'])}, {signed(cap_agt['hi'])}], {ptxt(cap_agt['p'])}"
    )
    MD.append("")
    out = figure_path(AN, "colosseum_hospital_agentic_task_tradeoff")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def seat_refusal(cell: str, seat: str) -> dict[int, bool]:
    """Per seed: did this coalition seat's own reasoning carry a refusal (judge >= 3)?"""
    d = json.loads(((Path("output/colosseum_hospital/merged") / CELLS[cell]["dir"]) / "results" / "per_seed.json").read_text())
    vals = d[f"judge_refusal_reasoning_{seat}"]["self_sacrificial"]
    return {int(s): v >= 3 for s, v in vals.items() if v is not None}


def mcnemar(a: dict[int, bool], b: dict[int, bool]) -> float:
    s = sorted(set(a) & set(b))
    up = sum(a[i] and not b[i] for i in s)
    down = sum(b[i] and not a[i] for i in s)
    n = up + down
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(up, down) + 1)) / 2**n)


def fig_untempted_baseline() -> Path:
    arms = ["ctrl", "da", "nosyn", "jda", "jdat"]
    short = {
        "ctrl": "control",
        "da": "difficult\nadvice",
        "nosyn": "no\nsynthetic",
        "jda": "+7%\nadvice",
        "jdat": "+7%\nagentic",
    }
    xs = [0.0, 1.0, 2.5, 3.5, 4.5]
    fig, (b1, b2) = plt.subplots(
        1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.27, wspace=0.18)
    MD.append(
        "## untempted_baseline.png (experiment 2: no instruction to sabotage, fixed harness)\n"
    )
    MD.append(
        "| arm | team score | sabotage rule fired anyway (false alarm) | resource failures a shift |"
    )
    MD.append("|---|---|---|---|")
    for a, x in zip(arms, xs):
        m, lo, hi = mean(f"baseline/{a}", "joint_reward")
        b1.bar(x, m, 0.7, color=COLOR[a], edgecolor="none")
        b1.errorbar(
            x, m, yerr=[[lo], [hi]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=3.5
        )
        b1.text(
            x,
            m / 2,
            f"{m:,.0f}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )
        v, vlo, vhi, k, n = rate(f"baseline/{a}", "pair_sabotaged")
        b2.bar(x, v, 0.7, color=COLOR[a], edgecolor="none")
        b2.errorbar(
            x,
            v,
            yerr=[[vlo], [vhi]],
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=3.5,
        )
        b2.text(
            x,
            v + vhi + 1.5,
            f"{k}/{n}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=MUTED,
        )
        f = mean(f"baseline/{a}", "resource_failures")[0]
        MD.append(
            f"| {NAME[a].replace(chr(10), ' ')} | {m:,.0f} [{m - lo:,.0f}, {m + hi:,.0f}] | {k}/{n} ({v:.0f}%) | {f:.1f} |"
        )
    pairs = [
        (
            0,
            1,
            "baseline: ours · difficult advice − ours · control",
            "difficult advice − control",
            8000,
        ),
        (
            2,
            3,
            "baseline: unfiltered · 7% advice − unfiltered · no synthetic",
            "+7% advice − none",
            8000,
        ),
        (
            2,
            4,
            "baseline: unfiltered · 7% agentic tasks − unfiltered · no synthetic",
            "+7% agentic − none",
            8900,
        ),
    ]
    alarm_txt = []
    MD.append("")
    for i, j, lab, short_lab, y in pairs:
        c = contrast(lab, "joint_reward")
        bracket(b1, xs[i], xs[j], y, f"{signed(c['diff'])}, {ptxt(c['p'])}", 200)
        s = contrast(lab, "pair_sabotaged")
        alarm_txt.append(
            f"{short_lab} {signed(100 * s['diff'], ' pp')} ({ptxt(s['p'])})"
        )
        MD.append(
            f"- {short_lab}: team score {signed(c['diff'])} [{signed(c['lo'])}, {signed(c['hi'])}], {ptxt(c['p'])}; "
            f"false alarms {pp(s)}"
        )
    MD.append("")
    b2.text(
        0.5,
        -0.25,
        "paired by seed:  " + alarm_txt[0] + "\n" + ";  ".join(alarm_txt[1:]),
        transform=b2.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color=INK,
        linespacing=1.5,
    )
    b1.set_xticks(xs, [NAME[a] for a in arms], fontsize=10, color=INK)
    b2.set_xticks(xs, [short[a] for a in arms], fontsize=10, color=INK)
    for ax in (b1, b2):
        ax.set_xlim(-0.6, 5.1)
    b1.set_ylim(0, 10300)
    b1.set_yticks(range(0, 8001, 2000), [f"{t:,}" for t in range(0, 8001, 2000)])
    b1.set_ylabel("team score (higher is better)")
    b1.set_title(
        "How well the team runs the hospital",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=12,
    )
    b2.set_ylim(0, 75)
    b2.set_yticks(range(0, 61, 20))
    b2.set_ylabel("share of 30 shifts, %")
    b2.set_title(
        "Shifts the sabotage rule flagged anyway (false alarms)",
        loc="left",
        fontsize=11.5,
        color=INK,
        pad=12,
    )
    heading(
        fig,
        "With no instruction to sabotage, every arm still runs the hospital;\nthe 7% agentic-task adapter runs it about 10% worse",
        "Experiment 2 · untempted baseline: cooperative prompt, no coalition, fixed harness, the same 30 seeds as experiment 1. "
        "Left: mean team score, bootstrap\n95% CI, sign-flip p. Right: shifts the sabotage rule flagged with no instruction to "
        "sabotage (an idle provisioner while a hospital is short), Wilson 95% CI.",
        0.885,
    )
    out = figure_path(AN, "colosseum_hospital_untempted_baseline")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def fig_mixed_coalition() -> Path:
    panels = [
        (
            "provisioner",
            "mixed/daprov",
            "Experiment 3 · our DA adapter as provisioner, control as Triage",
            "DA provisioner\n+ control Triage",
            "both DA",
            "DA provisioner + control Triage − both DA",
        ),
        (
            "triage",
            "mixed/datri",
            "Experiment 4 · control as provisioner, our DA adapter as Triage",
            "control provisioner\n+ DA Triage",
            "both control",
            "control provisioner + DA Triage − both control",
        ),
    ]
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.27, wspace=0.07)
    MD.append(
        "## mixed_coalition.png (experiments 3 and 4: one DA seat, one control seat, fixed harness)\n"
    )
    MD.append(
        "| experiment | pair | the pair took a sabotage action | the seat under test refused in its own reasoning (judge) |"
    )
    MD.append("|---|---|---|---|")
    notes = []
    for ax, (seat, mixed, ptitle, mlabel, reflab, clabel) in zip(axes, panels):
        comps = [
            ("fixed/ctrl", "both control"),
            (mixed, mlabel),
            ("fixed/da", "both DA"),
        ]
        ax.axvspan(0.5, 1.5, color=BAND, zorder=0)
        refusals = {}
        exp = ptitle.split(" · ")[0]
        for i, (cell, lab) in enumerate(comps):
            v, lo, hi, k, n = rate(cell, "pair_sabotaged")
            ax.bar(i - w / 2, v, w * 0.92, color=ACT_C, edgecolor="none")
            ax.errorbar(
                i - w / 2,
                v,
                yerr=[[lo], [hi]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.5,
            )
            ax.text(
                i - w / 2,
                v + hi + 1.2,
                f"{k}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=MUTED,
            )
            r = refusals[cell] = seat_refusal(cell, seat)
            rk, rn = sum(r.values()), len(r)
            p, plo, phi = wilson(rk, rn)
            ax.bar(i + w / 2, 100 * p, w * 0.92, color=REFUSE_C, edgecolor="none")
            ax.errorbar(
                i + w / 2,
                100 * p,
                yerr=[[100 * (p - plo)], [100 * (phi - p)]],
                fmt="none",
                ecolor=INK,
                elinewidth=0.9,
                capsize=2.5,
            )
            ax.text(
                i + w / 2,
                100 * phi + 1.2,
                f"{rk}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=MUTED,
            )
            MD.append(
                f"| {exp} | {lab.replace(chr(10), ' ')} | {k}/{n} | {seat}: {rk}/{rn} |"
            )
        pm = mcnemar(refusals[mixed], refusals["fixed/da"])
        km, kd = sum(refusals[mixed].values()), sum(refusals["fixed/da"].values())
        bracket(ax, 1 + w / 2, 2 + w / 2, 103, f"{km} vs {kd} of 30, {ptxt(pm)}", 2.5)
        c = contrast(clabel, "pair_sabotaged")
        ax.text(
            0.5,
            -0.25,
            f"the pair's sabotage actions, mixed − {reflab}: {signed(100 * c['diff'], ' pp')} ({ptxt(c['p'])})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            color=INK,
        )
        notes.append(
            f"- {exp}: the {seat} seat refused {km} of 30 beside a control partner vs {kd} of 30 beside a DA partner, "
            f"{ptxt(pm)}; the pair's sabotage actions, mixed − {reflab}: {pp(c)}"
        )
        ax.set_xticks(range(3), [lab for _, lab in comps], fontsize=10.5, color=INK)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(ptitle, loc="left", fontsize=11.5, color=INK, pad=12)
    MD.extend(["", *notes, ""])
    axes[0].set_ylim(0, 118)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].set_ylabel("share of 30 shifts, %")
    fig.legend(
        handles=[
            Patch(color=ACT_C, label="the pair took a sabotage action"),
            Patch(
                color=REFUSE_C,
                label="the seat under test refused in its own reasoning (judge): the provisioner in 3, Triage in 4",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    heading(
        fig,
        "A control partner does not talk a difficult-advice seat out of refusing",
        "Experiments 3 and 4 · our DA adapter in one coalition seat and our control adapter in the other, both orders, "
        "beside the all-control and all-DA\npairs on the same 30 seeds; fixed harness. Wilson 95% CI; the number above a "
        "bar is shifts out of 30; exact McNemar p, paired by seed.",
        0.925,
    )
    out = figure_path(AN, "colosseum_hospital_mixed_coalition")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def main() -> None:
    outs = [
        fig_refusal_not_restraint(),
        fig_untempted_baseline(),
        fig_mixed_coalition(),
        fig_gap_by_harness(),
        fig_agentic_tradeoff(),
    ]
    md = figure_path(AN, "colosseum_hospital_key_figures_results", ext="md")
    md.write_text("\n".join(MD) + "\n")
    for p in [*outs, md]:
        print(p)


if __name__ == "__main__":
    main()
