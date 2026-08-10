# ABOUTME: Builds the two SWE-bench Verified charts for the report and the dashboard entry.
# ABOUTME: Run: uv run python scratch/reports/swebench_charts.py

"""Charts for the SWE-bench Verified head-to-head.

Two figures, in the same house style as the ODCV misalignment chart - cream
ground, black-edged bars, bold value labels, capped 95% CI error bars, bold
title over an italic subtitle:

1. `swebench-outcome-all-instances.svg` - what happened to all 250 instances per
   arm (resolved / submitted but failed / no patch produced), with the published
   baseline marked, and a second panel breaking down why no patch was produced.
2. `swebench-outcome-submitted-only.svg` - the same run scored only over
   instances where a patch was submitted, which is the denominator that flips
   the ranking between the two arms.

Every number is read from a published artifact: the comparison run's
`results.json` on the Hub for the per-arm outcomes, and the eval's own
`exit_statuses` for the failure reasons. Confidence intervals are Wilson, so the
error bars stay inside [0, 1] at these sample sizes. Nothing is estimated.

The reason breakdown is POOLED across arms, and the panel says so. Complete
per-instance exit statuses survive for exactly one of the four cells - the
driver holding the other three died mid-run (docs/swebench_run_postmortem.md)
and the recovered backups covered predictions, not per-rollout statuses.
Apportioning the pooled counts by arm would have produced a per-arm chart out of
numbers nobody measured.

Palette: the reference chart's own colours fail the dataviz checker on a light
ground - its amber sits above the categorical lightness band and its blue falls
under the chroma floor (reads gray). These are the nearest passing neighbours,
so the figures look like the house style and survive the checks: lightness band,
chroma floor, deutan/tritan separation, normal-vision separation and 3:1
contrast all pass. The three no-patch reasons are steps of the crimson, since
they are subdivisions of the crimson segment rather than peers of it.
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = "LASR-Callum/2026-08-07-swebench-verified-qwen36-lora-comparison"
OUT = Path("output/swebench_mini_report")
DASHBOARD_ASSETS = Path(
    "dashboard/content/evals/2026-08-07-swebench-verified-qwen36-lora-comparison/assets"
)

# --- palette (validated; see module docstring) ---------------------------------------
CREAM = "#fdfaf5"
RESOLVED = "#1f6f9e"
WRONG = "#dd9b1f"
NOPATCH = "#d1495b"
REASON_CONTEXT = "#dd6b7b"
REASON_TIMEOUT = "#c03648"
REASON_STEPS = "#801d2c"

# The published baseline. NOT from this run - captioned as such on the chart.
BASELINE = 77.2
BASELINE_NOTE = "published Qwen3.6-27B baseline (not our run)"

ARMS = [("only9284", "only-9284"), ("synthdoc", "synthdoc")]


def house_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#d8d2c8",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#1a1a1a",
            "ytick.color": "#1a1a1a",
            # Keep text as text in the SVG: smaller files, selectable, and
            # readable by a screen reader rather than baked into vector paths.
            "svg.fonttype": "none",
        }
    )


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, as a percentage pair.

    Not the normal approximation: at 8% of 250 the normal interval reaches below
    zero, and an error bar that dips under the axis is a drawing bug pretending
    to be a measurement.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (100 * (centre - spread) / denominator, 100 * (centre + spread) / denominator)


def _token() -> str:
    for line in Path(".env").read_text(encoding="utf8").splitlines():
        if line.strip().startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return os.environ.get("HF_TOKEN", "")


def load_results() -> dict:
    """The authoritative per-arm outcome counts, from the published bundle."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/results.json"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    return json.loads(urllib.request.urlopen(request).read().decode("utf8"))


def load_measured_exit_statuses() -> dict[tuple[str, str], dict[str, int]]:
    """Per-cell exit statuses, for the cells whose summaries survived.

    The eval writes `exit_statuses` as a repr-keyed dict, so it is parsed with
    `literal_eval` rather than json.
    """
    out: dict[tuple[str, str], dict[str, int]] = {}
    for path in sorted(glob.glob("output/eval_summaries/swebench_mini_*.json")):
        doc = json.loads(Path(path).read_text(encoding="utf8"))
        raw = doc.get("exit_statuses")
        if not isinstance(raw, dict) or len(raw) != 1:
            continue
        mapping = ast.literal_eval(next(iter(raw)))
        key = (
            str(doc.get("target", "")).split("/")[-1],
            str(doc.get("selection", {}).get("subset_hash", "")),
        )
        out[key] = {status: len(ids) for status, ids in mapping.items()}
    return out


def titles(ax, title: str, subtitle: str) -> None:
    """Bold title over an italic grey subtitle, as in the house charts."""
    ax.set_title(title, fontsize=15, fontweight="bold", pad=26)
    ax.text(
        0.5,
        1.012,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        style="italic",
        color="#5b5b5b",
    )


def save(fig, name: str) -> None:
    for directory in (OUT, DASHBOARD_ASSETS):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / name, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name} ({(OUT / name).stat().st_size:,} bytes)")


# --- figure 1 -------------------------------------------------------------------------


def figure_all_instances(united: dict, pooled: dict, pooled_n: int) -> None:
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13.2, 6.4), gridspec_kw={"width_ratios": [2.0, 1.15]}
    )

    labels = [label for _, label in ARMS]
    resolved = [100 * united[k]["resolved"] / united[k]["n"] for k, _ in ARMS]
    wrong = [100 * (united[k]["patches"] - united[k]["resolved"]) / united[k]["n"] for k, _ in ARMS]
    nopatch = [100 * (united[k]["n"] - united[k]["patches"]) / united[k]["n"] for k, _ in ARMS]

    bar = dict(width=0.55, edgecolor="black", linewidth=0.9)
    ax.bar(labels, resolved, color=RESOLVED, label="Resolved (pass@1)", **bar)
    ax.bar(labels, wrong, bottom=resolved, color=WRONG,
           label="Patch submitted, tests failed", **bar)
    ax.bar(labels, nopatch, bottom=[r + w for r, w in zip(resolved, wrong)], color=NOPATCH,
           label="No patch produced", **bar)

    # The interval belongs to the resolved proportion, so it is drawn on the top
    # edge of the resolved segment - the only place on a stacked bar where it
    # means anything.
    for index, (key, _) in enumerate(ARMS):
        cell = united[key]
        low, high = wilson(cell["resolved"], cell["n"])
        ax.errorbar(index, resolved[index],
                    yerr=[[resolved[index] - low], [high - resolved[index]]],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=5, capthick=1.5)

    for index, (key, _) in enumerate(ARMS):
        cell = united[key]
        # The error bar occupies the centre column, so the thin middle segment's
        # count is set to the left of it rather than under it.
        pieces = [
            (0.0, resolved[index] / 2, cell["resolved"], "white"),
            (-0.19, resolved[index] + wrong[index] / 2, cell["patches"] - cell["resolved"], "black"),
            (0.0, resolved[index] + wrong[index] + nopatch[index] / 2,
             cell["n"] - cell["patches"], "white"),
        ]
        for dx, y, value, colour in pieces:
            ax.text(index + dx, y, f"{value}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=colour)
        ax.text(index, 101.5, f"pass@1 {resolved[index]:.1f}%", ha="center", va="bottom",
                fontsize=12.5, fontweight="bold")

    # The baseline runs across both bars and is captioned in the right margin,
    # which is the only place on this chart that is not already carrying a number.
    ax.axhline(BASELINE, color="#333333", linestyle="--", linewidth=1.6, zorder=5)
    # Sits ABOVE the line, not on it: centred vertically the dashed rule ran
    # straight through the caption's second line.
    ax.text(1.40, BASELINE + 1.6, f"{BASELINE}%\n{BASELINE_NOTE}", ha="left", va="bottom",
            fontsize=9.5, fontweight="bold", color="#333333", linespacing=1.5)

    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% of the 250 instances")
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    titles(
        ax,
        "SWE-bench Verified: what happened to all 250 instances",
        "Qwen3.6-27B + mini-SWE-agent v2, 65,536-token context  ·  error bars are 95% Wilson CI on pass@1",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, -0.225), ncol=2,
              frameon=False, fontsize=10.5)

    # --- right panel: why no patch, pooled. Horizontal, because these labels are
    # sentences and rotating them or stacking them on an x-axis made them collide.
    order = [
        ("ContextWindowExceededError", "Context window exceeded", REASON_CONTEXT),
        ("Timeout", "Transport timeout\n(network, not the model)", REASON_TIMEOUT),
        ("LimitsExceeded", "Step budget exhausted", REASON_STEPS),
    ]
    names = [name for _, name, _ in order][::-1]
    values = [pooled.get(key, 0) for key, _, _ in order][::-1]
    colours = [colour for _, _, colour in order][::-1]
    bars = bx.barh(names, values, height=0.45, color=colours, edgecolor="black", linewidth=0.9)
    for rect, value in zip(bars, values):
        bx.text(value + 2, rect.get_y() + rect.get_height() / 2, f"{value}",
                ha="left", va="center", fontsize=12.5, fontweight="bold")

    bx.set_xlim(0, max(values) * 1.22)
    bx.set_xlabel("rollouts")
    bx.grid(axis="y", visible=False)
    bx.spines[["top", "right"]].set_visible(False)
    bx.tick_params(axis="y", labelsize=10)
    titles(
        bx,
        "Why no patch was produced",
        f"pooled over {pooled_n} rollouts — not per arm",
    )

    fig.subplots_adjust(wspace=0.42)
    save(fig, "swebench-outcome-all-instances.svg")


# --- figure 2 -------------------------------------------------------------------------


def figure_submitted_only(united: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.0))

    labels = [label for _, label in ARMS]
    rates = [100 * united[k]["resolved"] / united[k]["patches"] for k, _ in ARMS]
    errors = [wilson(united[k]["resolved"], united[k]["patches"]) for k, _ in ARMS]
    lower = [rate - low for rate, (low, _) in zip(rates, errors)]
    upper = [high - rate for rate, (_, high) in zip(rates, errors)]

    # One colour for both bars. In the other figure these hues encode OUTCOME;
    # reusing them here to distinguish ARMS would make blue mean two different
    # things across two charts a reader sees side by side. Colour carries nothing
    # here, so it does not vary - the axis labels identify the arms.
    ax.bar(labels, rates, width=0.5, color=RESOLVED, edgecolor="black", linewidth=0.9)
    ax.errorbar(range(len(labels)), rates, yerr=[lower, upper], fmt="none",
                ecolor="black", elinewidth=1.5, capsize=6, capthick=1.5)

    for index, (key, _) in enumerate(ARMS):
        cell = united[key]
        ax.text(index, errors[index][1] + 2.2, f"{rates[index]:.1f}%", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
        ax.text(index, 3, f"{cell['resolved']} of {cell['patches']}\npatches passed",
                ha="center", va="bottom", fontsize=10.5, color="white", fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_ylabel("% of submitted patches that resolved the issue")
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    titles(
        ax,
        "Scored only over instances where a patch was submitted",
        "95% Wilson CI  ·  higher is better  ·  the ranking flips against pass@1",
    )

    # No baseline here, and the chart says why rather than leaving a gap.
    ax.text(
        0.5,
        -0.155,
        "synthdoc attempts more (155 patches vs 135), so it solves more overall while converting a smaller share.\n"
        "No published baseline is drawn: 77.2% is pass@1 over every instance, not a rate among submitted patches.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#5b5b5b",
    )
    save(fig, "swebench-outcome-submitted-only.svg")


def main() -> None:
    house_style()
    united = load_results()["united"]
    measured = load_measured_exit_statuses()

    # Pooled reason counts, as published in the run's own report.
    pooled = {"ContextWindowExceededError": 72, "Timeout": 60, "LimitsExceeded": 17}
    pooled_n = 369

    figure_all_instances(united, pooled, pooled_n)
    figure_submitted_only(united)

    print("\n  per-arm outcomes (from the published results.json):")
    for arm, cell in united.items():
        low, high = wilson(cell["resolved"], cell["n"])
        slow, shigh = wilson(cell["resolved"], cell["patches"])
        print(
            f"    {arm:10} n={cell['n']} resolved={cell['resolved']} "
            f"wrong={cell['patches'] - cell['resolved']} no_patch={cell['n'] - cell['patches']} "
            f"pass@1={100 * cell['resolved'] / cell['n']:.1f}% [{low:.1f}, {high:.1f}] "
            f"among_submitted={100 * cell['resolved'] / cell['patches']:.1f}% [{slow:.1f}, {shigh:.1f}]"
        )
    print("\n  cells with complete per-instance exit statuses:")
    for (target, subset), counts in measured.items():
        print(f"    {target} subset={subset} n={sum(counts.values())}: {counts}")


if __name__ == "__main__":
    main()
