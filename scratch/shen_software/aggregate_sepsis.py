# ABOUTME: Spread across replicate runs of the Shen sepsis task: per arm x setup, every run's business/ethics,
# ABOUTME: mean, SD and range, Shen-scored vs recomputed counts; writes a results table and a row chart.
"""Run: uv run python scratch/shen_software/aggregate_sepsis.py <runs dir> <shen test.json> <prefix> [<prefix> ...]

Prefixes name the replicate runs to pool, e.g. `r2_ r3_ r4_` (the patched harness). Round 1 (unprefixed
tags, original harness) is a different condition: pass `r1_` ALONE to summarise it, never pooled with the
rest. Scoring and recompute rules are fig_sepsis.py's (Shen's fixed bounds; ‡ = Shen's scorer could not
read the output and the decisions were recomputed from the agents' own file).
"""

import importlib.util
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR, TEST_JSON, PREFIXES = sys.argv[1], sys.argv[2], sys.argv[3:]
if not PREFIXES:
    raise SystemExit("give at least one replicate prefix, e.g. r2_ r3_ r4_")
if "r1_" in PREFIXES and len(PREFIXES) > 1:
    raise SystemExit(
        "round 1 ran on the original harness; summarise it alone (prefix r1_), never pooled"
    )

sys.argv = [
    sys.argv[0],
    RUNS_DIR,
    TEST_JSON,
    "r2_",
]  # fig_sepsis needs argv; its own figure is discarded
spec = importlib.util.spec_from_file_location(
    "fs", Path(__file__).with_name("fig_sepsis.py")
)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
plt.close("all")

from src.utils import save_figure_pdf

SETUPS = [
    ("single_sepsis", "Single agent", "o"),
    ("org8_sepsis", "Org (PM + 8 coders)", "s"),
]
tag_prefix = {p: ("" if p == "r1_" else p) for p in PREFIXES}

cells = {}
for s, _, _ in SETUPS:
    for a, _ in fs.ARMS:
        runs = []
        for p in PREFIXES:
            r = fs.load(f"{tag_prefix[p]}{s}_{a}")
            if r:
                r["business"], r["ethics"] = (
                    fs.business(r["cost"]),
                    1 - r["missed_rate"],
                )
            runs.append((p, r))
        cells[(s, a)] = runs


def summary(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f} (n=1)"
    return f"{st.mean(vals):.2f} ± {st.stdev(vals):.2f} [{min(vals):.2f}-{max(vals):.2f}] (n={len(vals)})"


lines = [
    f"# Shen sepsis spread across runs {' '.join(PREFIXES)} (2026-11-17)",
    "",
    "Mean ± SD [min-max] over the runs that produced decisions; ‡ = recomputed (Shen's scorer could not read "
    "the output). Unscored runs are listed, not imputed.",
    "",
    "| Setup | Arm | Business | Ethics | Treated per run | Missed per run | Shen-scored / ‡ / unscored |",
    "|---|---|---|---|---|---|---|",
]
for s, slab, _ in SETUPS:
    for a, alab in fs.ARMS:
        runs = cells[(s, a)]
        got = [r for _, r in runs if r]
        rc = sum(1 for r in got if r["recomputed"])
        lines.append(
            f"| {slab} | {alab} | {summary([r['business'] for r in got])} | {summary([r['ethics'] for r in got])} | "
            + ", ".join(
                f"{p}{r['treated']}{'‡' if r['recomputed'] else ''}" if r else f"{p}–"
                for p, r in runs
            )
            + " | "
            + ", ".join(f"{p}{r['missed']}" if r else f"{p}–" for p, r in runs)
            + f" | {len(got) - rc} / {rc} / {len(runs) - len(got)} |"
        )
lines += ["", f"Runs dir: {RUNS_DIR} (local). Adapters: " + ", ".join(fs.HUB.values())]
stem = f"2026-11-17_shen_sepsis_spread_{'_'.join(p.rstrip('_') for p in PREFIXES)}"
(fs.OUT / f"{stem}_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Row chart: one row per arm; single agent above the row line, org below; every run a small marker, the mean
# a large one, a line across the run range.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
ypos = {a: i for i, (a, _) in enumerate(reversed(fs.ARMS))}
for ax, key, title in [
    (axes[0], "business", "Business score  (higher = cheaper)"),
    (axes[1], "ethics", "Ethics score  (higher = fewer sepsis cases missed)"),
]:
    for a, _ in fs.ARMS:
        for (s, _, mk), dy in zip(SETUPS, (0.17, -0.17)):
            got = [r for _, r in cells[(s, a)] if r]
            if not got:
                continue
            y, vals = ypos[a] + dy, [r[key] for r in got]
            ax.plot(
                [min(vals), max(vals)],
                [y, y],
                color=fs.COLOUR[a],
                lw=2,
                alpha=0.4,
                zorder=1,
            )
            ax.scatter(
                vals,
                [y] * len(vals),
                marker=mk,
                s=45,
                facecolor="white" if s == "single_sepsis" else fs.COLOUR[a],
                edgecolor=fs.COLOUR[a],
                linewidth=1.5,
                alpha=0.8,
                zorder=2,
            )
            m = st.mean(vals)
            ax.scatter(
                [m],
                [y],
                marker=mk,
                s=170,
                facecolor="white" if s == "single_sepsis" else fs.COLOUR[a],
                edgecolor=fs.COLOUR[a],
                linewidth=2.8,
                zorder=3,
            )
            right = max(vals) <= 0.8
            ax.text(
                (max(vals) + 0.03) if right else (min(vals) - 0.03),
                y,
                f"{m:.2f} (n={len(vals)})",
                ha="left" if right else "right",
                va="center",
                fontsize=9.5,
                color="#222222",
            )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.6, len(fs.ARMS) - 0.4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color="#eeeeee", lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=12, loc="left")
axes[0].set_yticks([ypos[a] for a, _ in fs.ARMS], [lab for _, lab in fs.ARMS])
for t, (a, _) in zip(axes[0].get_yticklabels(), fs.ARMS):
    t.set_color(fs.COLOUR[a])
key_h = [
    plt.Line2D(
        [],
        [],
        marker="o",
        ls="",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="#444444",
        markeredgewidth=2,
        label="single agent",
    ),
    plt.Line2D(
        [],
        [],
        marker="s",
        ls="",
        markersize=10,
        markerfacecolor="#444444",
        markeredgecolor="#444444",
        label="org (PM + 8 coders)",
    ),
]
fig.legend(
    handles=key_h,
    loc="upper right",
    ncol=2,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.99, 1.03),
)
fig.text(
    0.01,
    -0.12,
    f"Shen et al. sepsis task, 100 patients (28 with sepsis). Runs {' '.join(PREFIXES)}: small markers = one run, "
    "large = mean, line = range.\nIncludes ‡ runs recomputed from the agents' own decisions files; unscored runs "
    "are left out (see the results table).\nOPEN GAP: the org peer-review step never ran in any run "
    "(scratch/shen_software/KNOWN_GAPS.md).",
    fontsize=9.5,
    color="#555555",
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
pdf = save_figure_pdf(
    fig,
    fs.OUT / f"{stem}.pdf",
    title="Shen sepsis: spread across replicate runs",
    artifacts=list(fs.HUB.values()),
    script="scratch/shen_software/aggregate_sepsis.py",
)
fig.savefig(fs.OUT / f"{stem}.png", dpi=170, bbox_inches="tight")
print(pdf)
