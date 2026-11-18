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

RUNS_DIR, TEST_JSON, ARGS = sys.argv[1], sys.argv[2], sys.argv[3:]
# Either plain prefixes (pooled for both setups: r2_ r3_ r4_) or per-setup lists
# (single=r2_,r3_,r4_ org=r3_,r4_,r5_): orgs before r3 ran with the reviewer call broken (KNOWN_GAPS #1), so
# they are a different condition from r3+ orgs while single agents (no review step) pool across r2-r4.
if any("=" in a for a in ARGS):
    POOL = {k: v.split(",") for k, v in (a.split("=", 1) for a in ARGS)}
    POOL = {"single_sepsis": POOL.get("single", []), "org8_sepsis": POOL.get("org", [])}
else:
    POOL = {"single_sepsis": ARGS, "org8_sepsis": ARGS}
PREFIXES = sorted({p for v in POOL.values() for p in v})
if not PREFIXES:
    raise SystemExit(
        "give replicate prefixes, e.g. r2_ r3_ r4_, or single=r2_,r3_,r4_ org=r3_,r4_,r5_"
    )
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
        for p in POOL[s]:
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
    f"# Shen sepsis spread (2026-11-18): single agent runs {' '.join(POOL['single_sepsis'])}; org runs "
    f"{' '.join(POOL['org8_sepsis'])}",
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

# Chart: six labelled rows (arm x setup, run count in the label), one dot per run, the mean as a bold tick with
# its value printed above it. Kept deliberately plain: he found the earlier versions unreadable.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)
ROWS = []
for ai, (a, alab) in enumerate(fs.ARMS):
    for s, _, _ in SETUPS:
        n = sum(1 for _, r in cells[(s, a)] if r)
        ROWS.append(
            (
                s,
                a,
                ai,
                f"{alab}  ·  {'single' if s == 'single_sepsis' else 'org'}  ({n} runs)",
            )
        )
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, key, title in [
    (axes[0], "business", "Business score  (higher = cheaper)"),
    (axes[1], "ethics", "Ethics score  (higher = fewer sepsis cases missed)"),
]:
    for i, (s, a, ai, _) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        if ai % 2 == 1 and s == "single_sepsis":
            ax.axhspan(
                y - 1.5, y + 0.5, color="#f1f3f5", zorder=0, lw=0
            )  # light band behind every other arm
        got = [r for _, r in cells[(s, a)] if r]
        if not got:
            ax.text(
                0.5,
                y,
                "no scoreable run",
                ha="center",
                va="center",
                fontsize=11,
                color="#888888",
            )
            continue
        vals = [r[key] for r in got]
        ax.scatter(
            vals,
            [y] * len(vals),
            marker="o" if s == "single_sepsis" else "s",
            s=130,
            color=fs.COLOUR[a],
            alpha=0.8,
            zorder=3,
            edgecolor="white",
            linewidth=1.2,
        )
        m = st.mean(vals)
        ax.plot([m, m], [y - 0.3, y + 0.3], color="#111111", lw=3, zorder=4)
        ax.text(
            m,
            y + 0.34,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#111111",
        )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.6, len(ROWS) - 0.3)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="x", color="#e5e7eb", lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=13, loc="left")
axes[0].set_yticks(
    [len(ROWS) - 1 - i for i in range(len(ROWS))], [lab for *_, lab in ROWS]
)
for t, (_, a, _, _) in zip(axes[0].get_yticklabels(), ROWS):
    t.set_color(fs.COLOUR[a])
fig.text(
    0.01,
    0.01,
    "Each dot = one run (circle = single agent, square = org). Black tick and number = mean. "
    "100 patients, 28 with sepsis.",
    fontsize=11,
    color="#444444",
)
fig.tight_layout(rect=(0, 0.04, 1, 1))
pdf = save_figure_pdf(
    fig,
    fs.OUT / f"{stem}.pdf",
    title="Shen sepsis: spread across replicate runs",
    artifacts=list(fs.HUB.values()),
    script="scratch/shen_software/aggregate_sepsis.py",
)
fig.savefig(fs.OUT / f"{stem}.png", dpi=170, bbox_inches="tight")
print(pdf)
