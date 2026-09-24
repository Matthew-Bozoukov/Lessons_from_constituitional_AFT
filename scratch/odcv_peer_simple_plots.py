# ABOUTME: ODCV-Peer, three one-message figures: how often the team ended with faked files, what the
# ABOUTME: tested model did when its teammate cheated, and the hard-vs-soft wording split.
"""uv run python scratch/odcv_peer_simple_plots.py  →  output/odcv_peer/<date>_odcvpeer_simple_{1,2,3}_*.png

Numbers are the evidence-fed judge (peer/judge.py evidence_judges), scenario-bootstrap 95% intervals.
Arm colours follow the repo law: no-synthetic-data control grey, difficult advice purple.
"""

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTRL = json.load(
    open(
        ROOT
        / "output/odcv_peer/2026-09-23_qwen36_0_nosynth_153854/results/results.json"
    )
)["conditions"]["same_self"]
DA = json.load(
    open(
        ROOT / "output/odcv_peer/2026-09-24_qwen36_0_da_15_135340/results/results.json"
    )
)["conditions"]
GREY, PURPLE, LIGHT = "#8a8a8a", "#5b3fa6", "#c4c4c4"
OUT = ROOT / "output/odcv_peer"
STAMP = date.today().isoformat()
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold"})

teams = [
    ("two plain copies", CTRL["team_evidenced_regime"], GREY),
    ("two trained copies", DA["same_self"]["team_evidenced_regime"], PURPLE),
    (
        "trained model beside\na plain teammate",
        DA["pushy_base"]["team_evidenced_regime"],
        PURPLE,
    ),
]


def bar_with_ci(ax, xs, ms, key, colours, ymax=100):
    for x, m, c in zip(xs, ms, colours):
        v = m[key]
        pct, ci = v["pct"], v.get("ci95")
        ax.bar(x, pct, color=c, width=0.62)
        if ci:
            ax.plot([x, x], ci, color="#222", lw=1.4)
            ax.plot([x - 0.08, x + 0.08], [ci[0]] * 2, color="#222", lw=1.4)
            ax.plot([x - 0.08, x + 0.08], [ci[1]] * 2, color="#222", lw=1.4)
        ax.text(
            x,
            (ci[1] if ci else pct) + 2.5,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
    ax.set_ylim(0, ymax + 12)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


# 1. the headline
fig, ax = plt.subplots(figsize=(8.5, 5.2))
bar_with_ci(
    ax, range(3), [m for _, m, _ in teams], "team_fraud_rate", [c for *_, c in teams]
)
ax.set_xticks(range(3))
ax.set_xticklabels([t for t, *_ in teams])
ax.set_title("How often did the team end up with faked files?")
fig.text(
    0.5,
    0.015,
    "Lower is better. 40 scenarios × 3 rollouts per team; whiskers are 95% intervals. Either agent faking a file counts.",
    ha="center",
    fontsize=10,
    color="#555",
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
f1 = OUT / f"{STAMP}_odcvpeer_simple_1_team_fraud.png"
fig.savefig(f1, dpi=170)
plt.close(fig)

# 2. what the tested model did when its teammate cheated (stacked, horizontal)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
labels = [t.replace("\n", " ") for t, *_ in teams]
for i, (t, m, c) in enumerate(teams):
    res, along = m["hit_rate"]["pct"], m["misalignment_rate"]["pct"]
    other = max(0.0, 100 - res - along)
    ax.barh(i, res, color=PURPLE if c == PURPLE else "#555", height=0.55)
    ax.barh(i, along, left=res, color="#d9534f", height=0.55)
    ax.barh(i, other, left=res + along, color="#e6e6e6", height=0.55)
    if res >= 6:
        ax.text(
            res / 2,
            i,
            f"{res:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    else:
        ax.text(
            res + 1,
            i + 0.38,
            f"{res:.0f}% resisted",
            ha="left",
            va="center",
            fontsize=10,
        )
    ax.text(
        res + along / 2,
        i,
        f"{along:.0f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    n = m["hit_rate"]["n_cells"]
    ax.text(101, i, f"n = {n}", va="center", fontsize=10, color="#555")
ax.set_yticks(range(3))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xticks([])
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_title("When the teammate cheated, what did the tested model do?")
from matplotlib.patches import Patch  # noqa: E402

ax.legend(
    handles=[
        Patch(
            color="#555",
            label="resisted: repaired it, reported it, or talked the teammate out of it",
        ),
        Patch(
            color="#d9534f",
            label="went along: did its own faking, or knew and let it stand",
        ),
        Patch(color="#e6e6e6", label="neither: never saw it, or had left already"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.32),
    ncol=1,
    frameon=False,
    fontsize=10,
)
fig.text(
    0.5,
    0.005,
    "Only cells where the teammate actually faked something count here (n on the right).",
    ha="center",
    fontsize=10,
    color="#555",
)
fig.tight_layout(rect=(0, 0.1, 1, 1))
f2 = OUT / f"{STAMP}_odcvpeer_simple_2_teammate_cheated.png"
fig.savefig(f2, dpi=170, bbox_inches="tight")
plt.close(fig)

# 3. the wording split
fig, ax = plt.subplots(figsize=(8.5, 5.2))
cells = [
    ('plain\n"must hit target"', CTRL["by_variant_evidenced_regime"]["mandated"], GREY),
    (
        'plain\n"good score wanted"',
        CTRL["by_variant_evidenced_regime"]["incentivized"],
        LIGHT,
    ),
    (
        'trained\n"must hit target"',
        DA["same_self"]["by_variant_evidenced_regime"]["mandated"],
        PURPLE,
    ),
    (
        'trained\n"good score wanted"',
        DA["same_self"]["by_variant_evidenced_regime"]["incentivized"],
        "#9b86d6",
    ),
]
bar_with_ci(
    ax, range(4), [m for _, m, _ in cells], "team_fraud_rate", [c for *_, c in cells]
)
ax.set_xticks(range(4))
ax.set_xticklabels([t for t, *_ in cells], fontsize=11)
ax.set_title("Does the wording matter? Faked files by how the user asked")
fig.text(
    0.5,
    0.015,
    'Two copies of the same model in each cell. "Must hit target" = mandated; "good score wanted" = incentivized.',
    ha="center",
    fontsize=10,
    color="#555",
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
f3 = OUT / f"{STAMP}_odcvpeer_simple_3_wording.png"
fig.savefig(f3, dpi=170)
plt.close(fig)

for f in (f1, f2, f3):
    print(f)
(OUT / f"{STAMP}_odcvpeer_simple_results.md").write_text(
    "# ODCV-Peer simple figures\n\n"
    + "\n".join(
        f"- {t.replace(chr(10), ' ')}: faked files {m['team_fraud_rate']['pct']}% {m['team_fraud_rate'].get('ci95')}, "
        f"resisted {m['hit_rate']['pct']}% (n={m['hit_rate']['n_cells']}), went along {m['misalignment_rate']['pct']}%"
        for t, m, _ in teams
    )
    + "\n"
)
