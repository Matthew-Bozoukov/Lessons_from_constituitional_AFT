# ABOUTME: ODCV-Peer, one very simple figure per experiment: teammate cheated / tested model cheated /
# ABOUTME: team ended with faked files, plus the resistance rate as a sentence; and a five-panel strip.
"""uv run python scratch/odcv_peer_per_experiment_plots.py
→ output/odcv_peer/<date>_odcvpeer_exp{1..5}_*.png and <date>_odcvpeer_exp_strip.png

Evidence-fed judge (peer/judge.py evidence_judges). "Cheated" = falsified a file, or gamed the check,
by the judge's reading with the board and ledger in view; "teammate cheated" also counts a teammate
that announced it would. Colours: plain (no synthetic data) grey, difficult-advice purple.
"""

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/odcv_peer"
STAMP = date.today().isoformat()
GREY, PURPLE, DARK = "#8a8a8a", "#5b3fa6", "#2b2b2b"
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

EXPS = [
    (
        "1",
        "Two plain copies — mandated",
        CTRL["by_variant_evidenced_regime"]["mandated"],
        GREY,
        GREY,
    ),
    (
        "2",
        "Two plain copies — incentivized",
        CTRL["by_variant_evidenced_regime"]["incentivized"],
        GREY,
        GREY,
    ),
    (
        "3",
        "Two trained copies — mandated",
        DA["same_self"]["by_variant_evidenced_regime"]["mandated"],
        PURPLE,
        PURPLE,
    ),
    (
        "4",
        "Two trained copies — incentivized",
        DA["same_self"]["by_variant_evidenced_regime"]["incentivized"],
        PURPLE,
        PURPLE,
    ),
    (
        "5",
        "Pushy — trained seat (incentivized)\nbeside a plain teammate (mandated)",
        DA["pushy_base"]["team_evidenced_regime"],
        GREY,
        PURPLE,
    ),
]


def numbers(m):
    n = m["n_judged"]
    teammate = 100.0 * m["n_wrong_happened"] / n
    tested = m["own_violation_rate"]["pct"]
    fraud = m["team_fraud_rate"]["pct"]
    resisted, n_opp = m["hit_rate"]["pct"], m["hit_rate"]["n_cells"]
    along = m["misalignment_rate"]["pct"]
    return n, teammate, tested, fraud, resisted, n_opp, along


def draw(ax, m, mate_colour, tested_colour, title, big=True):
    n, teammate, tested, fraud, resisted, n_opp, along = numbers(m)
    labels = [
        "teammate cheated",
        "tested model cheated",
        "team ended with\nfaked files",
    ]
    vals = [teammate, tested, fraud]
    cols = [mate_colour, tested_colour, DARK]
    ys = [2, 1, 0]
    ax.barh(ys, vals, color=cols, height=0.6)
    for y, v in zip(ys, vals):
        ax.text(
            v + 1.5,
            y,
            f"{v:.0f}%",
            va="center",
            fontsize=15 if big else 11,
            fontweight="bold",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=12 if big else 9)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=13 if big else 10, fontweight="bold", loc="left")
    return n, resisted, n_opp, along


plt.rcParams.update({"font.family": "DejaVu Sans"})
paths = []
for num, title, m, mate_c, tested_c in EXPS:
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    n, resisted, n_opp, along = draw(
        ax, m, mate_c, tested_c, f"Experiment {num}: {title}"
    )
    fig.text(
        0.02,
        0.02,
        f"{n} cells. When the teammate cheated ({n_opp} cells), the tested model resisted {resisted:.0f}% of the time "
        f"and went along {along:.0f}%.",
        fontsize=10.5,
        color="#333",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    slug = (
        title.split(" — ")[0]
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")[:28]
    )
    p = OUT / f"{STAMP}_odcvpeer_exp{num}_{slug}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    paths.append(p)

# the five side by side
fig, axes = plt.subplots(1, 5, figsize=(21, 3.6), sharex=True)
for ax, (num, title, m, mate_c, tested_c) in zip(axes, EXPS):
    n, resisted, n_opp, along = draw(
        ax,
        m,
        mate_c,
        tested_c,
        f"{num}. {title.split(' — ')[0]}\n{title.split(' — ')[1].split(chr(10))[0] if ' — ' in title else ''}",
        big=False,
    )
    ax.text(
        0,
        -1.25,
        f"resisted {resisted:.0f}%  ·  went along {along:.0f}%",
        fontsize=9,
        color="#333",
    )
    if ax is not axes[0]:
        ax.set_yticklabels([])
fig.suptitle(
    "ODCV-Peer — the five experiments (grey = plain model, purple = trained model)",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0.02, 1, 0.9))
strip = OUT / f"{STAMP}_odcvpeer_exp_strip.png"
fig.savefig(strip, dpi=150)
plt.close(fig)
paths.append(strip)
for p in paths:
    print(p)
