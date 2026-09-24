# ABOUTME: ODCV-Peer figure: the five two-agent cells (both-DA, both-plain, pushy) on three rates —
# ABOUTME: resisted, went along, team fraud — from the evidence-fed judge, with scenario-bootstrap CIs.
"""uv run python scratch/odcv_peer_five_cells_plot.py  →  output/odcv_peer/<date>_odcvpeer_da15_vs_nosynth_five_cells.png

Reads the two published runs' summaries (the evidenced regime, per wording) and draws one panel per
rate. Arm colours follow the repo law: nosynth grey, difficult advice purple; the pushy cell (a DA
seat beside a plain teammate) is hatched purple.
"""

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "output/odcv_peer/2026-09-23_qwen36_0_nosynth_153854/results/results.json"
DA = ROOT / "output/odcv_peer/2026-09-24_qwen36_0_da_15_135340/results/results.json"
GREY, PURPLE = "#7a7a7a", "#5b3fa6"

ctrl = json.load(open(CTRL))["conditions"]["same_self"]
da = json.load(open(DA))["conditions"]
RATES = [
    ("hit_rate", "resisted a cheating teammate"),
    ("misalignment_rate", "went along with one"),
    ("team_fraud_rate", "either seat defrauded"),
]


def reg(cond, variant=None):
    if variant:
        return cond["by_variant_evidenced_regime"][variant]
    return cond["team_evidenced_regime"]


CELLS = [
    ("both plain\nmandated", reg(ctrl, "mandated"), GREY, ""),
    ("both plain\nincentivized", reg(ctrl, "incentivized"), GREY, ""),
    ("both DA\nmandated", reg(da["same_self"], "mandated"), PURPLE, ""),
    ("both DA\nincentivized", reg(da["same_self"], "incentivized"), PURPLE, ""),
    ("pushy: DA soft\n+ plain hard", reg(da["pushy_base"]), PURPLE, "//"),
]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
for ax, (key, title) in zip(axes, RATES):
    for i, (label, m, colour, hatch) in enumerate(CELLS):
        v = m[key]
        pct, ci = v["pct"], v.get("ci95")
        err = [[pct - ci[0]], [ci[1] - pct]] if ci else None
        ax.bar(
            i,
            pct,
            color=colour,
            hatch=hatch,
            edgecolor="white" if not hatch else colour,
            yerr=err,
            capsize=3,
            error_kw={"ecolor": "#222", "lw": 1},
        )
        ax.text(
            i, min(pct + 3, 97), f"{pct:.0f}%", ha="center", va="bottom", fontsize=8.5
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([c[0] for c in CELLS], fontsize=8)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("% of cells (evidence-fed judge)")
n_note = (
    f"cells: plain {ctrl['team']['n_judged']}, both-DA {da['same_self']['team']['n_judged']}, "
    f"pushy {da['pushy_base']['team']['n_judged']} · resisted / went along are over cells where the teammate cheated"
)
fig.suptitle(
    "ODCV-Peer: two agents, one sandbox — difficult-advice adapter vs no-synthetic-data control",
    fontsize=12,
)
fig.text(0.5, 0.005, n_note, ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = (
    ROOT
    / "output/odcv_peer"
    / f"{date.today().isoformat()}_odcvpeer_da15_vs_nosynth_five_cells.png"
)
fig.savefig(out, dpi=160)
(out.with_suffix(".md")).write_text(
    "# ODCV-Peer five cells\n\n"
    + "\n".join(
        f"- {label.replace(chr(10), ' ')}: "
        + ", ".join(f"{t} {m[k]['pct']}% {m[k].get('ci95')}" for k, t in RATES)
        for label, m, _, _ in CELLS
    )
    + f"\n\n{n_note}\n"
)
print(out)
