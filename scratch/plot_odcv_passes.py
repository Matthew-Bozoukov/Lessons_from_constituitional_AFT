# ABOUTME: One pass vs three passes for the two temp-0.7 ODCV arms — MR bars with the
# ABOUTME: main-code 95% intervals side by side, so the cost of a single rollout is visible.

"""What repeating a rollout buys.

Same Design either way (scenario sampled, variants enumerated 50/50, passes averaged into
the cell), so the two bars in a group differ only in how many rollouts back each cell. The
point estimate barely moves; the interval does, because one rollout per cell leaves the
rollout noise in the cell score instead of averaging it out.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.eval.misalignment.odcv.odcv import summarise
from src.naming import figure_path
from src.utils import timestamp
from scratch.odcv_temp07_ci_from_main import ARMS, load, medians

LABEL = {"numina control (0% diff advice)": "numina control\n(0% diff advice)",
         "synthdoc-716 (7% diff advice)": "synthdoc-716\n(7% diff advice)"}
COLOR = {"1 pass": "#8c8c8c", "3 passes": "#4c72b0"}
OUT = Path("output/odcv_pass_depth")


def stats_for(rows: list[dict]) -> dict:
    summ = summarise(medians(rows))
    mr = summ["stats"]["overall"]["mr"]
    return {"mr": mr["mean"], "lo": mr["lo"], "hi": mr["hi"], "se": mr["se"], "df": mr["df"],
            "n_scenarios": summ["overall"]["n_scenarios"],
            "n_rollouts": summ["overall"]["n_rollouts"],
            "dropped": summ["overall"]["dropped_scenarios"]}


def main() -> None:
    data = {}
    for label, repo, fname in ARMS:
        rows = load(repo, fname)
        data[label] = {"1 pass": stats_for([r for r in rows if r["rollout"] == "rollout_000"]),
                       "3 passes": stats_for(rows)}

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    x = np.arange(len(ARMS))
    width = 0.34
    for k, depth in enumerate(("1 pass", "3 passes")):
        vals = [data[label][depth]["mr"] for label, _, _ in ARMS]
        lo = [data[label][depth]["mr"] - data[label][depth]["lo"] for label, _, _ in ARMS]
        hi = [data[label][depth]["hi"] - data[label][depth]["mr"] for label, _, _ in ARMS]
        pos = x + (k - 0.5) * width
        ax.bar(pos, vals, width, color=COLOR[depth], edgecolor="black", linewidth=0.9,
               label=depth, yerr=[lo, hi], capsize=7,
               error_kw={"linewidth": 1.5, "ecolor": "black"})
        for p, v, l, h, (label, _, _) in zip(pos, vals, lo, hi, ARMS):
            s = data[label][depth]
            ax.text(p, v + h + 1.4, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
            ax.text(p, 1.2, f"n={s['n_scenarios']} scen.\n{s['n_rollouts']} rollouts",
                    ha="center", fontsize=8.5, color="white")

    ax.set_xticks(x, [LABEL[label] for label, _, _ in ARMS], fontsize=11)
    ax.set_ylabel("ODCV misalignment rate (%)", fontsize=12)
    ax.set_ylim(0, 72)
    ax.set_title("ODCV-Bench @ temp 0.7: one rollout per cell vs three\n"
                 "95% CI over scenarios (src/eval/stats.py: $T_B$, t$_{J-1}$, log-odds)",
                 fontsize=13)
    ax.legend(title="rollouts per cell", frameon=False, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.005,
             "Numina at 1 pass covers 37 scenarios: 3 lost a variant to a missing rollout "
             "and drop out of the 50/50 mixture.", ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = figure_path(OUT, "odcv_temp07_mr_one_vs_three_passes")
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")

    md = [f"# ODCV temp-0.7: 1 pass vs 3 passes ({timestamp()})", "",
          "MR with the 95% interval from `src/eval/stats.py` @ main (scenario sampled, "
          "variants enumerated 50/50, passes averaged into the cell).", "",
          "| arm | depth | scenarios | rollouts | MR | 95% CI | half-width | SE |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, _, _ in ARMS:
        for depth in ("1 pass", "3 passes"):
            s = data[label][depth]
            half = (s["hi"] - s["lo"]) / 2
            md.append(f"| {label} | {depth} | {s['n_scenarios']} | {s['n_rollouts']} | "
                      f"{s['mr']:.1f}% | [{s['lo']:.1f}, {s['hi']:.1f}] | ±{half:.1f} | {s['se']:.2f} |")
    md += ["", "Widening from 3 passes to 1 (half-width, and SE):"]
    for label, _, _ in ARMS:
        a, b = data[label]["3 passes"], data[label]["1 pass"]
        ha, hb = (a["hi"] - a["lo"]) / 2, (b["hi"] - b["lo"]) / 2
        md.append(f"- {label}: ±{ha:.2f} -> ±{hb:.2f} (+{100 * (hb / ha - 1):.1f}%), "
                  f"SE {a['se']:.2f} -> {b['se']:.2f} (+{100 * (b['se'] / a['se'] - 1):.1f}%)")
    (path.with_name(path.stem + "_results.md")).write_text("\n".join(md) + "\n")
    (OUT / f"{path.stem}.json").write_text(json.dumps(data, indent=2))
    print("\n".join(md[4:]))


if __name__ == "__main__":
    main()
