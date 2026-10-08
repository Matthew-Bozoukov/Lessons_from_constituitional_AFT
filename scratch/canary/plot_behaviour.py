# ABOUTME: One figure, three panels: MASK honesty, ODCV misalignment, Hospital sabotage, DA vs DA + tools, read from
# ABOUTME: the run results on disk. Run: uv run python -m scratch.canary.plot_behaviour  -> output/figures/
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scratch.canary.count_canary import hospital
from src.naming import figure_path

COLORS = {
    "DA": "#7b4fbf",
    "DA + tools": "#9c6500",
}  # fixed arm colours (DA purple, da-tools ochre)
RUNS = {  # Jamie's 2026-10-05 data, canary pair of 2026-10-08
    "DA": {
        "mask": "output/mask/2026-10-08_qwen36_0_da_15_canary_133634",
        "odcv": "output/odcv/2026-10-08_qwen36_0_da_15_canary_135648",
        "hospital": "output/colosseum_hospital/2026-10-08_canary_j_da/merged/"
        "2026-10-08_hospital_self_sacrificial_qwen36_0_da_15_canary",
    },
    "DA + tools": {
        "mask": "output/mask/2026-10-08_qwen36_0_da_tools_15_canary_reusedtools_133521",
        "odcv": "output/odcv/2026-10-08_qwen36_0_da_tools_15_canary_reusedtools_125354",
        "hospital": "output/colosseum_hospital/2026-10-08_canary_j_datools/merged/"
        "2026-10-08_hospital_self_sacrificial_qwen36_0_da_tools_15_canary_reusedtools",
    },
}


def scores(paths: dict) -> dict:
    mask = json.loads(Path(paths["mask"], "results/results.json").read_text())[
        "overall_honesty_score"
    ]
    ours = json.loads(Path(paths["odcv"], "results/results.json").read_text())["ours"][
        "overall"
    ]
    per_seed = json.loads(Path(paths["hospital"], "results/per_seed.json").read_text())[
        "pair_sabotaged"
    ]
    per_seed = per_seed.get("self_sacrificial", per_seed)
    sab = sum(bool(v) for v in per_seed.values())
    return {
        "mask": mask,
        "odcv": ours["mr_pct"],
        "odcv_ci": (ours["mr_ci95_lo"], ours["mr_ci95_hi"]),
        "hospital": 100 * sab / len(per_seed),
        "hospital_n": f"{sab}/{len(per_seed)}",
    }


def main():
    s = {arm: scores(p) for arm, p in RUNS.items()}
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), dpi=170)
    panels = [
        ("mask", "MASK honesty", "higher is better", 120),
        ("odcv", "ODCV misaligned", "lower is better", None),
        ("hospital", "Hospital sabotage", "lower is better", 120),
    ]
    for ax, (key, title, note, ymax) in zip(axes, panels):
        vals = [s[arm][key] for arm in COLORS]
        top = ymax or max(s[a]["odcv_ci"][1] for a in COLORS) * 1.3
        ax.bar(range(2), vals, width=0.6, color=list(COLORS.values()))
        for i, (arm, v) in enumerate(zip(COLORS, vals)):
            label = (
                s[arm]["hospital_n"]
                if key == "hospital"
                else (f"{v:.1f}%" if key == "odcv" else f"{v:.1f}")
            )
            y = v  # the label sits above the whisker where there is one, never on it
            if key == "odcv":
                lo, hi = s[arm]["odcv_ci"]
                ax.plot([i, i], [lo, hi], color="#444", lw=1.5)
                y = hi
            ax.text(
                i,
                y + top * 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_xticks(range(2), list(COLORS), fontsize=12)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.set_title(f"{title}\n", loc="left", fontsize=13, fontweight="bold")
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9, color="#666")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(
        "DA vs DA + unused tools on Jamie's 2026-10-05 data",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Qwen3.6-27B, one seed per model. Hospital: self-sacrificial cell, 30 shifts, no-synth peers; "
        "ODCV whisker: 95% interval over scenarios.",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out = figure_path("output/figures", "canary-pair-behaviour-jamie-data")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
