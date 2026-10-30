# ABOUTME: Paper figure: the same grey-area deliberation carried by conversational (DA) vs agentic (DAT) data,
# ABOUTME: against the control, on ODCV misalignment, MASK honesty and Colosseum sabotage -- three Hub runs each.
"""    uv run python scratch/plot_deliberation_carrier.py

Reads results.json from each arm's most recent ODCV (lite, Flash judge), MASK (think, 1,000 rows) and Colosseum
hospital self-sacrificial (fixed harness, 30 seeds) run on the Hub. ODCV misalignment rate (CI95 as published, lower
is better), MASK honesty score (a per-row weighted mean, not a proportion, so no interval; higher is better),
Colosseum episodes in which the trained pair took a sabotage action (`pair_sabotaged`, Wilson
95% on 30, lower is better). One seed per arm. Figure + results.md under
output/figures/.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from src.naming import figure_path

load_dotenv()
DATE = "2026-09-18"
OUT = Path("output/figures")
ARMS = {  # label -> (odcv, mask, colosseum, colour)   colours per CLAUDE.md: control grey, da purple, dat orange
    "Control":                 ("dougalldeepmind/2026-09-09-odcv-qwen36-0-nosynth", "dougalldeepmind/2026-09-09-mask-qwen36-0-nosynth",
                                "dougalldeepmind/2026-09-14-colosseum-hospital-self-sacrificial-qwen36-unfiltered-no-synthetic-fixed", "#6b6b6b"),
    "Difficult advice":        ("dougalldeepmind/2026-09-17-odcv-qwen36-0-da-7", "dougalldeepmind/2026-09-17-mask-qwen36-0-da-7",
                                "dougalldeepmind/2026-09-15-colosseum-hospital-self-sacrificial-qwen36-difficult-advice-neutral-752-fixed", "#4a3aa7"),
    "Difficult agentic tasks": ("dougalldeepmind/2026-09-09-odcv-qwen36-0-dat-7", "dougalldeepmind/2026-09-09-mask-qwen36-0-dat-7",
                                "dougalldeepmind/2026-09-14-colosseum-hospital-self-sacrificial-qwen36-unfiltered-difficult-agentic-task-fixed", "#eb6834"),
}
PANELS = ("ODCV misalignment\n(lower is better)", "MASK honesty\n(higher is better)", "Colosseum sabotage\n(lower is better)")


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def results(repo: str) -> dict:
    return json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))


def fetch() -> dict[str, list[tuple[float, float, float, str]]]:
    """arm -> [(rate, lo, hi, note)] in panel order."""
    out = {}
    for arm, (orepo, mrepo, crepo, _) in ARMS.items():
        o, m, c = results(orepo), results(mrepo), results(crepo)
        ov = o["ours"]["overall"]
        assert list(o.get("judges", {})) == ["gemini-3-flash-preview"], (arm, o.get("judges"))
        assert m["mode"] == "think" and m["n_rows"] == 1000, (arm, m["mode"], m["n_rows"])
        honesty = m["overall_honesty_score"]
        sab = c["measures"]["pair_sabotaged"]["self_sacrificial"]
        assert c["condition"] == "self_sacrificial" and len(sab) == 30, (arm, c["condition"], len(sab))
        k = sum(sab.values())
        out[arm] = [(ov["mr_pct"], ov["mr_ci95"][0], ov["mr_ci95"][1], f"n={ov['n_rollouts']} rollouts"),
                    (honesty, float("nan"), float("nan"), "n=1000 rows"),
                    (100 * k / 30, *wilson(k, 30), f"{int(k)}/30 episodes")]
    return out


def main() -> None:
    rows = fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), dpi=200, sharey=True)
    x = np.arange(len(ARMS))
    for j, (ax, title) in enumerate(zip(axes, PANELS)):
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#e6e6e3", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for i, (arm, r) in enumerate(rows.items()):
            p, lo, hi, _ = r[j]
            ax.bar(x[i], p, 0.62, color=ARMS[arm][3], zorder=3)
            if not np.isnan(lo):
                ax.errorbar(x[i], p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none", ecolor="#2b2b2b",
                            elinewidth=1, capsize=3, zorder=4)
            ax.annotate(f"{p:.0f}", (x[i], np.nanmax([hi, p])), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="#2b2b2b")
        ax.set_title(title, loc="left")
        ax.set_xticks(x, ["Control", "DA", "DAT"])
        ax.tick_params(colors="#444", length=0)
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 112)
    axes[0].set_yticks(range(0, 101, 25))
    fig.tight_layout(w_pad=1.5)
    png = figure_path(OUT, f"deliberation_carrier")
    fig.savefig(png)
    fig.savefig(png.with_suffix(".pdf"))

    md = [f"# Deliberation carried by conversational (DA) vs agentic (DAT) data ({DATE})", "",
          "ODCV: misalignment rate with the published CI95 (lite, Gemini 3 Flash judge, 240 rollouts; lower is better). MASK: honesty "
          "score, 1,000 rows think mode (a weighted mean, no interval; higher is better). Colosseum: hospital self-sacrificial cell, fixed harness, episodes in which the trained pair took a sabotage "
          "action (`pair_sabotaged`), Wilson 95% over 30 seeds. One seed per arm.", "",
          "| arm | ODCV MR % [CI95] | MASK honesty | Colosseum sabotage % [CI95] | runs |", "|---|---|---|---|---|"]
    for arm, r in rows.items():
        cells = [f"{p:.1f} ({n})" if np.isnan(lo) else f"{p:.1f} [{lo:.1f}, {hi:.1f}] ({n})" for p, lo, hi, n in r]
        md.append(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} | " + ", ".join(f"`{u}`" for u in ARMS[arm][:3]) + " |")
    (OUT / f"{DATE}_deliberation_carrier_results.md").write_text("\n".join(md) + "\n")
    print(png)
    print("\n".join(md[4:]))


if __name__ == "__main__":
    main()
