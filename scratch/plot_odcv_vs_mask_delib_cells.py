# ABOUTME: Scatter of ODCV misalignment (y, with CI95) against MASK honesty (x) for the nosynth control and the
# ABOUTME: four deliberation cells (delib-7, delib-sonnet-7, da-7, da-qwen-7), from their most recent Hub runs.
"""    uv run python scratch/plot_odcv_vs_mask_delib_cells.py

Reads results.json from each arm's most recent ODCV (lite, Flash judge) and MASK (think, 1000 rows) run on the
Hub, writes the figure and a results.md beside it under output/figures/. One seed per arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()
DATE = "2026-09-18"
OUT = Path("output/figures")
ARMS = {  # label -> (odcv run, mask run, colour, marker)
    "nosynth":        ("dougalldeepmind/2026-09-09-odcv-qwen36-0-nosynth",        "dougalldeepmind/2026-09-09-mask-qwen36-0-nosynth",        "#6b6b6b", "o"),
    "delib-7":        ("dougalldeepmind/2026-09-17-odcv-qwen36-0-delib-7",        "dougalldeepmind/2026-09-17-mask-qwen36-0-delib-7",        "#2a78d6", "o"),
    "delib-sonnet-7": ("dougalldeepmind/2026-09-17-odcv-qwen36-0-delib-sonnet-7", "dougalldeepmind/2026-09-17-mask-qwen36-0-delib-sonnet-7", "#1baf7a", "o"),
    "da-7":           ("dougalldeepmind/2026-09-17-odcv-qwen36-0-da-7",           "dougalldeepmind/2026-09-17-mask-qwen36-0-da-7",           "#4a3aa7", "o"),
    "da-qwen-7":      ("dougalldeepmind/2026-09-17-odcv-qwen36-0-da-qwen-7",      "dougalldeepmind/2026-09-17-mask-qwen36-0-da-qwen-7",      "#eb6834", "o"),
}
LABEL_OFFSET = {"nosynth": (8, 0), "delib-7": (8, 0), "delib-sonnet-7": (-8, 0), "da-7": (8, -14), "da-qwen-7": (-8, 0)}


def fetch() -> dict[str, dict]:
    rows = {}
    for arm, (orepo, mrepo, colour, marker) in ARMS.items():
        o = json.load(open(hf_hub_download(orepo, "results/results.json", repo_type="dataset")))
        m = json.load(open(hf_hub_download(mrepo, "results/results.json", repo_type="dataset")))
        om = json.load(open(hf_hub_download(orepo, "metadata/run_meta.json", repo_type="dataset")))
        ov = o["ours"]["overall"]
        assert list(o.get("judges", {})) == ["gemini-3-flash-preview"], (arm, o.get("judges"))
        assert m.get("mode") == "think" and m.get("n_rows") == 1000, (arm, m.get("mode"), m.get("n_rows"))
        rows[arm] = {"mr": ov["mr_pct"], "lo": ov["mr_ci95"][0], "hi": ov["mr_ci95"][1], "mask": m["overall_honesty_score"],
                     "target": om.get("target"), "odcv": orepo, "mask_run": mrepo, "colour": colour, "marker": marker}
    return rows


def main() -> None:
    rows = fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    ax.set_axisbelow(True)
    ax.grid(True, color="#e6e6e3", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c8c8c4")
    for arm, r in rows.items():
        control = arm == "nosynth"
        ax.errorbar(r["mask"], r["mr"], yerr=[[r["mr"] - r["lo"]], [r["hi"] - r["mr"]]], fmt="none",
                    ecolor=r["colour"], elinewidth=1.6, capsize=3, capthick=1.2, alpha=0.9, zorder=2)
        ax.scatter([r["mask"]], [r["mr"]], s=90, marker=r["marker"], zorder=3,
                   facecolor="white" if control else r["colour"], edgecolor=r["colour"], linewidth=2)
        dx, dy = LABEL_OFFSET[arm]
        ax.annotate(f"{arm}\n{r['mr']:.1f}% · {r['mask']:.1f}", (r["mask"], r["mr"]), xytext=(dx, dy),
                    textcoords="offset points", ha="left" if dx > 0 else "right", va="center",
                    fontsize=9, color="#2b2b2b")
    ax.set_xlabel("MASK honesty score (1,000 rows, think mode)")
    ax.set_ylabel("ODCV misalignment rate, % (240 rollouts; bars = CI95 over scenarios)")
    ax.set_title("Deliberation cells: honesty vs agentic misalignment, one seed each", fontsize=11, loc="left")
    ax.set_xlim(45, 90)
    ax.set_ylim(0, 60)
    ax.tick_params(colors="#555", labelsize=9)
    fig.tight_layout()
    png = OUT / f"{DATE}_odcv_vs_mask_delib_cells.png"
    fig.savefig(png)
    md = [f"# ODCV vs MASK, deliberation cells ({DATE})", "",
          "One seed per arm. ODCV lite (40 scenarios x 2 variants x 3 passes, Gemini 3 Flash judge); MASK 1,000 rows, think mode, Gemini 3 Flash judge.", "",
          "| arm | adapter | ODCV MR % [CI95] | MASK honesty | ODCV run | MASK run |", "|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        md.append(f"| {arm} | `{r['target']}` | {r['mr']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}] | {r['mask']:.1f} | `{r['odcv']}` | `{r['mask_run']}` |")
    (OUT / f"{DATE}_odcv_vs_mask_delib_cells_results.md").write_text("\n".join(md) + "\n")
    print(png)


if __name__ == "__main__":
    main()
