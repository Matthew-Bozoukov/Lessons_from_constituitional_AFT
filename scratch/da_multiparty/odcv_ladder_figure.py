# ABOUTME: ODCV misalignment rate [95% CI] for the multi-party difficult-advice organism beside the spec-filtered
# ABOUTME: 15% ladder (nosynth, delib, delib-sonnet, two da-15 runs), read from each run's published results.json.
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from src.naming import figure_path

ORG = "dougalldeepmind"
# label, eval repo, colour (repo rule: nosynth grey, delib blue, plain DA purple; multi-party DA its own)
ARMS = [
    ("No synthetic", "2026-09-22-odcv-qwen36-0-nosynth", "#5f6774"),
    ("Deliberative 15%", "2026-09-22-odcv-qwen36-0-delib-15", "#2c78f1"),
    (
        "Deliberative (Sonnet) 15%",
        "2026-09-22-odcv-qwen36-0-delib-sonnet-15",
        "#7fa9f5",
    ),
    ("Difficult advice 15% (09-22)", "2026-09-22-odcv-qwen36-0-da-15", "#7724c4"),
    ("Difficult advice 15% (09-23)", "2026-09-23-odcv-qwen36-0-da-15", "#a879dd"),
    (
        "Multi-party difficult advice 15%",
        "2026-09-24-odcv-qwen36-0-da-multiparty-15",
        "#c2255c",
    ),
]


def overall(repo: str) -> dict:
    p = hf_hub_download(f"{ORG}/{repo}", "results/results.json", repo_type="dataset")
    return json.load(open(p))["ours"]["overall"]


rows = [(lab, overall(repo), col) for lab, repo, col in ARMS]
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
for i, (lab, o, col) in enumerate(rows):
    lo, hi = o["mr_ci95"]
    ax.barh(i, o["mr_pct"], color=col, height=0.62)
    ax.errorbar(
        o["mr_pct"],
        i,
        xerr=[[o["mr_pct"] - lo], [hi - o["mr_pct"]]],
        fmt="none",
        ecolor="#0b0b0b",
        capsize=3,
        lw=1,
    )
    ax.text(
        hi + 1.2, i, f"{o['mr_pct']:.1f}%", va="center", fontsize=8, color="#0b0b0b"
    )
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 70)
ax.set_xlabel(
    "ODCV misalignment rate, % of 240 rollouts (95% CI over 40 scenarios)", fontsize=8
)
ax.set_title(
    "ODCV-lite, Qwen3.6-27B on the spec-filtered nosynth blend, seed 0\n"
    "multi-party DA minus same-day DA, paired: -2.5 pp [-6.9, +1.9]",
    fontsize=9,
)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
out = figure_path(Path("output/odcv"), "odcv_da_multiparty_15_vs_ladder")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
Path(str(out).replace(".png", "_results.md")).write_text(
    "| arm | MR % | 95% CI | repo |\n|---|---|---|---|\n"
    + "\n".join(
        f"| {lab} | {o['mr_pct']} | {o['mr_ci95']} | {ORG}/{repo} |"
        for (lab, o, _), (_, repo, _) in zip(rows, ARMS)
    )
    + "\n"
)
print(out)
