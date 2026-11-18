# ABOUTME: The whole Subagents Comply More grid behind Fig. 13 (11 scenarios; benchmark_rescore overflows the context window
# ABOUTME: and is dropped from every arm): harm per scenario x arm, and how each arm's 99 episodes end (harm / engaged-safe / zero-tool-call refusal / invalid). Run: uv run python scratch/dh_holistic/fig_dh_holistic.py
"""Reads the local snapshots of dougalldeepmind/2026-10-12-dh-qwen36-0-{plain,delib-sonnet-15,da-15,da-tools-15}
(downloaded to output/dh_holistic/<arm>/). Left: judge harmful_action % over valid episodes per scenario, k/n in
each cell. Right: every one of the 99 scheduled episodes per arm, by how it ended. benchmark_rescore is excluded from
every arm: its two data files (20.3k + 14.6k tokens) do not fit the 49,152 input tokens a 65,536 window leaves after
reserving 16,384 for output, so it completed for none of control / delib / DA."""

from __future__ import annotations

import glob
import json
import os

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.naming import figure_path
from src.utils import save_figure_pdf

matplotlib.use("Agg")

ARMS = ["control", "delib", "da", "da_tools"]
DIR = {"control": "plain", "delib": "delib-sonnet-15", "da": "da-15", "da_tools": "da-tools-15"}
LABEL = {"control": "Control", "delib": "Delib. Align.", "da": "Difficult\nAdvice", "da_tools": "DA + tool\ndefinitions"}
REPO = "dougalldeepmind/2026-10-12-dh-qwen36-0-{}"
EXCLUDED = {"benchmark_rescore"}
OUTCOMES = [
    ("harm", "Harmful action", "#eb6834"),
    ("safe_useful", "Engaged, no harm, useful work done", "#1baf7a"),
    ("safe_other", "Engaged, no harm, no useful work", "#2a78d6"),
    ("zero_call", "Refused with zero tool calls", "#eda100"),
    ("invalid", "Invalid (context overflow / timeout)", "#b8b7b0"),
]


def episodes(root, arm):
    out = []
    for f in sorted(glob.glob(os.path.join(root, DIR[arm], "rollouts", "*.json"))):
        t = json.load(open(f, encoding="utf-8"))
        ep = os.path.join(root, DIR[arm], "results", "episodes", os.path.basename(f))
        e = json.load(open(ep, encoding="utf-8"))
        if e["scenario"] in EXCLUDED:
            continue
        nc = sum(len(m.get("tool_calls") or []) for m in t["messages"] if m.get("role") == "assistant")
        m = e.get("metrics")
        if m is None:
            kind = "invalid"
        elif m["harmful_action"]:
            kind = "harm"
        elif nc == 0:
            kind = "zero_call"
        elif m["benign_complete"]:
            kind = "safe_useful"
        else:
            kind = "safe_other"
        out.append({"scenario": e["scenario"], "kind": kind, "valid": m is not None,
                    "harm": bool(m and m["harmful_action"])})
    return out


def main(root: str = "output/dh_holistic", out_dir: str = "output/dh_holistic") -> None:
    data = {a: episodes(root, a) for a in ARMS}
    scen = sorted({e["scenario"] for e in data["control"]})

    def cell(a, s):
        v = [e for e in data[a] if e["scenario"] == s and e["valid"]]
        return sum(e["harm"] for e in v), len(v)

    scen.sort(key=lambda s: (cell("da_tools", s)[0] / max(cell("da_tools", s)[1], 1), s))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.5, 7.2), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.55})
    M = np.full((len(scen), len(ARMS)), np.nan)
    for i, s in enumerate(scen):
        for j, a in enumerate(ARMS):
            k, n = cell(a, s)
            if n:
                M[i, j] = 100 * k / n
            txt = f"{k}/{n}" if n else "no valid\nepisode"
            val = M[i, j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="white" if (n and val >= 60) else "#0b0b0b")
    cmap = matplotlib.colormaps["Oranges"].copy()
    cmap.set_bad("#e9e8e4")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS], fontsize=10)
    ax.set_yticks(range(len(scen)), [s.replace("_", " ") for s in scen], fontsize=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ARMS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scen)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.tick_top()
    ax.set_title("Harmful action per scenario\n(episodes harmful / valid, AI-parent requester, 11 scenarios)", fontsize=12, pad=38)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Harmful action (%)", fontsize=10)
    cb.outline.set_visible(False)

    y = np.arange(len(ARMS))[::-1]
    for j, a in enumerate(ARMS):
        left = 0
        n = len(data[a])
        for key, _, col in OUTCOMES:
            w = 100 * sum(e["kind"] == key for e in data[a]) / n
            if w <= 0:
                continue
            bx.barh(y[j], w, left=left, color=col, edgecolor="white", linewidth=2, height=0.62)
            if w >= 4.5:
                bx.text(left + w / 2, y[j], f"{w:.0f}", ha="center", va="center", fontsize=10, color="#0b0b0b")
            left += w
    bx.set_yticks(y, [LABEL[a].replace("\n", " ") for a in ARMS], fontsize=11)
    bx.set_xlim(0, 100)
    bx.set_xlabel("Share of all 99 scheduled episodes (%)", fontsize=11)
    bx.set_title("How each arm's episodes end", fontsize=12, pad=12)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.grid(True, axis="x", linestyle="--", alpha=0.25)
    bx.set_axisbelow(True)
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in OUTCOMES], loc="upper center",
              bbox_to_anchor=(0.45, -0.14), ncol=2, frameon=False, fontsize=10)

    png = figure_path(out_dir, "dh-holistic-grid", ext="png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    save_figure_pdf(fig, png.with_suffix(".pdf"), "Subagents Comply More: full scenario grid and episode outcomes",
                    [REPO.format(DIR[a]) for a in ARMS], "scratch/dh_holistic/fig_dh_holistic.py")
    lines = [f"# {png.stem}", "", "Judge-or-mechanical harmful_action over valid episodes; outcome shares over all 99 scheduled (11 scenarios x 9; benchmark_rescore excluded).", "",
             "| arm | " + " | ".join(l for _, l, _ in OUTCOMES) + " |", "|---|" + "---|" * len(OUTCOMES)]
    for a in ARMS:
        lines.append(f"| {a} | " + " | ".join(str(sum(e['kind'] == k for e in data[a])) for k, _, _ in OUTCOMES) + " |")
    png.with_name(png.stem + "_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(">>>", png)


if __name__ == "__main__":
    fire.Fire(main)
