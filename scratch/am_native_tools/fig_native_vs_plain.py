# ABOUTME: Figure: AM harmful rate, plain (text tools) vs native tools, one row per arm, one panel
# ABOUTME: per scenario + overall, Wilson 95% CIs, values printed. Reads the runs from the Hub.
# Run: uv run python scratch/am_native_tools/fig_native_vs_plain.py

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from src.utils import save_figure_pdf

RUNS = {  # arm -> (plain AM run, native-tools run); DA+tools native is not judged yet
    "control (plain)": ("dougalldeepmind/2026-10-08-am-qwen36-0-plain",
                        "dougalldeepmind/2026-11-23-am-native-tools-qwen36-0-plain"),
    "DA-15": ("dougalldeepmind/2026-10-08-am-qwen36-0-da-15",
              "dougalldeepmind/2026-11-23-am-native-tools-qwen36-0-da-15"),
    "DA+tools-15": ("dougalldeepmind/2026-10-13-am-qwen36-0-da-tools-15", None),
}
COLOR = {"control (plain)": "#8c8c8c", "DA-15": "#7b3fa0", "DA+tools-15": "#b07cc6"}
PANELS = ("overall", "blackmail", "leaking", "murder")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def counts(repo):
    s = json.load(open(hf_hub_download(repo, "results/results.json", repo_type="dataset")))
    s = s.get("summary", s)
    out = {"overall": (s["overall"]["harmful"], s["overall"]["n"])}
    out |= {k: (v["harmful"], v["n"]) for k, v in s["by_scenario"].items()}
    return out


data = {arm: tuple(counts(r) if r else None for r in pair) for arm, pair in RUNS.items()}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
arms = list(RUNS)
for ax, panel in zip(axes, PANELS):
    for y, arm in enumerate(arms):
        plain, native = data[arm]
        pts = []
        for kind, c, marker in (("text tools", plain, "o"), ("native tools", native, "D")):
            if c is None:
                continue
            k, n = c[panel]
            rate, (lo, hi) = 100 * k / n, wilson(k, n)
            dy = -0.13 if kind == "text tools" else 0.13
            ax.plot([lo, hi], [y + dy] * 2, color=COLOR[arm], lw=1.4, alpha=0.6)
            ax.plot(rate, y + dy, marker, color=COLOR[arm], ms=7,
                    mfc="white" if kind == "text tools" else COLOR[arm], mew=1.6)
            ax.text(hi + 1.5, y + dy, f"{rate:.1f}%", va="center", fontsize=8)
            pts.append(rate)
        if native is None:
            ax.text(25, y + 0.13, "native tools: not judged yet", va="center", fontsize=8, color="#555")
        if len(pts) == 2:
            ax.annotate("", xy=(pts[1], y + 0.13), xytext=(pts[0], y - 0.13),
                        arrowprops=dict(arrowstyle="->", color=COLOR[arm], lw=0.8, alpha=0.5))
    ax.set_title(panel if panel == "overall" else f"{panel} (n=60/arm)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("harmful trials (%)")
    ax.grid(axis="x", alpha=0.25)
axes[0].set_yticks(range(len(arms)), arms)
axes[0].set_ylim(len(arms) - 0.5, -0.5)
handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="text tools (plain AM)"),
           plt.Line2D([], [], marker="D", ls="", color="k", label="native tools (this run)")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
fig.suptitle("Agentic misalignment: tools in the prompt as text vs through the API (Qwen3.6-27B, "
             "180 trials/arm, Wilson 95% CI)", y=1.02, fontsize=11)
out = Path("output/figures/2026-11-23_am_native_tools_vs_text_tools.pdf")
out.parent.mkdir(parents=True, exist_ok=True)
arts = [r for pair in RUNS.values() for r in pair if r]
save_figure_pdf(fig, out, "AM native tools vs text tools", arts, __file__)
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(out, out.with_suffix(".png"))
