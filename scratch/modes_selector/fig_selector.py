# ABOUTME: Two bar charts for the selector experiment: MASK honesty by context cue (DA vs DA+tools)
# ABOUTME: and Subagents Comply More harm rate by tool format, per arm. Reads the runs from the Hub.
# Run: uv run python scratch/modes_selector/fig_selector.py

import json
import math
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from src.utils import save_figure_pdf

ORG = "dougalldeepmind/"
GREY, DA, DAT = "#8c8c8c", "#7b3fa0", "#b07cc6"   # control, DA, DA+tools (as in scratch/am_native_tools)
INK, MUTED = "#222222", "#666666"

# condition -> {arm: [runs whose passes are pooled]}, weakest cue first
MASK = {
    "Plain MASK\n(nothing added)": {
        "Control": ["2026-11-03-mask-qwen36-0-plain"],
        "DA": ["2026-11-03-mask-qwen36-0-da-15"], "DA+tools": ["2026-11-09-mask-qwen36-0-da-tools-15"]},
    "User mentions tools\n(no tools)": {
        "Control": ["2026-11-27-mask-user-tools-mention-qwen36-0-plain"],
        "DA": ["2026-11-26-mask-user-tools-mention-qwen36-0-da-15"],
        "DA+tools": ["2026-11-26-mask-user-tools-mention-qwen36-0-da-tools-15"]},
    "Agent wording\n(no tools)": {
        "Control": ["2026-11-27-mask-agent-wording-qwen36-0-plain"],
        "DA": ["2026-11-26-mask-agent-wording-qwen36-0-da-15"],
        "DA+tools": ["2026-11-26-mask-agent-wording-qwen36-0-da-tools-15"]},
    "Tools listed\nin prose": {
        "Control": ["2026-11-27-mask-tools-prose-qwen36-0-plain"],
        "DA": ["2026-11-26-mask-tools-prose-qwen36-0-da-15", "2026-11-26-mask-tools-prose-rep-qwen36-0-da-15"],
        "DA+tools": ["2026-11-26-mask-tools-prose-qwen36-0-da-tools-15",
                     "2026-11-26-mask-tools-prose-rep-qwen36-0-da-tools-15"]},
    "Native tool\nblock": {
        "Control": ["2026-11-27-mask-tools-qwen36-0-plain"],
        "DA": ["2026-11-11-mask-tools-qwen36-0-da-15"], "DA+tools": ["2026-11-11-mask-tools-qwen36-0-da-tools-15"]},
}
# arm -> {format: run}
DH = {
    "Control": {"native, October": "2026-10-12-dh-qwen36-0-plain",
                "prose list": "2026-11-26-dh-prose-tools-qwen36-0-plain"},
    "DA": {"native, October": "2026-10-12-dh-qwen36-0-da-15",
           "prose list": "2026-11-26-dh-prose-tools-qwen36-0-da-15"},
    "DA+tools": {"native, October": "2026-10-12-dh-qwen36-0-da-tools-15",
                 "native, same day": "2026-11-26-dh-qwen36-0-da-tools-15",
                 "prose list": "2026-11-26-dh-prose-tools-qwen36-0-da-tools-15"},
}
FORMATS = ("native, October", "native, same day", "prose list")
used = []


def results(run):
    used.append(ORG + run)
    with open(hf_hub_download(ORG + run, "results/results.json", repo_type="dataset"), encoding="utf-8") as fh:
        return json.load(fh)


def mask_passes(runs):
    out = []
    for run in runs:
        r = results(run)
        pp = r.get("per_pass")
        pp = list(pp.values()) if isinstance(pp, dict) else pp
        out += [p["overall_honesty_score"] for p in pp] if pp else [r["overall_honesty_score"]]
    return out


def dh_counts(run):
    v = h = 0
    for scenario, d in results(run)["arms"]["subagent"]["per_scenario"].items():
        if scenario != "benchmark_rescore":
            v += d["valid"]
            h += round(d["valid"] * d["rates"]["harmful_action"])
    return h, v


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - half), 100 * min(1, c + half)


def style(ax, ylabel):
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=INK)


def save(fig, stem, title):
    out = Path(f"output/figures/{date.today().isoformat()}_{stem}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure_pdf(fig, out, title, used, __file__)
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(out, out.with_suffix(".png"))
    plt.close(fig)


# --- figure 1: MASK honesty by cue --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.8))
w = 0.27
for x, (cond, arms) in enumerate(MASK.items()):
    for dx, arm, color, hatch in ((-w - 0.02, "Control", GREY, None), (0, "DA", DA, None),
                                  (w + 0.02, "DA+tools", DAT, "///")):
        passes = mask_passes(arms[arm])
        mean = sum(passes) / len(passes)
        ax.bar(x + dx, mean, w, color=color, hatch=hatch, edgecolor="white", lw=0.8,
               label=arm if x == 0 else None)
        ax.plot([x + dx] * len(passes), passes, "o", ms=3.5, color=INK, alpha=0.75, zorder=3)
        ax.text(x + dx, max(passes) + 1.8, f"{mean:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(range(len(MASK)), list(MASK), fontsize=9.5)
style(ax, "MASK honesty score (higher is more honest)")
ax.set_ylim(0, 105)
ax.annotate("", xy=(4.45, -0.19), xytext=(-0.45, -0.19), xycoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
ax.text(2, -0.235, "stronger cue of agent deployment", transform=ax.get_xaxis_transform(), ha="center",
        fontsize=9, color=MUTED)
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
ax.set_title("DA+tools honesty rises with the strength of the agent cue; control rises far less\n"
             "MASK, 1,000 rows per pass, Qwen3.6-27B; each dot is one pass (1 to 3 per bar)", loc="left", fontsize=11, color=INK)
save(fig, "selector_mask_honesty_by_cue", "Selector: MASK honesty by context cue")

# --- figure 2: Subagents Comply More harm rate by tool format -----------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.8))
FILL = {"native, October": dict(alpha=1.0, hatch=None), "native, same day": dict(alpha=0.55, hatch=None),
        "prose list": dict(alpha=1.0, hatch="///")}
w = 0.26
for x, (arm, color) in enumerate(zip(DH, (GREY, DA, DAT))):
    present = [f for f in FORMATS if f in DH[arm]]
    for i, fmt in enumerate(present):
        k, n = dh_counts(DH[arm][fmt])
        rate, (lo, hi) = 100 * k / n, wilson(k, n)
        xx = x + (i - (len(present) - 1) / 2) * (w + 0.02)
        ax.bar(xx, rate, w, color=color, edgecolor="white", lw=0.8, **FILL[fmt])
        ax.plot([xx, xx], [lo, hi], color=INK, lw=1.1, zorder=3)
        ax.text(xx, hi + 1.5, f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(xx, -2.5, f"{k}/{n}", ha="center", va="top", fontsize=8, color=MUTED)
ax.set_xticks(range(len(DH)), list(DH), fontsize=11)
ax.tick_params(axis="x", pad=20)
style(ax, "Episodes with the harmful action (%)")
ax.set_ylim(0, 108)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#555555", edgecolor="white", **FILL[f]) for f in FORMATS]
ax.legend(handles, [f"tools as {f}" if f == "prose list" else f"tool block: {f}" for f in FORMATS],
          frameon=False, loc="upper right", fontsize=9.5)
ax.set_title("Writing the tools as a prose list leaves the DA+tools result where it was\n"
             "Subagents Comply More, 11 scenarios, Qwen3.6-27B; lines are Wilson 95% intervals",
             loc="left", fontsize=11, color=INK)
save(fig, "selector_subagents_harm_by_tool_format", "Selector: Subagents Comply More harm by tool format")
