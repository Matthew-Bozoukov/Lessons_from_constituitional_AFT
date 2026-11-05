# ABOUTME: Fig. 13 remake: Qwen3.6-27B capabilities after SFT. MMLU bars (control / difficult advice 15%)
# ABOUTME: and the SWE-bench Lite difficult-advice share sweep as a line; Arena-Hard as a LaTeX tabular third panel.
# Run: uv run python scratch/figures/fig13_capabilities.py [--out_dir output/figures]
"""Two panels in the Fig. 4 house style, every number read from the Hub at render time, and the
LaTeX figure block (the PDF beside an Arena-Hard tabular: hard prompts, style-controlled, GPT-6 Luna judge;
GPT-4.1 kept in the results.md) written beside it as <stem>.tex.

MMLU   2026-10-14 matched screen: 57 subjects x 10 questions, one response each, thinking on. No interval:
       a subject's ten questions are different prompts, not repeats (the published subject-clustered
       non-inferiority test, docs/LOG.md 2026-10-14, is the formal comparison).
SWE    2026-10-07 Lite v5 sweep: 300 tasks, one rollout each, so by the rule there is no interval
       (the published Wilson band is a per-task binomial, not a repeat-based SE). 0% is the control.
Arena  GPT-4.1 (2026-10-12 DA vs control; 2026-10-13 both vs base) and GPT-6 Luna (2026-10-15, all three
       pairs); 500 hard prompts, both answer orders, length + Markdown control, prompt bootstrap CIs.
"""
from __future__ import annotations

import json
from datetime import datetime

import fire
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.naming import figure_path
from src.utils import save_figure_pdf
from scratch.figures import _hub

matplotlib.use("Agg")

BASE_NAME = "Qwen3.6-27B"
ARM = {"control": "+ Control (Standard SFT)", "da": "+ Difficult Advice mix"}
COLOUR = {"control": "#5a6472", "da": "#7e22ce"}
BASE_COLOUR = "#c1121f"

MMLU = {"base": "2026-10-14-mmlu-qwen36", "control": "2026-10-14-mmlu-qwen36-0-plain",
        "da": "2026-10-14-mmlu-qwen36-0-da-15"}
SWE = {"base": "2026-10-07-swebench-qwen36-lite-v5", 0: "2026-10-07-swebench-qwen36-0-plain-lite-v5",
       5: "2026-10-07-swebench-qwen36-0-da-5-lite-v5", 15: "2026-10-07-swebench-qwen36-0-da-15-lite-v5",
       25: "2026-10-07-swebench-qwen36-0-da-25-lite-v5"}
PCTS = [0, 5, 15, 25]
ARENA = {"gpt41_da_vs_control": "2026-10-12-ah-vs-qwen36-0-plain", "gpt41_vs_base": "2026-10-13-ah-vs-qwen36",
         "luna": "2026-10-15-ah-or-luna-low-qwen36-three-pairs"}

# Printed at ~0.57\textwidth (3.1 in), so the canvas is small and the strokes heavy: ~15pt type lands at ~8pt.
plt.rcParams.update({"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "axes.titlesize": 15, "legend.fontsize": 13,
                     "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4})


def mmlu(repo: str) -> float:
    """Accuracy over the 570 questions (57 subjects x 10). One response per question, so under the figure
    rule there is no interval: the ten questions of a subject are different prompts, not repeats."""
    path = hf_hub_download(f"{_hub.ORG}/{repo}", "rollouts/records.jsonl", repo_type="dataset", token=_hub._token())
    cells: dict[str, list[float]] = {}
    for line in open(path):
        r = json.loads(line)
        cells.setdefault(r["subject"], []).append(100.0 * bool(r["correct"]))
    assert len(cells) == 57 and {len(c) for c in cells.values()} == {10}, repo
    return float(np.mean([v for c in cells.values() for v in c]))


def swe(repo: str) -> float:
    r = _hub.fetch(repo, "results/results.json")
    assert r["status"] == "complete" and r["n_graded"] == 300, repo
    return 100.0 * r["pass_at_1"]


def arena() -> list[dict]:
    """Hard prompts, style-controlled score of the first-named model, per judge."""
    def flip(d):   # score of the other model in the pair
        return (1 - d["mean"], 1 - d["ci_upper"], 1 - d["ci_lower"])
    def take(d):
        return (d["mean"], d["ci_lower"], d["ci_upper"])
    g = _hub.fetch(ARENA["gpt41_da_vs_control"], "results/results.json")
    assert g["judge"] == "openai/gpt-4.1" and g["baseline"] == "qwen36_0_plain"
    da_vs_control = g["leaderboard"][0]["by_slice"]["hard_prompt"]["controlled"]
    b = _hub.fetch(ARENA["gpt41_vs_base"], "results/results.json")
    assert b["judge"] == "openai/gpt-4.1" and b["baseline"] == "qwen36"
    vs_base = {r["model_key"]: r["by_slice"]["hard_prompt"]["controlled"] for r in b["leaderboard"]}
    L = _hub.fetch(ARENA["luna"], "results/results.json")
    assert L["judge"] == "openai/gpt-6-luna"
    lu = {k: v["hard_prompt"]["controlled"] for k, v in L["results"].items()}
    return [
        {"first": "Control", "second": r"\DA{}", "gpt41": flip(da_vs_control), "luna": flip(lu["da15-vs-control"])},
        {"first": r"\DA{}", "second": "Base", "gpt41": take(vs_base["qwen36_0_da_15"]), "luna": take(lu["da15-vs-base"])},
        {"first": "Base", "second": "Control", "gpt41": flip(vs_base["qwen36_0_plain"]), "luna": flip(lu["control-vs-base"])},
    ]


def latex_figure(rows: list[dict]) -> str:
    """The whole figure block: the two-panel PDF beside the Arena-Hard tabular as the third panel."""
    def score(r):   # the control-vs-DA tie is the row the figure exists for, so it is set in bold
        s_ = f"{100 * r['luna'][0]:.1f}\\%"
        return f"\\textbf{{{s_}}}" if r["first"] == "Control" else s_
    body = "\n".join(f"        {r['first']} vs.\\ {r['second']} & {score(r)} & "
                     f"{100 * r['luna'][1]:.1f}--{100 * r['luna'][2]:.1f} \\\\" for r in rows)
    return "\n".join([
        r"\begin{figure}[tb]",
        r"  \centering",
        r"  \begin{minipage}[c]{0.57\textwidth}",
        r"    \centering",
        r"    \includegraphics[width=\linewidth]{figures/capabilities}",
        r"  \end{minipage}\hfill",
        r"  \begin{minipage}[c]{0.41\textwidth}",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \begin{tabular}{lrc}",
        r"      \toprule",
        r"      \multicolumn{3}{c}{Arena-Hard (GPT-6 Luna judge)} \\",
        r"      \midrule",
        r"      Comparison & Score & 95\% CI \\",
        r"      \midrule",
        body,
        r"      \bottomrule",
        r"    \end{tabular}",
        r"  \end{minipage}",
        r"  \caption{Capabilities of Qwen3.6-27B after SFT. Left: MMLU accuracy (57 subjects $\times$ 10 questions) and SWE-bench Lite tasks resolved (300 tasks), one response per item, against the \DA{} share of supervised tokens, 0\% being the control; the dashed line is the base model. Right: Arena-Hard style-controlled preference on 500 hard prompts with 95\% prompt-bootstrap CIs; above 50\% favours the first-named model. Neither MMLU nor SWE-bench Lite shows a capability cost; Arena-Hard favours the base model over both fine-tunes but rates them equally.}",
        r"  \label{fig:capabilities}",
        r"\end{figure}",
    ]) + "\n"


def main(out_dir: str = "output/figures") -> None:
    mm = {a: mmlu(MMLU[a]) for a in ("control", "da")}
    mm_base = mmlu(MMLU["base"])
    sw = {p: swe(SWE[p]) for p in PCTS}
    sw_base = swe(SWE["base"])
    rows = arena()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.3), gridspec_kw={"wspace": 0.45})
    for i, a in enumerate(("control", "da")):
        ax1.bar(i, mm[a], width=0.68, color=COLOUR[a], edgecolor="black", linewidth=1.2, zorder=3)
    ax1.axhline(mm_base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=2.4, zorder=2)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.65, 1.65)
    ax1.set_title("MMLU\n(higher is better)", pad=8)
    ax1.set_ylabel("Accuracy (%)")

    ys = [sw[p] for p in PCTS]
    ax2.plot(PCTS, ys, color=COLOUR["da"], linewidth=3.2, zorder=3)
    for p, y in zip(PCTS, ys):
        ax2.plot(p, y, marker="o", markersize=12, color=COLOUR["control" if p == 0 else "da"],
                 markeredgecolor="black", markeredgewidth=1.2, zorder=4)
    ax2.axhline(sw_base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=2.4, zorder=2)
    ax2.set_xticks(PCTS)
    ax2.set_xlim(-2, 27)
    ax2.set_xlabel("Difficult Advice\n(% of supervised tokens)")
    ax2.set_title("SWE-bench Lite\n(higher is better)", pad=8)
    ax2.set_ylabel("Tasks resolved (%)")

    for ax in (ax1, ax2):
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=1.2)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=2.4, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ("control", "da")]
    # The legend sits under the MMLU panel, in the space the SWE-bench x-label leaves empty on the left.
    fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.45)
    pos = ax1.get_position()
    fig.legend(handles=handles, frameon=False, loc="upper left", ncol=1, bbox_to_anchor=(pos.x0 - 0.02, pos.y0 - 0.03),
               fontsize=12.5, handlelength=1.5, handleheight=0.9, handletextpad=0.5, labelspacing=0.35)

    png = figure_path(out_dir, "fig13-capabilities", ext="png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    artifacts = [f"{_hub.ORG}/{r}" for r in list(MMLU.values()) + list(SWE.values())]
    save_figure_pdf(fig, png.with_suffix(".pdf"), "Fig. 13: MMLU and SWE-bench Lite after SFT", artifacts,
                    "scratch/figures/fig13_capabilities.py")
    tex = latex_figure(rows)
    png.with_name(png.stem + ".tex").write_text(tex, encoding="utf-8")

    md = [f"# {png.stem}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/figures/fig13_capabilities.py; every number read from the Hub at render time.",
          "MMLU (57 subjects x 10 questions) and SWE-bench Lite (300 tasks) are one response per item, so no interval. Arena: hard prompts, style-controlled, first-named model's score.", "",
          "| Panel | Arm | Value | 95% CI | Run |", "|---|---|---|---|---|"]
    for a in ("control", "da"):
        md.append(f"| mmlu | {ARM[a]} | {mm[a]:.1f} | none | {MMLU[a]} |")
    md.append(f"| mmlu | {BASE_NAME} (base line) | {mm_base:.1f} | | {MMLU['base']} |")
    for p in PCTS:
        md.append(f"| swe | {p}% difficult advice | {sw[p]:.1f} | none | {SWE[p]} |")
    md.append(f"| swe | {BASE_NAME} (base line) | {sw_base:.1f} | | {SWE['base']} |")
    md += ["", "| Arena comparison | GPT-4.1 | GPT-6 Luna | Runs |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['first']} vs {r['second']} | {100*r['gpt41'][0]:.1f} [{100*r['gpt41'][1]:.1f}, {100*r['gpt41'][2]:.1f}] "
                  f"| {100*r['luna'][0]:.1f} [{100*r['luna'][1]:.1f}, {100*r['luna'][2]:.1f}] | {', '.join(ARENA.values())} |")
    png.with_name(png.stem + "_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(tex)
    print(f">>> {png}")


if __name__ == "__main__":
    fire.Fire(main)
