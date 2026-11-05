# ABOUTME: GPT-OSS-120B base / Control / DA15 in the paper's Fig. 2 style: ODCV, MASK, MMLU, SWE-bench
# ABOUTME: Lite and Arena-Hard, base as a dashed line, every number read from a pinned Hub run at render.
# Run: uv run python scratch/plot_gptoss_five_panel.py
"""Unites the old Figs 16-19 (scratch/capability_preservation/) into one figure styled like
scratch/plot_probes_four_remake.py: same arm colours, red dashed base line, bold value labels.

Error bars follow the rule at the top of scratch/figures/_hub.py:
  * ODCV: `item_se` over the 80 scenario x variant cells, 3 rollouts each.
  * Arena-Hard: the run's own paired prompt-cluster bootstrap (a paired comparison, the rule's
    one exception). Shown as Control / DA15 win rate against Base, so Base sits at 50%.
  * MASK, MMLU, SWE-bench Lite: each arm was run once, so there are no repeats and no bar.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scratch.figures import _hub

OUT = Path("output/plots")
ORG = _hub.ORG
BASE_NAME = "GPT-OSS-120B"
ARMS = ["control", "da"]
ARM = {"control": "+ Control (Standard SFT)", "da": "+ difficult advice 15% mix"}
COLOUR = {"control": "#5a6472", "da": "#7e22ce"}   # as in plot_probes_four_remake.py
BASE_COLOUR = "#c1121f"
KEY = {"base": "base", "control": "control", "da": "da15"}   # this script's arm -> the runs' arm id

# Pins: the same revisions the old Figs 16-19 read (scratch/capability_preservation/*.yaml).
ODCV = ("2026-10-12-gptoss120b-base-control-da15-odcv-full56k", "ee25be032bfe137b1f61a896104e9d4260e54eac")
MASK = {"base": ("2026-10-06-mask-gptoss120b", "081ecc21725aa8c83dd30bde61a644d5fa09f0c3"),
        "control": ("2026-10-15-mask-gptoss120b-0-plain", "df0addf0f7e40ae7109a41f575f1b0d628e9fd58"),
        "da": ("2026-10-06-mask-gptoss120b-0-da-15", "4c1cbff18562c07727bbb02410b34991cdceceea")}
MMLU = {"base": ("2026-10-14-mmlu-gptoss120b-base", "345a70aa66775007425328a9b7c31c09f13ef390"),
        "control": ("2026-10-14-mmlu-gptoss120b-0-plain", "862acf3443fb49172a8202637c6381421641fc65"),
        "da": ("2026-10-14-mmlu-gptoss120b-0-da-15", "36ea3662506b356c4d27bc360ffd2525106f426d")}
SWE = ("2026-10-13-gptoss120b-base-control-da15-swe-lite300-context65000", "4ccd7690774909418179133c1204a0a058f86cc8")
ARENA = ("2026-10-14-ah-luna-low-gptoss120b-three-pairs", "0ca407d879ac54516b1cc0c0a66a6b7dc79f0c6b")

PANELS = ["odcv", "mask", "mmlu", "swe", "arena"]
TITLE = {"odcv": "ODCV (agentic)", "mask": "MASK honesty (chat)", "mmlu": "MMLU",
         "swe": "SWE-bench Lite", "arena": "Arena-Hard (hard prompts)"}
YLABEL = {"odcv": "Misalignment rate (%), lower is better",
          "mask": "Honesty score (%), higher is better",
          "mmlu": "Accuracy (%), higher is better",
          "swe": "Tasks resolved (%), higher is better",
          "arena": "Style-controlled win rate vs base (%)"}

plt.rcParams.update({"font.size": 19, "axes.labelsize": 17, "xtick.labelsize": 17,
                     "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def _dl(repo: str, path: str, rev: str) -> dict:
    return json.load(open(hf_hub_download(f"{ORG}/{repo}", path, repo_type="dataset", revision=rev,
                                          token=os.environ["HF_TOKEN"]), encoding="utf-8"))


def read_all() -> tuple[dict, dict, list[tuple]]:
    """Every bar (mean, lo, hi), every base line, and a provenance row per number."""
    vals: dict[str, dict] = {p: {} for p in PANELS}
    base: dict[str, float] = {}
    rows: list[tuple] = []
    swe = _dl(SWE[0], "results/summary.json", SWE[1])
    arena = _dl(ARENA[0], "results/results.json", ARENA[1])["results"]
    for a in ["base", *ARMS]:
        k = KEY[a]
        odcv = _hub.item_se(_hub.odcv_cells(ODCV[0], f"results/{k}/scores_gemini-3-flash-preview.json", ODCV[1]))
        m = _dl(MASK[a][0], "results/results.json", MASK[a][1])
        mmlu = _dl(MMLU[a][0], "results/results.json", MMLU[a][1])
        s = swe[k]
        point = {"odcv": (odcv, f"{ODCV[0]} results/{k}/ (80 cells x 3 rollouts)"),
                 "mask": ((float(m["overall_honesty_score"]), None, None),
                          f"{MASK[a][0]} ({m.get('n_scored', m['n_rows'])} rows, one pass)"),
                 "mmlu": ((100 * mmlu["n_correct"] / mmlu["n"], None, None),
                          f"{MMLU[a][0]} ({mmlu['n_correct']}/{mmlu['n']}, one pass)"),
                 "swe": ((100 * s["success"] / s["graded"], None, None),
                         f"{SWE[0]} {k} ({s['success']}/{s['graded']}, one attempt)")}
        for p, (v, src) in point.items():
            if a == "base":
                base[p] = v[0]
            else:
                vals[p][a] = v
            rows.append((p, a, v, src))
        if a != "base":
            c = arena[f"{k}-vs-base"]["hard_prompt"]["controlled"]
            v = (100 * c["mean"], 100 * c["ci_lower"], 100 * c["ci_upper"])
            vals["arena"][a] = v
            rows.append(("arena", a, v, f"{ARENA[0]} {k}-vs-base (n={c['n']} prompts, both orders; "
                                       f"{c['n_bootstrap']}-round paired bootstrap)"))
    base["arena"] = 50.0
    rows.append(("arena", "base", (50.0, None, None), "parity: base judged against itself"))
    return vals, base, rows


def main() -> None:
    vals, base, rows = read_all()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(22.0, 7.8), gridspec_kw={"wspace": 0.42})
    for ax, p in zip(axes, PANELS):
        for i, a in enumerate(ARMS):
            m, lo, hi = vals[p][a]
            ax.bar(i, m, width=0.62, color=COLOUR[a], edgecolor="black", linewidth=0.8, zorder=3)
            if lo is not None:   # one repeat gives no bar
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523",
                            elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
            ax.text(i, (hi if hi is not None else m) + 1.4, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=16, fontweight="bold", zorder=5, clip_on=False)
        ax.axhline(base[p], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
        ax.set_title(TITLE[p], pad=12)
        ax.set_ylabel(YLABEL[p])
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, label=BASE_NAME)]
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in ARMS]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=18, columnspacing=1.6, handlelength=1.8,
               handleheight=1.0)
    fig.subplots_adjust(top=0.82, bottom=0.06)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{datetime.now():%Y-%m-%d}_gptoss_five_panel"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    md = [f"# {stem.name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')} by "
          "scratch/plot_gptoss_five_panel.py; every number read from a pinned Hub revision at render time.", "",
          "| Panel | Arm | Value | 95% CI | Source |", "|---|---|---|---|---|"]
    md += [f"| {p} | {a} | {v[0]:.1f} | " + (f"[{v[1]:.1f}, {v[2]:.1f}]" if v[1] is not None else "none (one pass)")
           + f" | {src} |" for p, a, v, src in rows]
    md += ["", "Arena-Hard judge: GPT-6 Luna, low reasoning effort, via ChatGPT subscription (Codex CLI); "
           "not leaderboard-equivalent. DA15 vs Control directly: 52.0% [48.5, 55.4] (hard, style-controlled).",
           "ODCV interval: `_hub.item_se` (per-cell SE, log-odds), not the run's published scenario-level interval."]
    Path(f"{stem}_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{stem}.png")


if __name__ == "__main__":
    main()
