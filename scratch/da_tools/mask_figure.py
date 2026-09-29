# ABOUTME: MASK honesty for difficult advice + tools vs plain difficult advice vs difficult agentic tasks, in the
# ABOUTME: paper draft's MASK-panel style (Figure 6: honesty score %, higher is better), +-1 binomial SE over rows.
# Run: uv run python scratch/da_tools/mask_figure.py [--tools <run>] [--dat <run>] [--out-dir output/figures]
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

from src.naming import figure_path  # noqa: E402

ORG = "dougalldeepmind"
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def arms(tools: str, dat: str) -> list[tuple[str, str, str]]:
    # (x label, fixed arm colour, MASK run). Colours: the paper's grey / purple / orange, and the
    # da-tools ochre (validated beside them in scratch/da_tools/paper_figures.py).
    return [
        ("No synthetic\n(0%)", "#5f6774", "2026-09-22-mask-qwen36-0-nosynth"),
        ("Difficult advice\n(15%)", "#7724c4", "2026-09-26-mask-qwen36-0-da-15"),
        ("Difficult advice\n+ tools (15%)", "#9c6500", tools),
        ("Difficult agentic\ntasks (7%)", "#eda100", dat),
    ]


def honesty(repo: str) -> dict:
    res = json.load(open(hf_hub_download(f"{ORG}/{repo}", "results/results.json", repo_type="dataset")))
    meta = json.load(open(hf_hub_download(f"{ORG}/{repo}", "metadata/run_meta.json", repo_type="dataset")))
    p, n = res["overall_honesty_score"] / 100, res["n_rows"]
    cfg = meta.get("config") or {}
    return {"score": 100 * p, "se": 100 * (p * (1 - p) / n) ** 0.5, "n": n, "target": res["target"],
            "protocol": (res["judge"], cfg.get("max_tokens"), cfg.get("temperature"), res.get("passes", 1)),
            "per_archetype": res.get("per_archetype")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default="2026-09-28-mask-qwen36-0-da-tools-15")
    # The paper's DAT arm (2026-09-08-qwen36-0-dat-7) was scored on MASK on 2026-09-09; reused, not re-run.
    ap.add_argument("--dat", default="2026-09-09-mask-qwen36-0-dat-7")
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    rows = [(lab, col, repo, honesty(repo)) for lab, col, repo in arms(a.tools, a.dat)]
    # Same judge and sampling temperature are required; a differing generation cap (the 2026-09-09
    # DAT run used 12,288 tokens, today's config 16,384) is allowed and written into the notes.
    base = rows[1][3]["protocol"]
    for _, _, repo, h in rows:
        assert h["protocol"][0] == base[0] and h["protocol"][2] == base[2], (repo, h["protocol"], base)
    notes = [f"- `{repo}`: max_tokens {h['protocol'][1]} (the others {base[1]})"
             for _, _, repo, h in rows if h["protocol"][1] != base[1]]
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.3, 2.8))
        for i, (lab, col, _, h) in enumerate(rows):
            ax.bar(i, h["score"], 0.66, color=col, zorder=3)
            eb = ax.errorbar(i, h["score"], yerr=h["se"], fmt="none", ecolor="#333", elinewidth=0.7,
                             capsize=1.8, capthick=0.7, zorder=4)
            for art in (*eb[1], *eb[2]):
                art.set_path_effects([withStroke(linewidth=1.35, foreground="white")])
            ax.text(i, h["score"] + h["se"] + 1.5, f"{h['score']:.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#111")
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Honesty score (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("MASK honesty\n(higher is better)", fontsize=8, fontweight="bold")
        fig.tight_layout(pad=0.3)
        p = figure_path(Path(a.out_dir), "mask_da_tools_vs_da_vs_dat", ext="pdf")
        fig.savefig(p)
        fig.savefig(p.with_suffix(".png"), dpi=300)
        plt.close(fig)
    md = [f"# {p.stem}", "", "MASK honesty score (%), one pass over 1,000 rows, judge "
          f"{rows[0][3]['protocol'][0]}; +-1 binomial SE over rows.", "",
          "| arm | honesty (%) | +-1 SE | target | MASK run |", "|---|---|---|---|---|"]
    md += [f"| {lab.replace(chr(10), ' ')} | {h['score']:.1f} | {h['se']:.1f} | `{h['target']}` | `{ORG}/{repo}` |"
           for lab, _, repo, h in rows]
    if notes:
        md += ["", "Protocol differences:", *notes]
    (p.parent / f"{p.stem}_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", p, "and", p.with_suffix(".png"))


if __name__ == "__main__":
    main()
