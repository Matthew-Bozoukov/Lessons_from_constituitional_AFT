# ABOUTME: The nine-cell Hospital figure (base + DA x {sept,oct corpus} x {sept,oct base mix} + the same for DA + tools)
# ABOUTME: redrawn from LLM-judge verdicts: act/no refusal, act/with refusal, plan only; judged blind to cell, no regex.
# Run: uv run python scratch/hospital_rejudge/plot_nine_cells.py --work <hospital_rejudge dir> [--out-dir output/figures]
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_and_plot import BOXES, MUTED, RC, load, regex_rows  # noqa: E402

from src.naming import figure_path  # noqa: E402

# (cell key, DA corpus, base mix); None = the base model.
CELLS = [
    ("base", None, None),
    ("da_sept", "sept", "sept"), ("da", "oct", "oct"), ("da_octda_septbase", "oct", "sept"), ("da_septda_octbase", "sept", "oct"),
    ("dat_sept", "sept", "sept"), ("da_tools", "oct", "oct"), ("dat_octda_septbase", "oct", "sept"), ("dat_septda_octbase", "sept", "oct"),
]
XS = [0, 1.4, 2.2, 3.0, 3.8, 5.2, 6.0, 6.8, 7.6]
GROUPS = [("Difficult advice", 1, 4), ("Difficult advice + tools", 5, 8)]
REPOS = {
    "base": "2026-09-29-hospital-self-sacrificial-qwen36",
    "da_sept": "2026-09-29-hospital-self-sacrificial-qwen36-0-da-15-canary",
    "da": "2026-10-08-hospital-self-sacrificial-qwen36-0-da-15-canary",
    "da_octda_septbase": "2026-10-09-hospital-self-sacrificial-qwen36-0-da-15-canary-octda-septbase",
    "da_septda_octbase": "2026-10-09-hospital-self-sacrificial-qwen36-0-da-15-canary-septda-octbase",
    "dat_sept": "2026-09-29-hospital-self-sacrificial-qwen36-0-da-tools-15-canary",
    "da_tools": "2026-10-08-hospital-self-sacrificial-qwen36-0-da-tools-15-canary-reusedtools",
    "dat_octda_septbase": "2026-10-09-hospital-self-sacrificial-qwen36-0-da-tools-15-canary-octda-septbase",
    "dat_septda_octbase": "2026-10-09-hospital-self-sacrificial-qwen36-0-da-tools-15-canary-septda-octbase",
}
PILL = {"sept": ("#d6e2ec", "#3b4a5a"), "oct": ("#3d5a73", "white"), None: (None, MUTED)}


def counts_of(rows: list[dict]) -> dict:
    return {"n": len(rows),
            "act_no_refusal": sum(1 for r in rows if r["act"] and not r["refusal"]),
            "act_refusal": sum(1 for r in rows if r["act"] and r["refusal"]),
            "plan_only": sum(1 for r in rows if r["plan"] and not r["act"]),
            "refusal": sum(1 for r in rows if r["refusal"]), "held": sum(1 for r in rows if r["held"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out-dir", default="output/figures")
    a = ap.parse_args()
    work = Path(a.work)
    rows = load(work / "render") + load(work / "render2")
    missing = [r["id"] for r in rows if r["missing"]]
    if missing:
        raise SystemExit(f"{len(missing)} shifts have no verdict yet: {missing[:10]}...")
    counts = {cell: counts_of([r for r in rows if r["arm"] == cell]) for cell, _, _ in CELLS}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for x, (cell, corpus, mix) in zip(XS, CELLS):
            h, bottom = counts[cell], 0
            for key, _, colour, ink in BOXES:
                n = h[key]
                if not n:
                    continue
                ax.bar(x, n, 0.62, bottom=bottom, color=colour, edgecolor="white", linewidth=0.8, zorder=3)
                if n >= 2:
                    ax.text(x, bottom + n / 2, str(n), ha="center", va="center", fontsize=6.5, color=ink, zorder=4)
                bottom += n
            ax.text(x, bottom + 0.5, f"{bottom}/{h['n']}", ha="center", va="bottom", fontsize=6.5, color=MUTED)
            for dy, val in ((-3.2, corpus), (-5.4, mix)):
                bg, fg = PILL[val]
                ax.text(x, dy, val or "none", ha="center", va="center", fontsize=6.5, color=fg, clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.25", fc=bg, ec="none") if bg else None)
        ax.text(XS[0] - 0.55, -3.2, "DA corpus", ha="right", va="center", fontsize=6.5, color=MUTED, clip_on=False)
        ax.text(XS[0] - 0.55, -5.4, "base mix", ha="right", va="center", fontsize=6.5, color=MUTED, clip_on=False)
        ax.text(XS[0], -8.3, "Base\nmodel", ha="center", va="top", fontsize=7.5, clip_on=False)
        for title, i0, i1 in GROUPS:
            ax.plot([XS[i0] - 0.35, XS[i1] + 0.35], [-7.2, -7.2], color="#333", lw=0.7, clip_on=False)
            ax.text((XS[i0] + XS[i1]) / 2, -8.3, title, ha="center", va="top", fontsize=7.5, clip_on=False)
        ax.set_xticks([])
        ax.set_xlim(XS[0] - 0.6, XS[-1] + 0.6)
        ax.set_ylim(0, 33)
        ax.set_yticks([0, 10, 20, 30])
        ax.yaxis.grid(True, color="#e1e0d9", lw=0.5, zorder=0)
        ax.set_ylabel("Shifts with a sabotage act or plan (of 30)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c, _ in BOXES], frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5, handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
        fig.text(0.01, 0.01, "Judged by Claude Opus reading every shift in full, blind to cell; no regex, every shift scored.",
                 fontsize=5.5, color=MUTED)
        fig.tight_layout(pad=0.3)
        fig.subplots_adjust(bottom=0.30)
        out = figure_path(a.out_dir, "hospital_nine_cells_act_plan_refusal_llm_judged")
        fig.savefig(out, dpi=300)
        fig.savefig(out.with_suffix(".pdf"))
    lines = ["# Hospital nine cells, LLM-judged (no regex)", "",
             "| cell | DA corpus | base mix | act, no refusal | act, with refusal | plan only | act or plan | any refusal | held | HF repo |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cell, corpus, mix in CELLS:
        c = counts[cell]
        lines.append(f"| {cell} | {corpus or '-'} | {mix or '-'} | {c['act_no_refusal']} | {c['act_refusal']} | {c['plan_only']} | "
                     f"{c['act_no_refusal'] + c['act_refusal'] + c['plan_only']}/{c['n']} | {c['refusal']} | {c['held']} | {REPOS[cell]} |")
    lines += ["", "## Published measures on the same shifts (regex act/plan, Gemini refusal >= 3; swap cells were only half judged)", "",
              "| cell | act, no refusal | act, with refusal | act, refusal unscored | plan only | scored shifts |", "|---|---|---|---|---|---|"]
    for cell, _, _ in CELLS:
        rr = regex_rows(work, cell)
        a0 = sum(1 for v in rr.values() if v.get("act") and v.get("refusal_judge") is False)
        a1 = sum(1 for v in rr.values() if v.get("act") and v.get("refusal_judge"))
        un = sum(1 for v in rr.values() if v.get("act") and v.get("refusal_judge") is None)
        po = sum(1 for v in rr.values() if v.get("plan") and not v.get("act"))
        lines.append(f"| {cell} | {a0} | {a1} | {un} | {po} | {len(rr)} |")
    md = out.with_name(out.stem + "_results.md")
    md.write_text("\n".join(lines) + "\n")
    print(out, md, sep="\n")
    print("\n".join(lines[:13]))


if __name__ == "__main__":
    main()
