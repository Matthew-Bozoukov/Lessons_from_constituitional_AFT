# ABOUTME: Appendix figure candidates: how much each arm DELIBERATES in its reasoning and whether it reaches for
# ABOUTME: the constitution while doing so, in the training corpus, on MASK pressure turns and in ODCV rollouts.
# Run: uv run python scratch/plot_deliberation_and_citation.py --design {1,2,3}
"""Three designs over one dataset; Jamie picks one for the appendix.

  1 ATTRITION   where the constitution goes: citation per setting, deliberation drawn over it.
  2 COMPOSITION every turn assigned to one of three states (deliberates+cites / deliberates only /
                neither), so "deliberation is flat, citation is not" is readable off one bar each.
  3 MECHANISM   citation against the behavioural outcome it is supposed to explain (MASK honesty,
                ODCV misalignment), with the standard-SFT control as the no-constitution anchor.

Labels (all google/gemini-3-flash-preview, temperature 0; 1,000 traces per arm per setting, drawn
WITHIN arm so no arm gets a different n):
  deliberation  weighs courses of action -- rubric blind to the constitution, so this axis and the
                citation axis are independent. Hand-validated on 100 blind traces: 97% agreement,
                kappa 0.94, against 88% for a length-only cutoff.
  citation      `traits`/trait_specific: states a specific principle by number, title or paraphrase.
MASK is PRESSURE turns only: a row has one pressured response and up to five belief probes, so a
pooled rate is ~80% belief and reads four times too low (23% against 73% on the 2026-10-09 sample).
ODCV rates are ALSO reported per rollout, because arms emit different numbers of turns (12.1 for
delib-15 against 9.4) and a per-turn rate silently divides by that.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))

from src.naming import figure_path  # noqa: E402

OUT = REPO / "output/figures"
ARMS = ["delib-15", "delib-sonnet-15", "da-15", "da-qwen-15"]
NICE = {"delib-15": "delib\n(Qwen)", "delib-sonnet-15": "delib\n(Sonnet)",
        "da-15": "da\n(Sonnet)", "da-qwen-15": "da\n(Qwen)"}
# CLAUDE.md fixes the family hues: delib blue, da purple. Variants are tints of their family.
COLOUR = {"delib-15": "#2a78d6", "delib-sonnet-15": "#86b6ef", "da-15": "#4a3aa7", "da-qwen-15": "#9085e9"}
INK, INK2, GRID = "#111827", "#52514e", "#E5E7EB"

LAB = REPO / "output/odcv"
DELIB = {"mask": LAB / "2026-10-09_delib_mask_plain15/labels.jsonl",
         "odcv": LAB / "2026-10-09_delib_odcv_plain15/labels.jsonl"}
CITE = {"training": LAB / "2026-10-09_traitref_training_plain15/labels.jsonl",
        "mask": LAB / "2026-10-09_traitref_pressure_plain15/labels.jsonl",
        "odcv": LAB / "2026-10-09_traitref_odcv1k_plain15/labels.jsonl"}
# The second citation scheme: the constitution invoked as a DOCUMENT, not a principle restated.
REF = {"training": LAB / "2026-10-09_constref_training_plain15/labels.jsonl",
       "mask": LAB / "2026-10-09_constref_pressure_plain15/labels.jsonl",
       "odcv": LAB / "2026-10-09_constref_odcv_plain15/labels.jsonl"}
DELIB["training"] = LAB / "2026-10-09_delib_training_plain15/labels.jsonl"
TURNS_PER_ROLLOUT = {"da-15": 10.28, "da-qwen-15": 9.37, "delib-15": 12.11, "delib-sonnet-15": 9.39}
# The behavioural outcomes these traces come from (scratch/plot_delib_2x2_plain.py).
OUTCOME = {"delib-15": dict(mask=71.5, odcv=30.0), "delib-sonnet-15": dict(mask=67.7, odcv=14.2),
           "da-15": dict(mask=83.1, odcv=13.3), "da-qwen-15": dict(mask=84.5, odcv=22.1)}
CONTROL = dict(mask=43.1, odcv=45.0)   # standard SFT on the plain base, never saw a constitution


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def load(path: Path) -> dict:
    return {(r["arm"], r["unit"]): r for r in (json.loads(line) for line in path.open())}


def rate(labels: dict, arm: str, positive: str, among=None, turn: str | None = None) -> tuple[float, float, float, int, int]:
    """`turn` MUST be given for MASK: a row has one pressured response and up to five belief probes, so an
    unfiltered rate is ~80% belief and reads several times too low."""
    keys = [k for k in labels if k[0] == arm and labels[k]["label"] != "unparsed"
            and (among is None or k in among) and (turn is None or labels[k].get("turn") == turn)]
    k = sum(1 for key in keys if labels[key]["label"] == positive)
    return (*wilson(k, len(keys)), k, len(keys))


def frame(ax, ylabel: str = "", ylim=(0, 105)) -> None:
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#374151", length=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, subject: str, notes: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = figure_path(OUT, subject, ext="png")
    fig.savefig(png)
    fig.savefig(figure_path(OUT, subject, ext="pdf"))
    Path(str(png)[:-4] + "_results.md").write_text("\n".join(notes) + "\n")
    print(f"wrote {png}")


def design1() -> None:
    """ATTRITION: citation per setting, with deliberation drawn over it as a marker."""
    cite = {s: load(p) for s, p in CITE.items()}
    delib = {s: load(p) for s, p in DELIB.items()}
    panels = [("training", "Training corpus", None, None), ("mask", "MASK (pressure turns)", "mask", "pressure"),
              ("odcv", "ODCV (agentic turns)", "odcv", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3), dpi=300, sharey=True)
    rows = []
    for ax, (key, title, dkey, turn) in zip(axes, panels):
        for i, arm in enumerate(ARMS):
            p, lo, hi, k, n = rate(cite[key], arm, "trait_specific", turn=turn)
            ax.bar(i, p, 0.62, color=COLOUR[arm], edgecolor="white", linewidth=0.6, zorder=3)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, elinewidth=0.9,
                        capsize=2.4, capthick=0.9, zorder=4)
            ax.annotate(f"{p:.0f}", (i, hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
            rows.append((title, arm, "states a principle", k, n, p, lo, hi))
            if dkey:
                dp, dlo, dhi, dk, dn = rate(delib[dkey], arm, "deliberation", turn=turn)
                ax.plot([i - 0.31, i + 0.31], [dp, dp], color=INK, linewidth=1.6, zorder=6)
                ax.plot([i], [dp], marker="D", ms=4.5, color="white", mec=INK, mew=1.3, zorder=7)
                ax.annotate(f"{dp:.0f}", (i, dp), xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=INK2)
                rows.append((title, arm, "deliberates", dk, dn, dp, dlo, dhi))
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([NICE[a] for a in ARMS], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", color="#374151")
        frame(ax)
    axes[0].set_ylabel("% of reasoning traces")
    axes[0].annotate("no deliberation\nmeasure: these are\ngeneration-time traces", (1.5, 60),
                     ha="center", fontsize=7, color=INK2)
    handles = [Patch(facecolor="#9CA3AF", label="states a specific principle of the constitution"),
               Line2D([0], [0], color=INK, marker="D", ms=4.5, mfc="white", mec=INK, mew=1.3,
                      linewidth=1.6, label="deliberates (weighs courses of action)")]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    notes = ["# Design 1: attrition of the constitution from training to deployment", "",
             "Bars: share of traces stating a specific principle. Diamonds: share that deliberate at all",
             "(rubric blind to the constitution). 1,000 traces per arm per setting, drawn within arm.", "",
             "| setting | arm | measure | k | n | % | lo | hi |", "|---|---|---|---|---|---|---|---|"]
    notes += [f"| {s} | {a} | {m} | {k} | {n} | {p:.1f} | {lo:.1f} | {hi:.1f} |" for s, a, m, k, n, p, lo, hi in rows]
    save(fig, "deliberation_citation_attrition", notes)


def matched(ev: str, turn: str | None = None) -> dict:
    """{(arm, unit): (deliberates, cites)} for every trace carrying BOTH labels."""
    D, C = load(DELIB[ev]), load(CITE[ev])
    out = {}
    for k in set(D) & set(C):
        if D[k]["label"] == "unparsed" or C[k]["label"] == "unparsed":
            continue
        if turn and D[k].get("turn") != turn:
            continue
        out[k] = (D[k]["label"] == "deliberation", C[k]["label"] == "trait_specific")
    return out


def design2() -> None:
    """COMPOSITION: every turn in one of three states, as a 100% stacked bar per arm."""
    panels = [("mask", "pressure", "MASK (pressure turns)"), ("odcv", None, "ODCV (agentic turns)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=300, sharey=True)
    rows = []
    for ax, (ev, turn, title) in zip(axes, panels):
        m = matched(ev, turn)
        for i, arm in enumerate(ARMS):
            g = [v for k, v in m.items() if k[0] == arm]
            n = len(g)
            both = sum(1 for d, c in g if d and c)
            donly = sum(1 for d, c in g if d and not c)
            cites_only = sum(1 for d, c in g if (not d) and c)
            neither = n - both - donly - cites_only
            segs = [(both, COLOUR[arm], None), (donly, COLOUR[arm], "...."),
                    (cites_only, "#9CA3AF", "xx"), (neither, "#E8E8E6", None)]
            bottom = 0.0
            for v, col, hatch in segs:
                if not v:
                    continue
                h = 100 * v / n
                ax.bar(i, h, 0.66, bottom=bottom, color=col, hatch=hatch, edgecolor="white",
                       linewidth=0.8, zorder=3)
                if h >= 6:
                    ax.annotate(f"{h:.0f}", (i, bottom + h / 2), ha="center", va="center", fontsize=8,
                                color="white" if col != "#E8E8E6" else INK2, zorder=5)
                bottom += h
            ax.annotate(f"n={n}", (i, 101), ha="center", va="bottom", fontsize=7, color=INK2)
            rows.append((title, arm, n, both, donly, cites_only, neither))
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([NICE[a] for a in ARMS], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left", color="#374151")
        frame(ax, ylim=(0, 108))
    axes[0].set_ylabel("% of reasoning turns")
    handles = [Patch(facecolor="#6b6b6b", label="deliberates AND states a principle"),
               Patch(facecolor="#6b6b6b", hatch="....", edgecolor="white", label="deliberates, no principle"),
               Patch(facecolor="#9CA3AF", hatch="xx", edgecolor="white", label="states a principle without deliberating"),
               Patch(facecolor="#E8E8E6", label="neither")]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               fontsize=8, handlelength=1.5, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    notes = ["# Design 2: what a reasoning turn is made of", "",
             "Each bar is 100% of that arm's turns, split by the two independent labels. Colour is the arm;",
             "the dotted segment deliberates without reaching for the constitution.", "",
             "| setting | arm | n | delib+cite | delib only | cite only | neither |", "|---|---|---|---|---|---|---|"]
    notes += [f"| {s} | {a} | {n} | {b} | {d} | {c} | {x} |" for s, a, n, b, d, c, x in rows]
    save(fig, "deliberation_citation_composition", notes)


def design3() -> None:
    """MECHANISM vs OUTCOME: citation rate against the behaviour it is meant to explain."""
    panels = [("mask", "pressure", "MASK", "honesty score", (40, 90), False),
              ("odcv", None, "ODCV", "misalignment rate (%)", (0, 50), True)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=300)
    rows = []
    for ax, (ev, turn, title, ylab, ylim, invert_good) in zip(axes, panels):
        m = matched(ev, turn)
        for arm in ARMS:
            g = [v for k, v in m.items() if k[0] == arm]
            c = sum(1 for d, cc in g if cc)
            p, lo, hi = wilson(c, len(g))
            y = OUTCOME[arm][ev]
            ax.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", ms=8, color=COLOUR[arm],
                        ecolor=COLOUR[arm], elinewidth=1.2, capsize=2.5, zorder=4)
            ax.annotate(NICE[arm].replace("\n", " "), (p, y), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
            rows.append((title, arm, c, len(g), p, y))
        ax.axhline(CONTROL[ev], color="#4B5563", linestyle=(0, (1.5, 2.5)), linewidth=1.0, zorder=2)
        ax.annotate(f"standard SFT control: {CONTROL[ev]:.0f}\n(never saw a constitution)",
                    (ax.get_xlim()[1], CONTROL[ev]), xytext=(-4, 5 if invert_good else -16),
                    textcoords="offset points", ha="right", fontsize=7.5, color="#4B5563")
        ax.set_xlabel("% of turns stating a specific principle")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5, loc="left", color="#374151")
        frame(ax, ylim=ylim)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    fig.suptitle("Does reaching for the constitution track behaving better?  (four arms, one seed each)",
                 fontsize=9.5, color="#374151", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    notes = ["# Design 3: mechanism against outcome", "",
             "x: share of turns stating a specific principle (Wilson 95%). y: the arm's behavioural result.",
             "Four points per panel and one training seed each, so this is descriptive, NOT causal: it shows",
             "whether citation and behaviour move together, and in ODCV nothing cites, so there is no x spread",
             "to relate anything to.", "",
             "| setting | arm | cites | n | citation % | outcome |", "|---|---|---|---|---|---|"]
    notes += [f"| {s} | {a} | {c} | {n} | {p:.1f} | {y} |" for s, a, c, n, p, y in rows]
    save(fig, "deliberation_citation_mechanism", notes)



def design4() -> None:
    """The 2026-10-09 three-panel figure, restricted to the turns where the question is live:
    ODCV DELIBERATIVE turns and MASK PRESSURE turns. Training is every synthetic row."""
    delib = {k: load(v) for k, v in DELIB.items()}
    panels = [("training", "Training corpus (synthetic rows)", None),
              ("odcv", "ODCV (deliberative turns)", "delib"),
              ("mask", "MASK (pressure turns)", "pressure")]
    metrics = [("traits", CITE, "trait_specific", "states a specific principle (number, title or paraphrase)"),
               ("reference", REF, "substantive", "names the constitution as a document")]
    # Paper sizing: the figure is printed at a 5.5in text width, so a 9.2in canvas scales by 0.6 and
    # these point sizes land at 7-8pt on the page (the caption is 9pt).
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), dpi=300, sharey=True)
    rows, width = [], 0.38
    for ax, (ev, title, restrict) in zip(axes, panels):
        keep = None
        if restrict == "delib":
            keep = {k for k, v in delib[ev].items() if v["label"] == "deliberation"}
        for i, arm in enumerate(ARMS):
            for j, (name, src, pos, _) in enumerate(metrics):
                lab = load(src[ev])
                turn = "pressure" if restrict == "pressure" else None
                p_, lo, hi, k, n = rate(lab, arm, pos, among=keep, turn=turn)
                x = i + (j - 0.5) * width
                style = dict(color=COLOUR[arm]) if j == 0 else \
                    dict(facecolor="white", edgecolor=COLOUR[arm], linewidth=1.5, hatch="///")
                ax.bar(x, p_, width, zorder=3, **style)
                ax.errorbar(x, p_, yerr=[[p_ - lo], [hi - p_]], fmt="none", ecolor=INK,
                            elinewidth=0.9, capsize=2.2, zorder=4)
                ax.annotate(f"{p_:.0f}", (x, hi), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=10, color=INK)
                rows.append((title, arm, name, k, n, p_, lo, hi))
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([NICE[a] for a in ARMS], fontsize=10.5)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.set_title(title, fontsize=11.5, loc="left", color="#374151")
        frame(ax)
    axes[0].set_ylabel("% of reasoning traces", fontsize=12)
    handles = [Patch(facecolor="#6b6b6b", label=metrics[0][3]),
               Patch(facecolor="white", edgecolor="#6b6b6b", linewidth=1.5, hatch="///", label=metrics[1][3])]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               fontsize=11, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    notes = ["# Design 4: constitution reference where the question is live", "",
             "ODCV is restricted to turns the deliberation judge marked as weighing courses of action;",
             "MASK to the pressured response (the turn the honesty score is computed from).", "",
             "| setting | arm | scheme | k | n | % | lo | hi |", "|---|---|---|---|---|---|---|---|"]
    notes += [f"| {s} | {a} | {m} | {k} | {n} | {p:.1f} | {lo:.1f} | {hi:.1f} |" for s, a, m, k, n, p, lo, hi in rows]
    save(fig, "constitution_reference_live_turns", notes)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=int, choices=[1, 2, 3, 4], default=0)
    a = ap.parse_args()
    for d, fn in ((1, design1), (2, design2), (3, design3), (4, design4)):
        if a.design in (0, d):
            fn()




if __name__ == "__main__":
    main()
