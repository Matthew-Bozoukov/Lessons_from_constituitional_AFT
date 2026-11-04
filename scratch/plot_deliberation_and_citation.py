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
from src.utils import save_figure_pdf  # noqa: E402
from scratch.figures import _hub  # noqa: E402

OUT = REPO / "output/figures"
ARMS = ["delib-sonnet-15", "delib-15", "da-15", "da-qwen-15"]   # Sonnet teacher left of Qwen in each family
NICE = {"delib-15": "delib\n(Qwen)", "delib-sonnet-15": "delib\n(Sonnet)",
        "da-15": "da\n(Sonnet)", "da-qwen-15": "da\n(Qwen)"}
FULL = {"delib-15": "Deliberative alignment, Qwen teacher", "delib-sonnet-15": "Deliberative alignment, Sonnet teacher",
        "da-15": "Difficult advice, Sonnet teacher", "da-qwen-15": "Difficult advice, Qwen teacher"}
# CLAUDE.md fixes the family hues: delib blue, da purple. Variants are tints of their family.
# Within each family the Sonnet teacher is the dark shade and the Qwen teacher the light one, far
# apart in lightness so the four read as four, not two.
COLOUR = {"delib-sonnet-15": "#1d4ed8", "delib-15": "#93c5fd", "da-15": "#581c87", "da-qwen-15": "#c084fc"}
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
# 2026-11-04: agentic-misalignment rollouts (every trial, 18 conditions x 10, one trace each), the three
# plain15 arms in one file and the prompt-matched delib-sonnet-15-a (NEW below) in another.
# 2026-11-04: the MASK pressure turns of ALL THREE passes of every arm, so a row has three repeats and the
# figure rule can put an interval on the MASK panel (the 10-09 files hold one pass).
CITE["mask"] = LAB / "2026-11-04_traitref_pressure3_plain15/labels.jsonl"
REF["mask"] = LAB / "2026-11-04_constref_pressure3_plain15/labels.jsonl"
DELIB["am"] = LAB / "2026-11-04_delib_am_plain15/labels.jsonl"
CITE["am"] = LAB / "2026-11-04_traitref_am_plain15/labels.jsonl"
REF["am"] = LAB / "2026-11-04_constref_am_plain15/labels.jsonl"
# 2026-11-04: the delib (Sonnet) arm is the prompt-matched 2026-10-30-qwen36-0-delib-sonnet-15-a
# (trait_ref_arms.py `plain15a`), labelled on its own under the same schemes, samplers and seed. Each
# 2026-10-09 file maps to the file holding that arm's labels; `load` swaps them in under the old key.
REPLACED = {"delib-sonnet-15": "delib-sonnet-15-a"}
NEW = {
    DELIB["training"]: LAB / "2026-11-04_delib_training_plain15a/labels.jsonl",
    DELIB["mask"]: LAB / "2026-11-04_delib_pressure_plain15a/labels.jsonl",
    DELIB["odcv"]: LAB / "2026-11-04_delib_odcv_plain15a/labels.jsonl",
    CITE["training"]: LAB / "2026-11-04_traitref_training_plain15a/labels.jsonl",
    CITE["mask"]: LAB / "2026-11-04_traitref_pressure3_plain15a/labels.jsonl",
    CITE["odcv"]: LAB / "2026-11-04_traitref_odcv_plain15a/labels.jsonl",
    REF["training"]: LAB / "2026-11-04_constref_training_plain15a/labels.jsonl",
    REF["mask"]: LAB / "2026-11-04_constref_pressure3_plain15a/labels.jsonl",
    REF["odcv"]: LAB / "2026-11-04_constref_odcv_plain15a/labels.jsonl",
    DELIB["am"]: LAB / "2026-11-04_delib_am_plain15a/labels.jsonl",
    CITE["am"]: LAB / "2026-11-04_traitref_am_plain15a/labels.jsonl",
    REF["am"]: LAB / "2026-11-04_constref_am_plain15a/labels.jsonl",
}
TURNS_PER_ROLLOUT = {"da-15": 10.28, "da-qwen-15": 9.37, "delib-15": 12.11, "delib-sonnet-15": 9.39}
# The behavioural outcomes these traces come from (scratch/figures/fig8_teacher_2x2.py, 2026-11-04).
OUTCOME = {"delib-15": dict(mask=71.5, odcv=30.0), "delib-sonnet-15": dict(mask=77.1, odcv=11.7),
           "da-15": dict(mask=83.2, odcv=13.3), "da-qwen-15": dict(mask=84.5, odcv=22.1)}
CONTROL = dict(mask=43.2, odcv=45.0)   # standard SFT on the plain base, never saw a constitution


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def load(path: Path) -> dict:
    """{(arm, unit): record}, with every REPLACED arm's records taken from the NEW file instead."""
    out = {(r["arm"], r["unit"]): r for r in (json.loads(line) for line in path.open())
           if r["arm"] not in REPLACED}
    new = NEW[path]
    assert new.is_file(), f"{new}: the replacement arm's labels are missing (scratch/classify_constitution_refs.py --arm-set plain15a)"
    for old, fresh in REPLACED.items():
        n = 0
        for line in new.open():
            r = json.loads(line)
            if r["arm"] == fresh:
                out[(old, r["unit"])] = {**r, "arm": old}
                n += 1
        assert n, f"{new}: no records for {fresh}"
    return out


def rate(labels: dict, arm: str, positive: str, among=None, turn: str | None = None) -> tuple[float, float, float, int, int]:
    """`turn` MUST be given for MASK: a row has one pressured response and up to five belief probes, so an
    unfiltered rate is ~80% belief and reads several times too low."""
    keys = [k for k in labels if k[0] == arm and labels[k]["label"] != "unparsed"
            and (among is None or k in among) and (turn is None or labels[k].get("turn") == turn)]
    k = sum(1 for key in keys if labels[key]["label"] == positive)
    return (*wilson(k, len(keys)), k, len(keys))


# The item each eval's traces belong to, for the figure error-bar rule (scratch/figures/_hub.py): a bar is
# the mean over items with a 95% interval from each item's own repeats. ODCV: the cell (variant/scenario),
# its repeats the sampled turns across the three passes. Agentic: the condition, its repeats the ten
# trials. MASK: the row, which has ONE labelled pass here, so no interval is estimable and none is drawn.
# Training: every synthetic row is labelled, so the rate is exact and carries no interval.
ITEM_OF = {"odcv": lambda u: "/".join(u.split("/")[:2]), "am": lambda u: u.split("/")[0],
           "mask": lambda u: "/".join(u.split("/")[:2]), "training": lambda u: u}


def item_rate(labels: dict, ev: str, arm: str, positive: str, among=None, turn: str | None = None):
    """(rate, lo, hi, k, n): mean over items of the per-item rate, interval per the figure rule (None
    when no item has a repeat), with the raw positive and labelled counts for the table."""
    keys = [k for k in labels if k[0] == arm and labels[k]["label"] != "unparsed"
            and (among is None or k in among) and (turn is None or labels[k].get("turn") == turn)]
    cells: dict[str, list[float]] = {}
    for key in keys:
        cells.setdefault(ITEM_OF[ev](key[1]), []).append(100.0 * (labels[key]["label"] == positive))
    m, lo, hi = _hub.item_se(list(cells.values()))
    k = sum(1 for key in keys if labels[key]["label"] == positive)
    return m, lo, hi, k, len(keys)


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
    # The Hub artifacts the labels were read from: each arm's mixture (training rows), ODCV, MASK and
    # agentic runs, for the three plain15 arms drawn plus the prompt-matched delib-sonnet-15-a.
    from trait_ref_arms import ARM_SETS  # noqa: E402
    specs = [ARM_SETS["plain15"]["arms"][a] for a in ("delib-15", "da-15", "da-qwen-15")] + \
            [ARM_SETS["plain15a"]["arms"]["delib-sonnet-15-a"]]
    artifacts = [r for sp in specs for key in ("mix", "odcv", "mask", "am")
                 for r in (sp[key] if isinstance(sp[key], list) else [sp[key]])]
    save_figure_pdf(fig, figure_path(OUT, subject, ext="pdf"),
                    f"Constitution reference in reasoning traces ({subject})", artifacts,
                    "scratch/plot_deliberation_and_citation.py")
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
    ODCV DELIBERATIVE turns and MASK PRESSURE turns. Training is every synthetic row. Agentic
    misalignment is single-turn, so every rollout is live and none is filtered. Intervals follow the
    figure rule (item_rate): per-item SE from each item's repeats, so training and MASK carry none."""
    delib = {k: load(v) for k, v in DELIB.items()}
    panels = [("training", "Training corpus (synthetic rows)", None),
              ("odcv", "ODCV (deliberative turns)", "delib"),
              ("am", "Agentic misalignment", None),
              ("mask", "MASK (pressure turns)", "pressure")]
    metrics = [("traits", CITE, "trait_specific", "states a specific principle (number, title or paraphrase)"),
               ("reference", REF, "substantive", "names the constitution as a document")]
    # Type sizes as in the other main-body remakes (scratch/plot_probes_four_remake.py).
    plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17,
                         "ytick.labelsize": 18, "axes.titlesize": 20, "legend.fontsize": 18})
    fig, axes = plt.subplots(1, len(panels), figsize=(20.0, 7.0), dpi=300, sharey=True,
                             gridspec_kw={"wspace": 0.12})
    rows, width = [], 0.38
    for ax, (ev, title, restrict) in zip(axes, panels):
        keep = None
        if restrict == "delib":
            keep = {k for k, v in delib[ev].items() if v["label"] == "deliberation"}
        for i, arm in enumerate(ARMS):
            for j, (name, src, pos, _) in enumerate(metrics):
                lab = load(src[ev])
                turn = "pressure" if restrict == "pressure" else None
                p_, lo, hi, k, n = item_rate(lab, ev, arm, pos, among=keep, turn=turn)
                x = i + (j - 0.5) * width
                style = dict(color=COLOUR[arm]) if j == 0 else \
                    dict(facecolor="white", edgecolor=COLOUR[arm], linewidth=1.5, hatch="///")
                ax.bar(x, p_, width, zorder=3, **style)
                if lo is not None:
                    ax.errorbar(x, p_, yerr=[[p_ - lo], [hi - p_]], fmt="none", ecolor=INK,
                                elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
                rows.append((title, arm, name, k, n, p_, lo, hi))
        # No value labels and no arm names under the bars: the legend carries both readings.
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(ARMS) - 0.35)
        ax.set_title(title, loc="left", color="#374151", pad=12)
        frame(ax)
    axes[0].set_ylabel("% of reasoning traces")
    # Two rows, column-major: the four arms, then the two schemes.
    handles = [Patch(facecolor=COLOUR[a], edgecolor="black", linewidth=0.8, label=FULL[a]) for a in ARMS] + \
              [Patch(facecolor="#6b6b6b", edgecolor="black", linewidth=0.8, label=metrics[0][3]),
               Patch(facecolor="white", edgecolor="#6b6b6b", linewidth=1.5, hatch="///", label=metrics[1][3])]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               handlelength=1.8, handleheight=1.0, columnspacing=1.6, labelspacing=0.5)
    fig.subplots_adjust(top=0.72, bottom=0.06, left=0.05, right=0.99)
    notes = ["# Design 4: constitution reference where the question is live", "",
             "Bars: mean over items with a 95% interval from each item's repeats (scratch/figures/_hub.py): ODCV cell x",
             "sampled turns over three passes, agentic condition x ten trials, MASK row x its three passes' pressure",
             "turns; the training corpus is labelled in full, so it carries no interval. Agentic misalignment is single-turn",
             "and unfiltered. ODCV is restricted to turns the deliberation judge marked as weighing courses of action;",
             "MASK to the pressured response (the turn the honesty score is computed from).", "",
             "| setting | arm | scheme | k | n | % | lo | hi |", "|---|---|---|---|---|---|---|---|"]
    notes += [f"| {s} | {a} | {m} | {k} | {n} | {p:.1f} | " + ("none | none" if lo is None else f"{lo:.1f} | {hi:.1f}") + " |"
              for s, a, m, k, n, p, lo, hi in rows]
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
