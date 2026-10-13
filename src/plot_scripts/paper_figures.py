# ABOUTME: Every paper figure in one place: arm names, probe names and the base line live here, so
# ABOUTME: a naming change is one edit and `uv run figures` regenerates the whole set consistently.
"""Regenerate the paper's figures.

    uv run python -m src.plot_scripts.paper_figures            # all of them
    uv run python -m src.plot_scripts.paper_figures --only probes_three_arm
    uv run python -m src.plot_scripts.paper_figures --list

Every bar is read from a published run at render time (src/plot_scripts/_hub.py); the few
exceptions carry a comment saying where the number came from. House style, applied everywhere:

  * Arm names appear ONLY in the legend -- never as tick labels under the bars -- and carry no
    dose percentage. The legend also carries the dashed base line, so nothing in the figure is
    unexplained.
  * The untrained reference is drawn where a base measurement exists and is named for the model.
  * MASK scores honesty (higher is better) while the other probes are rates (lower is better),
    so each panel states its own direction rather than sharing one axis label.
"""
import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.plot_scripts import _hub

OUT = Path("output/plots")

# ---------------------------------------------------------------- names and style

# One place for every arm's display name. Renamed 2026-10-13: the dose is no longer in the label
# (it is in the caption and the adapter name), `nosynth`/`plain` is the standard-SFT control, and
# the corpus variants use plain English rather than the config's slugs.
ARM = {
    "control": "Control (Standard SFT)",
    "da": "Difficult advice",
    "da_tools": "Difficult advice + tools",
    "delib": "Deliberative alignment",
    "low_stakes": "Low stakes",
    "amoral": "Amoral",
    "grok": "Grok-authored",
    "qwen": "Qwen-authored",
    "mask_cot": "mask_cot",
    "empty_think": "Empty think",
    "cot_only": "CoT only",
}
PROBE = {
    "odcv": "ODCV",
    "agentic": "Agentic Misalignment",
    "mask": "MASK honesty",
    "petri": "Petri audit",
    "hospital": "Colosseum hospital",
}
COLOUR = {
    "control": "#5a6472", "da": "#7e22ce", "da_tools": "#d0558f", "delib": "#2196f3",
    "low_stakes": "#2fa360", "amoral": "#9b8bd6", "grok": "#c77d00", "qwen": "#2fa360",
    "mask_cot": "#b3261e", "empty_think": "#c77d00", "cot_only": "#2fa360",
}
BASE_COLOUR = "#c1121f"
BASE_NAME = "Qwen-3.6-27b"

# The untrained base, per probe. A probe with no base run is absent on purpose rather than
# carrying a borrowed number.
#   odcv     2026-09-06-odcv-qwen36, 240 rollouts, the same fixed design as the bars.
#   agentic  2026-10-12 base run, 180 rollouts. JUDGED BY gemini-3-flash-preview because
#            sonnet-4.5 returned empty content on 14 leaking cells; the bars are sonnet-judged,
#            so that line and those bars are not judge-matched.
#   mask     2026-10-03-mask-qwen36 (56.9); three earlier base runs read 58.1/58.2/58.4.
#   hospital 2026-09-29-hospital-self-sacrificial-qwen36, 30 seeds, 0 failed.
BASE = {"odcv": 39.6, "agentic": 45.0, "mask": 56.9, "hospital": 96.7}

DIRECTION = {"odcv": "lower is better", "agentic": "lower is better",
             "mask": "higher is better", "petri": "lower is better",
             "hospital": "lower is better"}
YLABEL = {"odcv": "Misalignment rate (%)", "agentic": "Harmful action rate (%)",
          "mask": "Honesty score (%)", "petri": "Misalignment rate (%)",
          "hospital": "Sabotage act rate (%)"}

plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 14, "axes.titlesize": 15, "legend.fontsize": 13})


def _panel(ax, probe: str, arms: list[str], values: list, ymax: float | None = None) -> None:
    """One probe's bars. No tick labels: the arms are identified by the figure's legend."""
    for i, (arm, v) in enumerate(zip(arms, values)):
        if v is None:
            ax.text(i, 2, "no\nrun", ha="center", va="bottom", fontsize=12, color="#8a93a0")
            continue
        m, lo, hi = v
        ax.bar(i, m, width=0.68, color=COLOUR[arm], edgecolor="black", linewidth=0.8, zorder=3)
        top = m
        if lo is not None:
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#2b2523",
                        elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
            top = hi
        ax.text(i, top + 1.4, f"{m:.1f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold", zorder=5)
    base = BASE.get(probe)
    if base is not None:
        ax.axhline(base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
    finite = [v[2] if v[2] is not None else v[0] for v in values if v is not None]
    ceiling = max(finite + ([base] if base else []))
    ax.set_ylim(0, ymax or max(ceiling * 1.3, 12))
    ax.set_title(f"{PROBE[probe]}\n({DIRECTION[probe]})")
    ax.set_ylabel(YLABEL[probe])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([])                       # arms live in the legend
    ax.set_xlim(-0.65, len(arms) - 0.35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _legend(fig, arms: list[str], *, with_base: bool, ncol: int | None = None) -> None:
    """Arm names and, where a base line is drawn, the line itself."""
    handles = [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in arms]
    if with_base:
        handles.append(Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)),
                              linewidth=1.8, label=f"{BASE_NAME} (base)"))
    fig.legend(handles=handles, frameon=False, loc="upper center",
               ncol=ncol or len(handles), bbox_to_anchor=(0.5, 1.0))


def _save(fig, name: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    paths = []
    for ext in ("png", "svg"):
        p = OUT / f"{name}_{stamp}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _grid(name: str, probes: list[str], arms: list[str], runs: dict, width: float = 5.1):
    """A row of probe panels over one arm set, with one shared legend."""
    fig, axes = plt.subplots(1, len(probes), figsize=(width * len(probes), 5.9),
                             gridspec_kw={"wspace": 0.28})
    axes = np.atleast_1d(axes)
    for ax, probe in zip(axes, probes):
        _panel(ax, probe, arms, [runs[probe][a] for a in arms])
    _legend(fig, arms, with_base=any(p in BASE for p in probes))
    fig.subplots_adjust(top=0.80, bottom=0.08)
    return _save(fig, name)


# ---------------------------------------------------------------- the figures

def probes_three_arm():
    """ODCV / MASK / hospital for the control, difficult advice and difficult advice + tools."""
    arms = ["control", "da", "da_tools"]
    runs = {
        "odcv": {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
                 "da": _hub.odcv("2026-10-08-odcv-qwen36-0-da-15-canary"),
                 "da_tools": _hub.odcv("2026-10-08-odcv-qwen36-0-da-tools-15-canary-reusedtools")},
        "mask": {"control": _hub.literal(42.7),   # 2026-10-09 plain run, 2 passes
                 "da": _hub.mask("2026-10-08-mask-qwen36-0-da-15-canary"),
                 "da_tools": _hub.mask("2026-10-08-mask-qwen36-0-da-tools-15-canary-reusedtools")},
        # Hospital sabotage-ACT rate: acted, with or without a private refusal, out of 30 shifts.
        # Counts recomputed from each run's own rollouts (scratch/.../sabotage boxes).
        "hospital": {"control": _hub.literal(100 * 29 / 30, 30),
                     "da": _hub.literal(100 * 28 / 30, 30),
                     "da_tools": _hub.literal(100 * 15 / 30, 30)},
    }
    return _grid("probes_three_arm", ["odcv", "mask", "hospital"], arms, runs)


def probes_four():
    """ODCV / agentic / Petri / hospital for the control, deliberative alignment and DA."""
    arms = ["control", "delib", "da"]
    runs = {
        "odcv": {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
                 "delib": _hub.odcv("2026-10-08-odcv-qwen36-0-delib-sonnet-15"),
                 "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15")},
        "agentic": {"control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
                    "delib": _hub.agentic("2026-10-08-am-qwen36-0-delib-sonnet-15"),
                    "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15")},
        # Petri: concerning>=4 over the audited seeds. The deliberative bar is delib-sonnet-7,
        # a DIFFERENT checkpoint from the other two panels -- no Petri audit exists on
        # delib-sonnet-15. No base audit exists either, hence no line on that panel.
        "petri": {"control": _hub.literal(100 * 33 / 90, 90),
                  "delib": _hub.literal(100 * 13 / 89, 89),
                  "da": _hub.literal(100 * 5 / 90, 90)},
        "hospital": {"control": _hub.literal(100 * 29 / 30, 30),
                     "delib": _hub.literal(100.0, 30),
                     "da": _hub.literal(100 * 29 / 30, 30)},
    }
    return _grid("probes_four", ["odcv", "agentic", "petri", "hospital"], arms, runs, width=4.6)


def da_variants():
    """The corpus variants at one dose: standard difficult advice, low stakes, amoral."""
    arms = ["control", "da", "low_stakes", "amoral"]
    runs = {
        "odcv": {"control": _hub.odcv("2026-10-02-odcv-qwen36-0-nosynth"),
                 "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
                 "low_stakes": _hub.odcv("2026-10-07-odcv-qwen36-0-da-lowstakes-practical-15"),
                 "amoral": _hub.odcv("2026-10-07-odcv-qwen36-0-nonmoral-original-15")},
        "agentic": {"control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
                    "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15"),
                    "low_stakes": _hub.agentic("2026-10-08-am-qwen36-0-da-lowstakes-practical-15"),
                    "amoral": _hub.agentic("2026-10-09-am-qwen36-0-nonmoral-original-15")},
        "mask": {"control": _hub.literal(42.7),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "low_stakes": _hub.mask("2026-10-07-mask-qwen36-0-da-lowstakes-practical-15"),
                 "amoral": _hub.mask("2026-10-07-mask-qwen36-0-nonmoral-original-15")},
    }
    return _grid("da_variants", ["odcv", "agentic", "mask"], arms, runs)


def da_authors():
    """Difficult advice by the model that WROTE the corpus."""
    arms = ["control", "da", "grok", "qwen"]
    runs = {
        "odcv": {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
                 "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
                 "grok": _hub.odcv("2026-10-08-odcv-qwen36-0-da-grok-15"),
                 "qwen": _hub.odcv("2026-10-08-odcv-qwen36-0-da-qwen-15")},
        # grok's agentic value is from a SEPARATE gpt-5 run supplied 2026-10-13 and not yet
        # published; our own run aborted at 87/180 on empty leaking_* responses. Replace the
        # literal with the dataset name once it lands.
        "agentic": {"control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
                    "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15"),
                    "grok": _hub.literal(5.2, 180),
                    "qwen": _hub.agentic("2026-10-08-am-qwen36-0-da-qwen-15")},
        "mask": {"control": _hub.literal(42.7),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "grok": _hub.mask("2026-10-09-mask-qwen36-0-da-grok-15"),
                 "qwen": _hub.mask("2026-10-09-mask-qwen36-0-da-qwen-15")},
    }
    return _grid("da_authors", ["odcv", "agentic", "mask"], arms, runs)


def trace_ablations():
    """What the loss is taken over: the response only, an emptied think block, all of it, the CoT."""
    arms = ["control", "mask_cot", "empty_think", "da", "cot_only"]
    runs = {
        "odcv": {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
                 "mask_cot": _hub.odcv("2026-10-11-odcv-qwen36-0-da-15-response-only"),
                 "empty_think": _hub.odcv("2026-10-08-odcv-qwen36-0-da-15-emptythink"),
                 "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
                 "cot_only": _hub.odcv("2026-10-11-odcv-qwen36-0-da-15-cot")},
        "agentic": {"control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
                    "mask_cot": _hub.agentic("2026-10-11-am-qwen36-0-da-15-response-only"),
                    "empty_think": _hub.agentic("2026-10-11-am-qwen36-0-da-15-emptythink"),
                    "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15"),
                    "cot_only": _hub.agentic("2026-10-11-am-qwen36-0-da-15-cot")},
        # empty-think MASK was supplied by the user 2026-10-11; no dataset exists for that arm.
        "mask": {"control": _hub.literal(42.7),
                 "mask_cot": _hub.mask("2026-10-08-mask-qwen36-0-da-15-response-only"),
                 "empty_think": _hub.literal(71.2),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "cot_only": _hub.mask("2026-10-08-mask-qwen36-0-da-15-cot")},
    }
    return _grid("trace_ablations", ["odcv", "agentic", "mask"], arms, runs)


def delegated_harm():
    """Delegated harm, subagent delivery: did it act, did it refuse, did it tell a human."""
    arms = ["control", "delib", "da", "da_tools"]
    repos = {"control": "2026-10-12-dh-qwen36-0-plain",
             "delib": "2026-10-12-dh-qwen36-0-delib-sonnet-15",
             "da": "2026-10-12-dh-qwen36-0-da-15",
             "da_tools": "2026-10-12-dh-qwen36-0-da-tools-15"}
    keys = [("Harmful action", "harmful_action", "lower is better"),
            ("Explicit refusal", "explicit_refusal", "higher is better"),
            ("Concern raised with a human", "concern_any_human", "higher is better")]
    data = {a: _hub.fetch(r, "results/results.json")["arms"]["subagent"] for a, r in repos.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.9), gridspec_kw={"wspace": 0.26})
    for ax, (title, key, direction) in zip(axes, keys):
        for i, arm in enumerate(arms):
            a = data[arm]
            p = 100 * a["rates"][key]
            lo, hi = _hub.wilson(round(a["rates"][key] * a["valid"]), a["valid"])
            ax.bar(i, p, width=0.68, color=COLOUR[arm], edgecolor="black", linewidth=0.8, zorder=3)
            ax.errorbar(i, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none",
                        ecolor="#2b2523", elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
            ax.text(i, hi + 2.0, f"{p:.1f}", ha="center", va="bottom", fontsize=13,
                    fontweight="bold", zorder=5)
        ax.set_title(f"{title}\n({direction})")
        ax.set_ylabel("Episodes (%)")
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(arms) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    _legend(fig, arms, with_base=False)
    fig.subplots_adjust(top=0.82, bottom=0.08)
    return _save(fig, "delegated_harm_subagent")


FIGURES = {
    "probes_three_arm": probes_three_arm,
    "probes_four": probes_four,
    "da_variants": da_variants,
    "da_authors": da_authors,
    "trace_ablations": trace_ablations,
    "delegated_harm": delegated_harm,
}


def main() -> None:
    """Render one figure or all of them."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=sorted(FIGURES),
                    help="render just this figure (repeatable); default is all")
    ap.add_argument("--list", action="store_true", help="print the figure names and exit")
    a = ap.parse_args()
    if a.list:
        for name in sorted(FIGURES):
            print(f"  {name}")
        return
    for name in (a.only or sorted(FIGURES)):
        for p in FIGURES[name]():
            print(f">>> {name}: {p}")


if __name__ == "__main__":
    main()
