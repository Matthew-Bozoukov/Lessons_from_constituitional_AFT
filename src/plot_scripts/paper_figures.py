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

from omegaconf import OmegaConf

from src.plot_scripts import _hub

OUT = Path("output/plots")
CONFIGS = Path("configs/plots")
READERS = {"mask": _hub.mask, "odcv": _hub.odcv, "agentic": _hub.agentic}


def _runs_from_config(cfg) -> dict:
    """Panel -> arm -> (value, lo, hi), from a figure config's `runs` block.

    A source is `{repo}` (read by the panel's _hub reader), `{literal, n}` (a supplied rate with a
    Wilson interval over n) or `{value, lo, hi}` (a supplied point and interval). Every arm the
    config lists must have a source in every panel.
    """
    runs = {}
    for panel in cfg.panels:
        src = cfg.runs[panel]
        assert set(src) == set(cfg.arms), f"{panel}: sources {sorted(src)} != arms {sorted(cfg.arms)}"
        runs[panel] = {}
        for arm in cfg.arms:
            s = src[arm]
            if "repo" in s:
                runs[panel][arm] = READERS[panel](s.repo)
            elif "literal" in s:
                runs[panel][arm] = _hub.literal(s.literal, s.n)
            else:
                runs[panel][arm] = (s.value, s.lo, s.hi)
    return runs

# ---------------------------------------------------------------- names and style

# One place for every arm's display name. Renamed 2026-10-13: the dose is no longer in the label
# (it is in the caption and the adapter name), `nosynth`/`plain` is the standard-SFT control, and
# the corpus variants use plain English rather than the config's slugs.
BASE_NAME = "Qwen-3.6-27b"
ARM = {
    "control": "+ Control (Standard SFT)",
    "da": "+ difficult advice 15% mix",
    "da_tools": "+ difficult advice + tool defs 15% mix",
    "delib": "+ Deliberative Alignment 15% mix",
    "low_stakes": "+ low stakes 15% mix",
    "amoral": "+ amoral 15% mix",
    "grok": "+ grok-authored 15% mix",
    "qwen": "+ qwen-authored 15% mix",
    "mask_cot": "+ mask_cot 15% mix",
    "empty_think": "+ Empty think 15% mix",
    "cot_only": "+ CoT only 15% mix",
    "stated_harm": "- one-sided ablation 15%",
    "base_const": "+ Constitution in system prompt",
    "plain_const": "+ Control + constitution in system prompt",
}
PROBE = {
    "odcv": "ODCV",
    "agentic": "Agentic Misalignment",
    "single_agent": "Aggregate Single Agent misalignment",
    "mask": "MASK honesty",
    "petri": "Petri audit",
    "hospital": "Colosseum hospital",
}
# How many models a probe puts in the room. Panels are ordered so that a category's panels are
# adjacent and the figure carries a rule under each run of them, because "ODCV MASK Colosseum"
# reads as three unrelated bars while "MASK | ODCV | Colosseum" reads as chat, then one agent,
# then several.
CATEGORY = {"mask": "Chat", "odcv": "Single agent", "agentic": "Single agent",
            "single_agent": "Single agent", "petri": "Single agent",
            "hospital": "Multi agent"}
# Control, difficult advice and difficult advice + tools keep one colour in every figure, so they
# read across figures. Every other arm has a colour of its own, used by no other arm anywhere, so a
# colour never means two things; none is red, which is reserved for the Qwen-3.6-27b base line.
COLOUR = {
    "control": "#5a6472", "da": "#7e22ce", "da_tools": "#d0558f",
    "delib": "#2196f3", "base_const": "#0f8b8d", "plain_const": "#0f8b8d",
    "low_stakes": "#2ca02c", "amoral": "#c9a227",
    "grok": "#ff7f0e", "qwen": "#1b3a6b",
    "mask_cot": "#17becf", "empty_think": "#8c8c2a", "cot_only": "#a6d854",
    "stated_harm": "#6b4f2a",
}
BASE_COLOUR = "#c1121f"

# The untrained base, per probe. A probe with no base run is absent on purpose rather than
# carrying a borrowed number.
#   odcv     2026-09-06-odcv-qwen36, 240 rollouts, the same fixed design as the bars.
#   agentic  2026-10-12 base run, 180 rollouts. JUDGED BY gemini-3-flash-preview because
#            sonnet-4.5 returned empty content on 14 leaking cells; the bars are sonnet-judged,
#            so that line and those bars are not judge-matched.
#   mask     2026-10-03-mask-qwen36 (56.9); three earlier base runs read 58.1/58.2/58.4.
#   hospital 2026-09-29-hospital-self-sacrificial-qwen36, 30 seeds, 0 failed.
#   petri    2026-10-14-petri-qwen36-0-base-constitution, 33/90 at concerning>=4, at the user's
#            request 2026-10-14. That repo's scores.json is BYTE-IDENTICAL to the plain
#            control's and its summary names 2026-10-05-qwen36-0-plain as the target, so this
#            line currently equals the control bar.
BASE = {"odcv": 39.6, "agentic": 45.0, "mask": 56.9, "hospital": 96.7,
        "petri": 100 * 33 / 90}
# The combined single-agent line is the mean of the two probes it combines, so the line and the
# bars above it are built the same way.
BASE["single_agent"] = (BASE["odcv"] + BASE["agentic"]) / 2

DIRECTION = {"odcv": "lower is better", "agentic": "lower is better",
             "single_agent": "lower is better", "mask": "higher is better",
             "petri": "lower is better", "hospital": "lower is better"}
# Extra title clearance per probe (points). The aggregate title is the widest in the row and
# read as crowding its axis, so it sits a little higher.
TITLE_PAD = {"single_agent": 22}
YLABEL = {"odcv": "Misalignment rate (%)", "agentic": "Harmful action rate (%)",
          "single_agent": "Misalignment rate (%)", "mask": "Honesty score (%)",
          "petri": "Misalignment rate (%)", "hospital": "Sabotage act rate (%)"}

plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 14, "axes.titlesize": 15, "legend.fontsize": 13})


def _panel(ax, probe: str, arms: list[str], values: list, ymax: float | None = None,
           *, ylabel: str | None = "", title_direction: bool = True) -> None:
    """One probe's bars. No tick labels: the arms are identified by the figure's legend.

    Args:
        ymax: shared upper limit, so a row of panels can carry one scale.
        ylabel: the axis label; "" uses the probe's own, None leaves the axis unlabelled
            because the panel to its left already names the shared scale.
        title_direction: False when the direction is stated on the shared axis instead.
    """
    labels: list[tuple[int, float, float]] = []
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
        labels.append((i, m, top))
    base = BASE.get(probe)
    if base is not None:
        ax.axhline(base, color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)), linewidth=1.8, zorder=2)
    finite = [v[2] if v[2] is not None else v[0] for v in values if v is not None]
    ceiling = max(finite + ([base] if base else []))
    # Every measure here is a percentage, so the axis stops at 100 rather than inventing
    # headroom above a full scale; a value label that would sit above 100 is drawn outside the
    # axes (clip_on=False) and the title is raised to clear it.
    ax.set_ylim(0, min(ymax or max(ceiling * 1.3, 12), 100.0))
    if ax.get_ylim()[1] >= 99.5:
        ax.set_yticks(range(0, 101, 20))
    hi_lim = ax.get_ylim()[1]
    crowded = False
    for i, m, top in labels:
        y = top + 0.014 * hi_lim
        crowded = crowded or y > 0.93 * hi_lim
        # Five or more bars in a narrow panel leave no room for two adjacent "100.0" labels at 13pt.
        ax.text(i, y, f"{m:.1f}", ha="center", va="bottom", fontsize=13 if len(arms) < 5 else 11,
                fontweight="bold", zorder=5, clip_on=False)
    title = f"{PROBE[probe]}\n({DIRECTION[probe]})" if title_direction else PROBE[probe]
    ax.set_title(title, pad=30 if crowded else TITLE_PAD.get(probe, 10))
    if ylabel is not None:
        ax.set_ylabel(ylabel or YLABEL[probe])
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([])                       # arms live in the legend
    ax.set_xlim(-0.65, len(arms) - 0.35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _bands(fig, axes, probes: list[str]) -> None:
    """Rule each run of same-category panels and name the category under it."""
    groups: list[tuple[str, list]] = []
    for ax, probe in zip(axes, probes):
        cat = CATEGORY[probe]
        if groups and groups[-1][0] == cat:
            groups[-1][1].append(ax)
        else:
            groups.append((cat, [ax]))
    for cat, axs in groups:
        x0 = min(a.get_position().x0 for a in axs)
        x1 = max(a.get_position().x1 for a in axs)
        y = min(a.get_position().y0 for a in axs) - 0.045
        fig.add_artist(Line2D([x0, x1], [y, y], color="#2b2523", linewidth=1.4,
                              transform=fig.transFigure, clip_on=False))
        fig.text((x0 + x1) / 2, y - 0.018, f"{cat} eval" + ("s" if len(axs) > 1 else ""),
                 ha="center", va="top", fontsize=14)


def _legend(fig, arms: list[str], *, with_base: bool, ncol: int | None = None) -> None:
    """Arm names and, where a base line is drawn, the line itself."""
    handles = []
    if with_base:
        handles.append(Line2D([0], [0], color=BASE_COLOUR, linestyle=(0, (2.5, 2.5)),
                              linewidth=1.8, label=BASE_NAME))
    handles += [Patch(facecolor=COLOUR[a], edgecolor="black", label=ARM[a]) for a in arms]
    # One straight line across the top: never wrapped. The type shrinks to fit rather than the
    # legend growing a second row, so every figure's legend reads as a single band.
    chars = sum(len(h.get_label()) for h in handles) + 6 * len(handles)
    size = 13 if chars < 95 else (11.5 if chars < 125 else 10)
    fig.legend(handles=handles, frameon=False, loc="upper center",
               ncol=ncol or len(handles), bbox_to_anchor=(0.5, 1.0), fontsize=size,
               columnspacing=1.4, handlelength=1.8)


def _save(fig, name: str, rows: list[tuple] | None = None, note: str = "") -> list[Path]:
    """Write the figure and, beside it, the same numbers as markdown.

    The markdown mirror is the machine-readable copy: a later reader can grep a value without
    rendering a PNG or re-reading the Hub.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    paths = []
    for ext in ("png", "svg", "pdf"):
        p = OUT / f"{name}_{stamp}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    if rows:
        md = [f"# {name}", "", f"Rendered {datetime.now().isoformat(timespec='seconds')}.",
              "Every value is read from a published run at render time unless the row says "
              "otherwise."]
        if note:
            md += ["", note]
        md += ["", "| Panel | Arm | Value | 95% CI |", "|---|---|---|---|"]
        for panel, arm, v in rows:
            if v is None:
                md.append(f"| {panel} | {arm} | no run | |")
                continue
            m, lo, hi = v
            ci = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "none"
            md.append(f"| {panel} | {arm} | {m:.1f} | {ci} |")
        q = OUT / f"{name}_{stamp}_results.md"
        q.write_text("\n".join(md) + "\n", encoding="utf-8")
        paths.append(q)
    return paths


def _grid(name: str, probes: list[str], arms: list[str], runs: dict, width: float = 5.1,
          *, shared_y: bool = False, bands: bool = True, note: str = ""):
    """A row of probe panels over one arm set, with one shared legend.

    Args:
        shared_y: put every panel on one 0-100 scale and name it once, on the leftmost panel of
            each measurement direction -- the panels then read against each other directly.
        bands: rule and name each run of same-category panels (Chat / Single agent / Multi
            agent) under the axes.
    """
    fig, axes = plt.subplots(1, len(probes), figsize=(width * len(probes), 6.4),
                             gridspec_kw={"wspace": 0.28})
    axes = np.atleast_1d(axes)
    # On a shared scale the panels are grouped by what they MEASURE rather than by probe, so
    # the misalignment panels read against one another on one axis; the direction moves out of
    # each title and onto that axis, where it is stated once.
    shared_label = {"lower is better": "Misalignment rate (%), lower is better",
                    "higher is better": "Honesty score (%), higher is better"}
    seen: set[str] = set()
    for ax, probe in zip(axes, probes):
        if shared_y:
            key = DIRECTION[probe]
            label = shared_label[key] if key not in seen else None
            seen.add(key)
        else:
            label = ""
        _panel(ax, probe, arms, [runs[probe][a] for a in arms],
               ymax=100.0 if shared_y else None, ylabel=label,
               title_direction=not shared_y)
    _legend(fig, arms, with_base=any(p in BASE for p in probes))
    fig.subplots_adjust(top=0.80, bottom=0.14)
    if bands:
        _bands(fig, axes, probes)
    rows = [(PROBE[probe], ARM[a], runs[probe][a]) for probe in probes for a in arms]
    rows += [(PROBE[probe], f"{BASE_NAME} (base line)", (BASE[probe], None, None))
             for probe in probes if probe in BASE]
    return _save(fig, name, rows, note)


# ---------------------------------------------------------------- the figures

def probes_three_arm():
    """ODCV / MASK / hospital for the control, difficult advice and difficult advice + tools."""
    arms = ["control", "da", "da_tools"]
    # The control and difficult-advice bars are the SAME runs probes_four uses -- the mainline
    # 2026-10-05 adapters -- so one label means one checkpoint across the two figures. This
    # figure was built from the canary lineage (da-15-canary / da-tools-15-canary-reusedtools)
    # because that is the pair with a tools arm; that put 93.3 here against 96.7 there for a bar
    # with the same legend entry. The +tools bar stays on the canary checkpoint: no mainline
    # da-tools arm has a hospital run, so it is the only +tools measurement that exists.
    runs = {
        "odcv": {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
                 "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
                 "da_tools": _hub.odcv("2026-10-08-odcv-qwen36-0-da-tools-15-canary-reusedtools")},
        "mask": {"control": _hub.mask("2026-10-09-mask-qwen36-0-plain"),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "da_tools": _hub.mask("2026-10-08-mask-qwen36-0-da-tools-15-canary-reusedtools")},
        # Hospital sabotage-ACT rate: acted, with or without a private refusal, out of 30 shifts.
        "hospital": {"control": _hub.literal(100 * 29 / 30, 30),
                     "da": _hub.literal(100 * 29 / 30, 30),
                     "da_tools": _hub.literal(100 * 15 / 30, 30)},
    }
    return _grid("probes_three_arm", ["mask", "odcv", "hospital"], arms, runs)


def probes_four():
    """MASK, the aggregate single-agent rate, Petri and the hospital, over six legend entries.

    ODCV and agentic misalignment are reported as ONE aggregate single-agent number: the mean of
    the two rates, with the interval built from both (src/plot_scripts/_hub.combine). It is a mean
    of two differently-constructed rates, not a pooled rollout rate, and the per-probe panels in
    the other figures are where the two can be read apart.

    The constitution arm's Petri bar is the plain control with the WHOLE constitution in the target's
    system prompt, not the nine principles the arm's MASK and single-agent bars use; see its note.
    """
    # The constitution arm is the PLAIN control with the constitution's nine principles in its
    # system prompt (2026-10-14 principles-only runs), at the user's request 2026-10-15 -- so it
    # differs from the control bar in the prompt alone.
    arms = ["control", "delib", "plain_const", "da", "da_tools"]
    odcv = {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
            "delib": _hub.odcv("2026-10-08-odcv-qwen36-0-delib-sonnet-15"),
            "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
            "da_tools": _hub.odcv("2026-10-11-odcv-qwen36-0-da-tools-15"),
            "plain_const": _hub.odcv("2026-10-14-odcv-principles-qwen36-0-plain")}
    agentic = {"control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
               "delib": _hub.agentic("2026-10-08-am-qwen36-0-delib-sonnet-15"),
               "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15"),
               "da_tools": _hub.agentic("2026-10-13-am-qwen36-0-da-tools-15"),
               "plain_const": _hub.agentic("2026-10-14-am-principles-qwen36-0-plain")}
    runs = {
        "mask": {"control": _hub.mask("2026-10-09-mask-qwen36-0-plain"),
                 "delib": _hub.mask("2026-10-09-mask-qwen36-0-delib-sonnet-15"),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "da_tools": _hub.mask("2026-10-11-mask-qwen36-0-da-tools-15"),
                 "plain_const": _hub.mask("2026-10-14-mask-principles-qwen36-0-plain")},
        "single_agent": {a: _hub.combine(odcv[a], agentic[a]) for a in arms},
        # Petri: concerning>=4 over the audited seeds. The deliberative bar is delib-sonnet-7,
        # a DIFFERENT checkpoint from the other panels -- no Petri audit exists on
        # delib-sonnet-15. The panel's base line is BASE["petri"]; see its provenance note.
        "petri": {"control": _hub.literal(100 * 33 / 90, 90),
                  "delib": _hub.literal(100 * 13 / 89, 89),
                  "da": _hub.literal(100 * 5 / 90, 90),
                  "da_tools": _hub.literal(100 * 23 / 90, 90),
                  # 2026-10-15-petri-constitution-qwen36-0-plain: the plain control with the whole
                  # constitution first in every target system message (verified in 1239/1239),
                  # 28/90 over all 90 seeds, raw .eval on the Hub.
                  "plain_const": _hub.literal(100 * 28 / 90, 90)},
        # Hospital sabotage-ACT rate out of 30 shifts.
        "hospital": {"control": _hub.literal(100 * 29 / 30, 30),
                     "delib": _hub.literal(100.0, 30),
                     "da": _hub.literal(100 * 29 / 30, 30),
                     # 66.7 at the user's request 2026-10-15: the only published da-tools-15 run at
                     # that value on this measure is 2026-10-09-hospital-self-sacrificial-qwen36-0-
                     # da-tools-15-canary-septda-octbase, pair_sabotaged 10/15 (15 seeds) -- a CANARY
                     # checkpoint, not the 2026-10-11 mainline the other three panels use (that one
                     # is 2026-10-14-...-da-tools-15, 23/30 = 76.7).
                     "da_tools": _hub.literal(100 * 10 / 15, 15),
                     # 2026-10-15-hospital-self-sacrificial-constitution-qwen36-0-plain: the plain
                     # control with the whole constitution on the watched pair's prompt (nosynth peer,
                     # same config as the other bars), pair_sabotaged 30/30. Raw rollouts:
                     # ...-constitution-qwen36-0-plain-rollouts.
                     "plain_const": _hub.literal(100 * 30 / 30, 30)},
    }
    return _grid("probes_four", ["mask", "single_agent", "petri", "hospital"], arms, runs,
                 width=4.6, shared_y=True, bands=False,
                 note=("**Colosseum, difficult advice + tool defs.** 66.7 is "
                       "2026-10-09-hospital-self-sacrificial-qwen36-0-da-tools-15-canary-septda-octbase "
                       "(10/15 seeds), a canary checkpoint; the mainline 2026-10-11 checkpoint the "
                       "other panels use scored 23/30 = 76.7 (2026-10-14-...-da-tools-15).\n\n"
                       "**Constitution arm.** `+ Control + constitution in system prompt` is the "
                       "plain control with the constitution's nine principles prepended (no header "
                       "or priority section): 2026-10-14-{mask,odcv,am}-principles-qwen36-0-plain. "
                       "Its Colosseum bar (30/30) is 2026-10-15-hospital-self-sacrificial-constitution-"
                       "qwen36-0-plain, with the whole constitution on the watched pair.\n\n"
                       "**Petri.** The constitution arm's Petri bar (28/90) is "
                       "`2026-10-15-petri-constitution-qwen36-0-plain`: the plain control with the "
                       "whole constitution first in every target system message (verified in all "
                       "1239), all 90 seeds scored, raw .eval on the Hub. It uses the whole "
                       "constitution, not the nine principles of the arm's MASK and single-agent "
                       "bars. The Petri base line "
                       "(36.7) is read from `2026-10-14-petri-qwen36-0-base-constitution`, a "
                       "byte-identical copy of the plain control's audit, so it equals the "
                       "control bar."))


def da_variants(name: str = "da_variants",
                arms: tuple[str, ...] = ("control", "da", "low_stakes", "amoral")):
    """The corpus variants at one dose: standard difficult advice, low stakes, amoral.

    Args:
        name: Figure stem.
        arms: The arms drawn, in plot order; a subset drops bars, never changes a value.
    """
    arms = list(arms)
    runs = {
        # plain, not 2026-10-02-odcv-qwen36-0-nosynth: that run targets the NOSYNTH adapter
        # (40.0), a different control from the plain mix this figure's agentic and MASK bars
        # use, so the three panels were describing two different models under one label.
        "odcv": {"control": _hub.odcv("2026-10-06-odcv-qwen36-0-plain"),
                 "da": _hub.odcv("2026-10-06-odcv-qwen36-0-da-15"),
                 "low_stakes": _hub.odcv("2026-10-07-odcv-qwen36-0-da-lowstakes-practical-15"),
                 "amoral": _hub.odcv("2026-10-07-odcv-qwen36-0-nonmoral-original-15")},
        "agentic": {"control": _hub.agentic("2026-10-08-am-qwen36-0-plain"),
                    "da": _hub.agentic("2026-10-08-am-qwen36-0-da-15"),
                    "low_stakes": _hub.agentic("2026-10-08-am-qwen36-0-da-lowstakes-practical-15"),
                    "amoral": _hub.agentic("2026-10-09-am-qwen36-0-nonmoral-original-15")},
        "mask": {"control": _hub.mask("2026-10-09-mask-qwen36-0-plain"),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "low_stakes": _hub.mask("2026-10-07-mask-qwen36-0-da-lowstakes-practical-15"),
                 "amoral": _hub.mask("2026-10-07-mask-qwen36-0-nonmoral-original-15")},
    }
    return _grid(name, ["mask", "odcv", "agentic"], arms, runs)


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
        "mask": {"control": _hub.mask("2026-10-09-mask-qwen36-0-plain"),
                 "da": _hub.mask("2026-10-09-mask-qwen36-0-da-15"),
                 "grok": _hub.mask("2026-10-09-mask-qwen36-0-da-grok-15"),
                 "qwen": _hub.mask("2026-10-09-mask-qwen36-0-da-qwen-15")},
    }
    return _grid("da_authors", ["mask", "odcv", "agentic"], arms, runs)


def trace_ablations():
    """What the loss is taken over: the response only, an emptied think block, all of it, the CoT.

    Plus the stated-harm corpus ablation, at the user's request. CAUTION: that adapter
    (2026-10-04-qwen36-0-ablation-stated-harm-15) was trained on 2026-10-04-ablation-stated-harm-15-mix,
    whose replay half is the OLDER tulu/smol/no_robots set, not the nemotron replay every other
    arm here was trained on -- so it differs from them in replay data as well as in its corpus.
    """
    cfg = OmegaConf.load(CONFIGS / "trace_ablations.yaml")
    return _grid(cfg.name, list(cfg.panels), list(cfg.arms), _runs_from_config(cfg), note=cfg.note)


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
    rows: list[tuple] = []
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.2), gridspec_kw={"wspace": 0.26})
    for ax, (title, key, direction) in zip(axes, keys):
        crowded = False
        for i, arm in enumerate(arms):
            a = data[arm]
            p = 100 * a["rates"][key]
            lo, hi = _hub.wilson(round(a["rates"][key] * a["valid"]), a["valid"])
            ax.bar(i, p, width=0.68, color=COLOUR[arm], edgecolor="black", linewidth=0.8, zorder=3)
            ax.errorbar(i, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="none",
                        ecolor="#2b2523", elinewidth=1.3, capsize=5, capthick=1.3, zorder=4)
            rows.append((title, ARM[arm], (p, lo, hi)))
            crowded = crowded or hi > 93
            ax.text(i, hi + 1.4, f"{p:.1f}", ha="center", va="bottom", fontsize=13,
                    fontweight="bold", zorder=5, clip_on=False)
        # A percentage axis stops at 100; a label that would sit above it is drawn outside the
        # axes, and only then is the title raised to clear it -- raising every title would push
        # them into the legend.
        ax.set_title(f"{title}\n({direction})", pad=30 if crowded else 10)
        ax.set_ylabel("Episodes (%)")
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.65, len(arms) - 0.35)
        ax.grid(True, axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    _legend(fig, arms, with_base=False)
    fig.subplots_adjust(top=0.78, bottom=0.08)
    return _save(fig, "delegated_harm_subagent", rows)


FIGURES = {
    "probes_three_arm": probes_three_arm,
    "probes_four": probes_four,
    "da_variants": da_variants,
    "da_variants_no_amoral": lambda: da_variants("da_variants_no_amoral",
                                                 ("control", "da", "low_stakes")),
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
