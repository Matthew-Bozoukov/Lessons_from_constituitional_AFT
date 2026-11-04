# ABOUTME: Figure code draws the mean with a 95% interval from the per-item SE, built on the log-odds scale;
# ABOUTME: this refuses figure scripts that build pass-level t-intervals or import the stats machinery.
"""The rule lives at the top of scratch/figures/_hub.py. The benchmark is fixed: a single-model
bar is the mean over its items with a 95% interval from the standard error propagated from each
item's own repeats, built on the log-odds scale so it stays inside the range (the SE has one
degree of freedom per disagreeing item, so the normal multiplier is honest); paired-difference
plots are the only exception. Scripts that still draw Wilson intervals over ITEMS (rows, episodes,
seeds) are listed in WILSON_OVER_ITEMS. Colosseum is the one case where that is RIGHT: its
episodes are independent draws of one scenario, so Wilson over episodes is the interval (kept
deliberately, 2026-11-03). The per-row and per-seed uses (reference rate, lie rate, Petri seeds)
measure a per-item proportion over a fixed benchmark, not a repeat, and each still needs its own
decision before the rule can cover it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_FILES = sorted(ROOT.glob("scratch/figures/*.py")) + sorted(ROOT.glob("scratch/plot_*.py"))
FORBIDDEN = (r"\blogit_t\(", r"\bt_mean\(", r"\bDesign\(", r"_shape_interval", r"from src\.eval\.stats import")
WILSON_OVER_ITEMS = {
    "scratch/plot_constitution_reference_rate.py", "scratch/plot_delib_2x2_odcv_mask.py",
    "scratch/plot_deliberation_carrier.py", "scratch/plot_deliberation_and_citation.py",
    "scratch/plot_stakes_two_panel.py", "scratch/plot_probes_four_remake.py",
}


def test_figures_do_not_build_pass_level_intervals():
    bad = []
    for f in FIGURE_FILES:
        text = f.read_text()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                bad.append(f"{f.relative_to(ROOT)}: {pat}")
    assert not bad, "figure code must take its bars from scratch/figures/_hub.item_se:\n" + "\n".join(bad)


def test_wilson_use_is_declared():
    """A new Wilson interval in figure code is a new per-item estimand: declare it or use item_se."""
    undeclared = []
    for f in FIGURE_FILES:
        rel = str(f.relative_to(ROOT))
        if rel == "scratch/figures/_hub.py":
            continue   # defines wilson for the literal() placeholder only
        if re.search(r"\bwilson\(", f.read_text()) and rel not in WILSON_OVER_ITEMS:
            undeclared.append(rel)
    assert not undeclared, "undeclared Wilson intervals in figure code: " + ", ".join(undeclared)
