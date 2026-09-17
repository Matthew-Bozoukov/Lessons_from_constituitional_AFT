# ABOUTME: The seed-mean plot names ONE difficult-advice baseline arm, tied to the baseline corpus, and
# ABOUTME: labels any superseded arm it still draws. Run: uv run pytest tests/test_baselines.py -q

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLOT = ROOT / "scratch/gpt_seeds/plot_seed_mean.py"

# The difficult-advice baseline, spelled the way each layer spells it. Its artifacts (corpus,
# mixture, adapter, eval runs) and their numbers live on the Hub, not in a doc: docs/BASELINES.md
# was deleted 2026-09-17 after it went stale the day its arm was measured. Changing the baseline
# means changing this tuple and plot_seed_mean.BASELINE_ARM together.
BASELINE_ARM_KEY = "neutral"
BASELINE_SUBJECT = "2026-09-14-da-synth"
SUPERSEDED = ("principle-scoped 702", "da716", "synthdoc-716")


def _plot():
    spec = importlib.util.spec_from_file_location("plot_seed_mean", PLOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_plot_names_the_baseline_arm_and_ties_it_to_the_baseline_corpus():
    mod = _plot()
    assert mod.BASELINE_ARM == BASELINE_ARM_KEY
    assert BASELINE_ARM_KEY in mod.ARMS, "the baseline must be an arm the figure draws"
    # And the arm the figure calls the baseline must be the baseline's corpus, not merely a key
    # that happens to match: its seeds' eval runs, or -- until it has one -- the corpus it
    # declares, must point at the baseline's artifacts.
    arm = mod.ARMS[BASELINE_ARM_KEY]
    srcs = " ".join(
        s if isinstance(s, str) else "/".join(s) for s in arm["seeds"].values()
    )
    assert BASELINE_SUBJECT in f"{srcs} {arm.get('corpus', '')}"


@pytest.mark.parametrize("old", SUPERSEDED)
def test_the_superseded_arms_are_labelled_as_superseded_where_they_are_still_drawn(old):
    """da716 stays on the chart as the generator sweep's control. Its label has to say so,
    or the figure quietly shows two difficult-advice baselines and no way to tell them apart."""
    longs = " ".join(a["long"] for a in _plot().ARMS.values()).upper()
    if old.upper() in longs:
        assert "SUPERSEDED" in longs or "BASELINE" in longs
