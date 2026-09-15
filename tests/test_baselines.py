# ABOUTME: docs/BASELINES.md is the authority on which arm new work builds on and compares
# ABOUTME: against, so these lock it to the code and check every path it quotes still resolves.

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/BASELINES.md"
PLOT = ROOT / "scratch/gpt_seeds/plot_seed_mean.py"

# The difficult-advice baseline, spelled the way each layer spells it. Changing the baseline
# means changing this tuple, docs/BASELINES.md and plot_seed_mean.BASELINE_ARM together --
# which is the point: no one layer can drift on its own.
BASELINE_ARM_KEY = "principle_scoped"
BASELINE_SUBJECT = "difficult-advice-principle-scoped-702"
SUPERSEDED = ("da716", "synthdoc-716")


def test_baselines_doc_exists_and_names_the_difficult_advice_baseline():
    assert DOC.is_file(), "docs/BASELINES.md is the pointer CLAUDE.md sends people to"
    text = DOC.read_text(encoding="utf-8")
    assert "principle-scoped 702" in text
    # It must say what NOT to use, or a reader who already knows da716 will just keep using it.
    for old in SUPERSEDED:
        assert old in text, f"the doc should say explicitly not to start from {old}"


def test_every_repo_path_the_doc_quotes_resolves():
    """The failure this catches actually happened: a rename left the doc's train-config row
    pointing at a file that no longer existed. `uv run names` cannot see it — it skips HF
    extraction for .md and never stats a path — so a clean lint proved nothing."""
    text = DOC.read_text(encoding="utf-8")
    quoted = set(
        re.findall(
            r"(?:configs|scripts|src|tests)/[A-Za-z0-9_./{},-]+\.(?:yaml|py|sh|md)",
            text,
        )
    )
    dead = []
    for q in sorted(quoted):
        # A `{42,69}` brace pair stands for two real files; check both.
        for one in _expand_braces(q):
            if not (ROOT / one).exists():
                dead.append(one)
    assert not dead, f"docs/BASELINES.md points at files that do not exist: {dead}"


def _expand_braces(path: str) -> list[str]:
    m = re.search(r"\{([^}]*)\}", path)
    if not m:
        return [path]
    return [
        p
        for opt in m.group(1).split(",")
        for p in _expand_braces(path[: m.start()] + opt.strip() + path[m.end() :])
    ]


def test_the_plot_and_the_doc_agree_on_which_arm_is_the_baseline():
    spec = importlib.util.spec_from_file_location("plot_seed_mean", PLOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.BASELINE_ARM == BASELINE_ARM_KEY
    assert BASELINE_ARM_KEY in mod.ARMS, "the baseline must be an arm the figure draws"
    # And the arm the figure calls the baseline must be the corpus the doc names, not merely
    # a key that happens to match: check its sources point at the principle-scoped artifacts.
    srcs = " ".join(
        s if isinstance(s, str) else "/".join(s)
        for s in mod.ARMS[BASELINE_ARM_KEY]["seeds"].values()
    )
    assert "principle-scoped" in srcs or "principle_scoped" in srcs


@pytest.mark.parametrize("old", SUPERSEDED)
def test_the_superseded_arms_are_labelled_as_superseded_where_they_are_still_drawn(old):
    """da716 stays on the chart as the generator sweep's control. Its label has to say so,
    or the figure quietly shows two difficult-advice baselines and no way to tell them apart."""
    spec = importlib.util.spec_from_file_location("plot_seed_mean", PLOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    longs = " ".join(a["long"] for a in mod.ARMS.values()).upper()
    if old.upper() in longs:
        assert "SUPERSEDED" in longs or "BASELINE" in longs
