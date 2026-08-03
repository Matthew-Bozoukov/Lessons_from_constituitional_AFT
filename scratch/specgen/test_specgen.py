# ABOUTME: Offline tests for specgen: section splitting, unit parsing, metrics, ARI,
# ABOUTME: and prompt template formatting. No network, no tokenizer downloads.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import prompts  # noqa: E402
from metrics import ari, doc_metrics, parse_units  # noqa: E402
from pipeline import PKG, _assemble, sections  # noqa: E402

WORDS = lambda text: len(text.split())  # noqa: E731 - injectable token counter

SOURCE = """# Claude's constitution

Intro prose.

## Being broadly safe

Safety text.

## Being broadly ethical

Ethics text.

## Being broadly ethical again

More ethics.
"""

UNIT = """## Be honest
You must never deceive. Prefer plain statements and weigh the costs honestly.

*Why:* deception compounds and generally corrodes trust across repeated interactions,
which is the core reason this constraint exists at all in the source document.

- Say the unwelcome thing plainly.

*When this does NOT apply:* do not weaponise honesty as an excuse to moralise."""


def _doc(units):
    return _assemble(units, (PKG / "preamble.md").read_text(),
                     (PKG / "closing.md").read_text())


def test_sections_split_on_h2_with_unique_slugs():
    secs = sections(SOURCE)
    assert [s["title"] for s in secs] == \
        ["Being broadly safe", "Being broadly ethical", "Being broadly ethical again"]
    ids = [s["section_id"] for s in secs]
    assert len(set(ids)) == 3 and ids[0] == "saf"


def test_parse_units_recovers_structure():
    units = parse_units(_doc([UNIT, UNIT]))
    assert len(units) == 2
    u = units[0]
    assert u["title"] == "1. Be honest"
    assert u["statement"].startswith("You must never deceive")
    assert "compounds" in u["why"] and "- Say" not in u["why"]
    assert u["cues"] == ["- Say the unwelcome thing plainly."]
    assert u["not_apply"].startswith("do not weaponise")


def test_doc_metrics_counts_and_ratios():
    m = doc_metrics(_doc([UNIT, UNIT]), WORDS)
    assert m["n_units"] == 2
    assert 0 < m["explanation_ratio"]["mean"] < 1
    assert m["modality_profile"]["hard"] > 0 and m["modality_profile"]["soft"] > 0
    assert m["tokens_per_unit"]["min"] == m["tokens_per_unit"]["max"]


def test_ari_identical_and_disjoint_partitions():
    a = {f"c{i}": i % 3 for i in range(30)}
    assert ari(a, dict(a)) == 1.0
    relabelled = {k: (v + 1) % 3 for k, v in a.items()}  # same partition, new labels
    assert ari(a, relabelled) == 1.0
    smashed = {k: 0 for k in a}  # everything in one cluster
    assert ari(a, smashed) < 0.1


def test_prompt_templates_format_cleanly():
    assert "exactly 12" in prompts.CLUSTER.format(n_principles=12, claims="x | y | z")[:600] \
        or "12 principles" in prompts.CLUSTER.format(n_principles=12, claims="x")
    unit = prompts.WRITE_UNIT.format(title="T", claims="- c", token_budget=280,
                                     cue_block=prompts.cue_block(2))
    assert "about 280 tokens" in unit and "exactly 2" in unit
    assert prompts.cue_block(0) == ""
    assert set(prompts.hashes()) == {"extract", "cluster", "write_unit", "revise"}
