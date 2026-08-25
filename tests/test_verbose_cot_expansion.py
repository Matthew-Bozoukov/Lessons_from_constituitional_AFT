# ABOUTME: Unit tests for the CoT verbosity-expansion primitives: the paragraph-budget
# ABOUTME: deriver and the relative-length lint contract. No network.

from __future__ import annotations

import pytest

from src.data.synth.derive import derive_vars, expansion_plan, split_sentences
from src.data.synth.stage_runtime import lint_problems

def _para(n: int, tag: str) -> str:
    """`n` distinct 17-word sentences, so split_sentences has real seams to find."""
    return " ".join(
        f"Consideration {tag}{i} weighs the situation and arrives somewhere worth "
        f"stating plainly before moving on." for i in range(1, n + 1))


# Sized like a real difficult-advice deliberation (~440 words over four uneven
# paragraphs), because the budget arithmetic only bites at realistic length: a toy
# 94-word source never exceeds max_alloc on any paragraph and would test nothing.
SOURCE = "\n\n".join([_para(4, "a"), _para(12, "b"), _para(7, "c"), _para(3, "d")])
REC = {"source_reasoning": SOURCE}


def test_split_sentences_never_returns_empty():
    assert split_sentences("") == [""]
    assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_plan_preserves_every_word_in_order():
    """The unit cut is a re-chunking, not an edit: no word may be lost or reordered."""
    out = expansion_plan(REC, source="source_reasoning", multiple=4.6)
    units = [b.split("]\n", 1)[1] for b in out["plan"].split("\n\n")]
    assert " ".join(units).split() == SOURCE.split()


def test_plan_emits_one_block_per_unit_and_counts_them():
    out = expansion_plan(REC, source="source_reasoning", multiple=4.6)
    blocks = out["plan"].split("\n\n")
    assert len(blocks) == int(out["n_runs"])
    assert all(b.startswith("[paragraph ") for b in blocks)
    # Numbered 1..n in order, so the model's `<run n=...>` sequence is checkable.
    assert [int(b.split()[1]) for b in blocks] == list(range(1, len(blocks) + 1))


def test_long_paragraph_is_cut_and_short_ones_are_not():
    """A 62-word paragraph carries more than max_alloc at 4.6x, so it must be split."""
    out = expansion_plan(REC, source="source_reasoning", multiple=4.6, max_alloc=3)
    assert int(out["n_runs"]) > 4, "the long paragraph should have been cut"
    out_uncut = expansion_plan(REC, source="source_reasoning", multiple=1.0, max_alloc=3)
    assert int(out_uncut["n_runs"]) == 4, "at 1x nothing needs cutting"


def test_budget_is_quoted_per_sentence():
    """The per-sentence figure is the fix for short paragraphs being under-expanded."""
    out = expansion_plan(REC, source="source_reasoning", multiple=4.6, para_words=170)
    assert "words of thinking per sentence]" in out["plan"]
    assert out["per_para_words"] == "170"


def test_every_unit_gets_at_least_one_output_paragraph():
    """A unit allocated zero paragraphs is content dropped from the corpus."""
    for multiple in (1.0, 2.0, 3.0, 4.6, 8.0):
        out = expansion_plan(REC, source="source_reasoning", multiple=multiple)
        allocs = [int(b.split(" -> ")[1].split()[0]) for b in out["plan"].split("\n\n")]
        assert min(allocs) >= 1, f"a unit got no budget at {multiple}x"


def test_plan_rejects_an_empty_source():
    with pytest.raises(ValueError, match="is empty"):
        expansion_plan({"source_reasoning": "   "}, source="source_reasoning",
                       multiple=3.0)


def test_derive_vars_rejects_an_unregistered_function():
    with pytest.raises(ValueError, match="unknown deriver"):
        derive_vars({"fn": "not_a_real_deriver"}, REC)


def test_derive_vars_is_inert_without_a_spec():
    assert derive_vars(None, REC) == {}


# --- the relative-length lint contract -------------------------------------------------

RATIO_SPEC = {"fields": ["reasoning"], "ratio_of": "source_reasoning",
              "min_word_ratio": 2.0, "max_word_ratio": 4.5}
BASE = {"source_reasoning": " ".join(["word"] * 100)}


def test_ratio_lint_accepts_in_band():
    assert lint_problems({"reasoning": " ".join(["w"] * 300)}, RATIO_SPEC, BASE) == []


def test_ratio_lint_rejects_an_echo():
    problems = lint_problems({"reasoning": " ".join(["w"] * 100)}, RATIO_SPEC, BASE)
    assert len(problems) == 1 and "under the 2.0x minimum" in problems[0]


def test_ratio_lint_rejects_a_runaway():
    problems = lint_problems({"reasoning": " ".join(["w"] * 500)}, RATIO_SPEC, BASE)
    assert len(problems) == 1 and "over the 4.5x maximum" in problems[0]


def test_ratio_lint_fails_loudly_without_the_record():
    """A ratio contract that silently no-ops is worse than none: the run reports clean."""
    with pytest.raises(ValueError, match="needs the record"):
        lint_problems({"reasoning": "x"}, RATIO_SPEC, None)


def test_ratio_lint_fails_loudly_on_a_missing_base_field():
    with pytest.raises(ValueError, match="empty or missing"):
        lint_problems({"reasoning": "x"}, RATIO_SPEC, {"other": "y"})


def test_ratio_lint_requires_ratio_of():
    with pytest.raises(ValueError, match="require `ratio_of`"):
        lint_problems({"reasoning": "x"}, {"fields": ["reasoning"],
                                           "min_word_ratio": 2.0}, BASE)


def test_absolute_and_ratio_contracts_compose():
    """min_chars and the ratio band are independent guards on the same tag."""
    spec = {**RATIO_SPEC, "min_chars": 10_000}
    problems = lint_problems({"reasoning": " ".join(["w"] * 300)}, spec, BASE)
    assert len(problems) == 1 and "under the 10000 minimum" in problems[0]


# --- prompt scaffolding must not reach the corpus ---------------------------------------

from src.data.synth.stage_operators import strip_scaffolding  # noqa: E402

RUN_TAGS = [r'</?run[^>]*>']


def test_strip_removes_run_tags_and_closes_the_seam():
    raw = '<run n="1">\nfirst para\n\nsecond para\n</run>\n<run n="2">\nthird para\n</run>'
    out = strip_scaffolding(raw, RUN_TAGS)
    assert "<run" not in out and "</run>" not in out
    assert out == "first para\n\nsecond para\n\nthird para"


def test_strip_never_leaves_a_triple_newline():
    """A widened paragraph break is a visible seam in text that must read continuously."""
    out = strip_scaffolding('a\n</run>\n<run n="2">\nb', RUN_TAGS)
    assert "\n\n\n" not in out and out == "a\n\nb"


def test_strip_is_inert_without_patterns():
    text = "untouched\n\ntext"
    assert strip_scaffolding(text, []) == text


def test_exhausted_mark_wins_over_also():
    """`also` stamps a status on every record; `on_exhausted.mark` overrides it for the
    failures. Merged the other way round a fallback row reports itself as a clean one."""
    record, also = {"reasoning": "new", "source_reasoning": "orig"}, {"status": "expanded"}
    copy, mark = {"reasoning": "source_reasoning"}, {"status": "fallback"}
    merged = {**record, **{f: record[s] for f, s in copy.items()}, **also, **mark}
    assert merged["status"] == "fallback"
    assert merged["reasoning"] == "orig"
