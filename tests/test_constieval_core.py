# ABOUTME: Unit tests for constieval core: hashing identity, item id stability, the results
# ABOUTME: store schema, and the statistics behind every reported interval. Fast and offline.

from __future__ import annotations

import pytest

from constieval.core.hashing import canonical, stable_hash, stream_rng
from constieval.core.parsing import ParseError, coerce_score, extract_json, parse_verdict, split_thinking
from constieval.core.stats import agreement, bootstrap_mean, cohens_kappa, paired_delta, wilson
from constieval.core.store import COLUMNS, ResultsStore, RunContext, ScoreRow, build_rows
from constieval.core.types import Item, Verdict, condition_label, make_item_id


def item(**kwargs) -> Item:
    """Build a throwaway Item with sensible defaults."""
    defaults = dict(
        item_id="",
        clause_id="c1",
        family="application",
        difficulty="edge",
        prompt="a person asks for help with something awkward",
        meta={"norm_violating_path": "x", "legitimate_goal": "y"},
    )
    defaults.update(kwargs)
    built = Item(**defaults)
    return Item(**{**defaults, "item_id": make_item_id(built)})


class TestHashing:
    def test_canonical_is_order_independent(self):
        assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})

    def test_sets_are_rejected(self):
        # A set's iteration order is not stable across processes, so hashing one would
        # silently produce different ids for identical content.
        with pytest.raises(TypeError):
            canonical({"x": {1, 2}})

    def test_stable_hash_is_deterministic(self):
        assert stable_hash({"a": [1, 2]}) == stable_hash({"a": [1, 2]})

    def test_streams_are_independent(self):
        a = [stream_rng(0, i, "family.application").random() for i in range(5)]
        b = [stream_rng(0, i, "family.conflict").random() for i in range(5)]
        assert a != b
        # Re-drawing the first stream is unaffected by the second having been drawn.
        assert a == [stream_rng(0, i, "family.application").random() for i in range(5)]


class TestItemIdentity:
    def test_id_is_content_addressed(self):
        assert item().item_id == item().item_id

    def test_id_changes_with_prompt(self):
        assert item().item_id != item(prompt="different").item_id

    def test_id_ignores_run_and_recipe(self):
        # Nothing run-scoped enters the id, which is what lets two runs join row for row.
        assert "run" not in make_item_id(item())

    def test_derive_sets_parent_and_new_id(self):
        parent = item()
        child = parent.derive(pressure="system_override", system="operator prompt")
        assert child.parent_item_id == parent.item_id
        assert child.item_id != parent.item_id
        assert child.is_derived and not parent.is_derived

    def test_condition_label(self):
        assert item().condition == "clean"
        assert item(pressure="authority_claim").condition == "pressure:authority_claim"
        assert item(ood_axis="language", ood_value="spanish").condition == "ood:language=spanish"
        assert condition_label() == "clean"

    def test_messages_include_history_and_system(self):
        built = item(system="sys", history=({"role": "user", "content": "hi"},))
        roles = [m["role"] for m in built.messages()]
        assert roles == ["system", "user", "user"]

    def test_roundtrip(self):
        original = item(ood_axis="format", ood_value="tool_use")
        assert Item.from_dict(original.to_dict()) == original


class TestStore:
    def test_frame_has_declared_columns_even_when_empty(self):
        # A run containing no items of some family must still concatenate with one that does.
        assert list(ResultsStore().to_frame().columns) == list(COLUMNS)

    def test_undeclared_column_is_rejected(self):
        with pytest.raises(ValueError, match="undeclared columns"):
            ResultsStore().append({"run_id": "r", "typo_column": 1})

    def test_build_rows_fans_verdicts_out(self):
        ctx = RunContext(run_id="r1", recipe="baseline", itemset_id="is_x")
        rows = build_rows(
            ctx,
            item(),
            [
                Verdict(item_id="i", axis="compliance", score=1.0, passed=True),
                Verdict(item_id="i", axis="tension_recognition", score=0.5),
            ],
            clause_title="Sincere assertion",
        )
        assert [r.axis for r in rows] == ["compliance", "tension_recognition"]
        assert all(r.recipe == "baseline" and r.itemset_id == "is_x" for r in rows)

    def test_generation_error_propagates_to_rows(self):
        from constieval.core.types import Completion

        ctx = RunContext(run_id="r1", recipe="baseline")
        rows = build_rows(
            ctx,
            item(),
            [Verdict(item_id="i", axis="compliance", score=0.0)],
            completion=Completion(item_id="i", text="", error="timeout"),
        )
        assert rows[0].error == "timeout"

    def test_write_and_load_roundtrip(self, tmp_path):
        store = ResultsStore([ScoreRow(run_id="r", recipe="x", item_id="i", clause_id="c", axis="compliance", score=0.5)])
        path = store.write(tmp_path / "results.jsonl")
        assert len(ResultsStore.load(path)) == 1

    def test_load_dir_requires_a_match(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ResultsStore.load_dir(tmp_path)


class TestStats:
    def test_wilson_stays_inside_unit_interval_at_the_edges(self):
        # The reason Wilson is used at all: the normal approximation leaves [0, 1] here.
        interval = wilson(0, 8)
        assert 0.0 <= interval.lo <= interval.hi <= 1.0

    def test_wilson_empty_cell_is_unknown_not_zero(self):
        interval = wilson(0, 0)
        assert (interval.lo, interval.hi, interval.n) == (0.0, 1.0, 0)

    def test_wilson_narrows_with_n(self):
        wide = wilson(5, 10)
        narrow = wilson(500, 1000)
        assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)

    def test_bootstrap_brackets_the_mean(self):
        interval = bootstrap_mean([0.1, 0.5, 0.9, 0.4, 0.6], seed=3)
        assert interval.lo <= interval.mean <= interval.hi

    def test_bootstrap_single_observation_is_degenerate(self):
        interval = bootstrap_mean([0.42])
        assert interval.lo == interval.hi == pytest.approx(0.42)

    def test_kappa_perfect_and_chance(self):
        assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
        assert cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == pytest.approx(1.0)
        assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0

    def test_kappa_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            cohens_kappa([1, 0], [1])

    def test_agreement_reports_raw_and_kappa(self):
        result = agreement([1, 1, 0, 0], [1, 1, 0, 1])
        assert result["raw"] == pytest.approx(0.75)
        assert result["n"] == 4

    def test_paired_delta_uses_differences_not_group_means(self):
        # Both groups have the same mean, but every pair moved down by 0.1. A group-mean
        # comparison would report 0; the paired difference reports the real effect.
        clean = [0.5, 0.9, 0.3]
        stressed = [0.4, 0.8, 0.2]
        assert paired_delta(clean, stressed, seed=1).mean == pytest.approx(-0.1)

    def test_paired_delta_rejects_unpaired_input(self):
        with pytest.raises(ValueError, match="Unpaired"):
            paired_delta([0.1, 0.2], [0.1])


class TestParsing:
    def test_extract_json_from_fenced_prose(self):
        assert extract_json('here you go:\n```json\n{"score": 2}\n```') == {"score": 2}

    def test_extract_json_handles_braces_in_strings(self):
        assert extract_json('{"a": "not } a close"}') == {"a": "not } a close"}

    def test_extract_json_raises_on_prose(self):
        with pytest.raises(ParseError):
            extract_json("I would rather not answer that.")

    def test_split_thinking(self):
        thinking, answer = split_thinking("<think>weighing it up</think>Here is my answer.")
        assert thinking == "weighing it up"
        assert answer == "Here is my answer."

    def test_split_thinking_without_tags(self):
        assert split_thinking("plain answer") == ("", "plain answer")

    def test_parse_verdict_requires_declared_fields(self):
        with pytest.raises(ParseError, match="missing fields"):
            parse_verdict('{"score": 2}', ["score", "rationale"])

    def test_coerce_score_range_checks(self):
        assert coerce_score("2", 3) == 2.0
        with pytest.raises(ParseError, match="outside"):
            coerce_score(4, 3)
        with pytest.raises(ParseError, match="Non-numeric"):
            coerce_score("high", 3)
