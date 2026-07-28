# ABOUTME: Unit tests for synthdoc core: hashing identity, RNG stream independence,
# ABOUTME: the data model, JSON parsing, and recipe mixtures. Fast and offline.

from __future__ import annotations

import pytest

from synthdoc.core.hashing import canonical, stable_hash, stream_rng
from synthdoc.core.parsing import (
    ParseError,
    extract_json,
    parse_scores,
    parse_turns,
    render_document,
)
from synthdoc.core.recipe import Recipe, RecipeError, draw, normalize
from synthdoc.core.registry import RegistryError, has, register, resolve
from synthdoc.core.types import Document, ScenarioSpec, SpecChunk, Turn, make_doc_id


def chunk(idx: int, parent: str = "sec_a", spec_id: str = "s") -> SpecChunk:
    """Build a throwaway SpecChunk for tests."""
    return SpecChunk(
        spec_id=spec_id,
        chunk_id=f"{spec_id}/bullet/{parent}/{idx:03d}",
        text=f"principle number {idx} about honesty and care",
        granularity="bullet",
        parent_id=f"{spec_id}/{parent}",
        order_idx=idx,
    )


def scenario(**kwargs) -> ScenarioSpec:
    """Build a ScenarioSpec with sensible defaults."""
    defaults = dict(
        chunks=(chunk(0),),
        grouping_strategy="single",
        doc_type="difficult_advice",
        axes={"tools": "none", "reasoning": "terse"},
        seed=7,
    )
    defaults.update(kwargs)
    return ScenarioSpec(**defaults)


# --- hashing ------------------------------------------------------------------


def test_canonical_is_key_order_independent():
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})


def test_stable_hash_is_deterministic_across_calls():
    payload = {"x": [1, 2, {"y": "z"}]}
    assert stable_hash(payload) == stable_hash(payload)
    assert stable_hash(payload) != stable_hash({"x": [1, 2, {"y": "w"}]})


def test_stream_rng_is_independent_per_stream():
    """Different streams at the same index must not be correlated."""
    a = [stream_rng(0, i, "doc_type").random() for i in range(50)]
    b = [stream_rng(0, i, "chunks").random() for i in range(50)]
    assert a != b


def test_stream_rng_is_reproducible():
    assert stream_rng(3, 11, "axis.tools").random() == stream_rng(3, 11, "axis.tools").random()


# --- data model ---------------------------------------------------------------


def test_scenario_hash_is_stable_and_order_sensitive():
    s1 = scenario()
    s2 = scenario()
    assert s1.scenario_hash == s2.scenario_hash
    assert s1.scenario_hash != scenario(doc_type="trait_conflict").scenario_hash
    assert s1.scenario_hash != scenario(axes={"tools": "agentic", "reasoning": "terse"}).scenario_hash


def test_scenario_hash_changes_when_chunk_text_changes():
    """A spec edit must invalidate scenarios rather than reuse stale conditions."""
    edited = SpecChunk(
        spec_id="s",
        chunk_id=chunk(0).chunk_id,
        text="a completely different principle",
        granularity="bullet",
        parent_id=chunk(0).parent_id,
        order_idx=0,
    )
    assert scenario().scenario_hash != scenario(chunks=(edited,)).scenario_hash


def test_doc_id_is_constant_across_stages():
    doc = Document(doc_id=make_doc_id(scenario().scenario_hash, "run1"), scenario=scenario())
    advanced = doc.advanced(3, "stage_03_filtered")
    assert advanced.doc_id == doc.doc_id
    assert advanced.input_doc_id == doc.doc_id
    assert advanced.stage_name == "stage_03_filtered"


def test_doc_id_depends_on_run_id():
    h = scenario().scenario_hash
    assert make_doc_id(h, "run1") != make_doc_id(h, "run2")


def test_advanced_does_not_alias_mutable_state():
    doc = Document(doc_id="d", scenario=scenario(), turns=[Turn("user", "hi")])
    advanced = doc.advanced(1, "stage_01_revised")
    advanced.turns.append(Turn("assistant", "yo"))
    assert len(doc.turns) == 1


def test_document_roundtrips_through_dict():
    doc = Document(
        doc_id="d",
        scenario=scenario(chunks=(chunk(0), chunk(1)), grouping_strategy="adjacent"),
        turns=[Turn("user", "u"), Turn("assistant", "a", thinking="t")],
        filter_scores={"autorater_overall": 4.0},
        filter_verdict="keep",
    )
    back = Document.from_dict(doc.to_dict())
    assert back.to_dict() == doc.to_dict()
    assert back.scenario.scenario_hash == doc.scenario.scenario_hash


def test_turn_rejects_unknown_role():
    with pytest.raises(ValueError):
        Turn("narrator", "once upon a time")


# --- registry -----------------------------------------------------------------


def test_register_and_resolve():
    @register("test_kind", "thing")
    class Thing:
        pass

    assert has("test_kind", "thing")
    assert resolve("test_kind", "thing") is Thing


def test_resolve_unknown_name_lists_alternatives():
    @register("test_kind2", "known")
    class Known:
        pass

    with pytest.raises(RegistryError, match="known"):
        resolve("test_kind2", "missing")


def test_duplicate_registration_is_rejected():
    @register("test_kind3", "dup")
    class A:
        pass

    with pytest.raises(RegistryError):

        @register("test_kind3", "dup")
        class B:
            pass


# --- parsing ------------------------------------------------------------------


def test_extract_json_handles_bare_fenced_and_wrapped():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure, here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}
    assert extract_json('prose [1, 2] more') == [1, 2]


def test_extract_json_ignores_braces_inside_strings():
    assert extract_json('{"a": "a } brace"}') == {"a": "a } brace"}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ParseError):
        extract_json("no json at all")


def test_parse_turns_accepts_object_and_list():
    payload = '{"turns": [{"role": "user", "content": "hi"}]}'
    assert parse_turns(payload)[0].role == "user"
    assert parse_turns('[{"role": "assistant", "content": "yo"}]')[0].content == "yo"


def test_parse_turns_rejects_bad_role_and_empty():
    with pytest.raises(ParseError):
        parse_turns('{"turns": [{"role": "wizard", "content": "hi"}]}')
    with pytest.raises(ParseError):
        parse_turns('{"turns": [{"role": "user", "content": ""}]}')


def test_parse_turns_normalizes_tool_calls_to_string():
    turns = parse_turns(
        '{"turns": [{"role": "assistant", "content": "c",'
        ' "tool_calls": [{"name": "f", "arguments": {}}]}]}'
    )
    assert isinstance(turns[0].tool_calls, str)


def test_parse_scores_falls_back_to_mean_overall():
    scores, overall, _ = parse_scores(
        '{"scores": {"realism": 4, "difficulty": 2}}', ["realism", "difficulty"], 5
    )
    assert scores == {"realism": 4.0, "difficulty": 2.0}
    assert overall == 3.0


def test_parse_scores_clamps_out_of_range():
    scores, _, _ = parse_scores('{"scores": {"realism": 99}}', ["realism"], 5)
    assert scores["realism"] == 5.0


def test_render_document_marks_roles_and_thinking():
    text = render_document([Turn("assistant", "answer", thinking="reasoning")])
    assert "ASSISTANT" in text and "[thinking]" in text and "answer" in text


# --- recipe -------------------------------------------------------------------


def test_normalize_sums_to_one():
    out = normalize({"a": 3, "b": 1}, "x")
    assert out == {"a": 0.75, "b": 0.25}


def test_normalize_rejects_empty_and_negative():
    with pytest.raises(RecipeError):
        normalize({}, "x")
    with pytest.raises(RecipeError):
        normalize({"a": -1}, "x")


def test_draw_respects_a_degenerate_mixture():
    import random

    assert draw({"only": 1.0}, random.Random(0)) == "only"


def test_recipe_reads_unreserved_keys_as_axes():
    recipe = Recipe.from_config(
        {
            "n": 10,
            "chunks_per_example": {1: 1.0},
            "grouping": {"random": 1.0},
            "doc_type": {"difficult_advice": 1.0},
            "tools": {"none": 1.0},
            "reasoning": {"terse": 1.0},
        }
    )
    assert recipe.axis_names == ["reasoning", "tools"]
    assert "single" in recipe.strategies


def test_recipe_rejects_missing_fields_and_bad_k():
    with pytest.raises(RecipeError):
        Recipe.from_config({"chunks_per_example": {1: 1.0}, "grouping": {}, "doc_type": {}})
    with pytest.raises(RecipeError):
        Recipe.from_config(
            {
                "n": 1,
                "chunks_per_example": {"many": 1.0},
                "grouping": {"random": 1.0},
                "doc_type": {"difficult_advice": 1.0},
            }
        )
