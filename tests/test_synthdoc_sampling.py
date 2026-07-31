# ABOUTME: Tests for chunking, grouping, and the paired-sampling property that makes
# ABOUTME: sweep arms comparable. The pairing test is the one that protects the science.

from __future__ import annotations

import random

import pytest

from src.data.synthdoc.core.embeddings import EmbeddingIndex, HashingEmbedder
from src.data.synthdoc.core.recipe import MixtureSampler, Recipe
from src.data.synthdoc.core.specs import load_spec
from src.data.synthdoc.plugins.chunkers import build_chunker
from src.data.synthdoc.plugins.groupers import (
    GroupingContext,
    GroupingError,
    build_groupers,
)

SPEC_ID = "demo_spec"


@pytest.fixture(scope="module")
def chunks():
    """Bullet chunks of the demo spec."""
    spec = load_spec(SPEC_ID)
    return build_chunker({"granularity": "bullet", "min_words": 5}).chunk(spec)


@pytest.fixture(scope="module")
def index(chunks):
    """Embedding index over the demo spec chunks."""
    return EmbeddingIndex.build(
        ids=[c.chunk_id for c in chunks],
        texts=[c.text for c in chunks],
        embedder=HashingEmbedder(dim=256),
        cache_dir=None,
        tag=SPEC_ID,
    )


def recipe(**overrides):
    """Build a Recipe with defaults, overriding individual blocks."""
    cfg = {
        "n": 40,
        "chunks_per_example": {2: 1.0},
        "grouping": {"random": 1.0},
        "doc_type": {"difficult_advice": 0.5, "trait_conflict": 0.5},
        "tools": {"none": 0.5, "agentic": 0.5},
        "reasoning": {"terse": 0.5, "visible_cot": 0.5},
    }
    cfg.update(overrides)
    return Recipe.from_config(cfg)


# --- chunkers -----------------------------------------------------------------


def test_every_granularity_produces_chunks():
    spec = load_spec(SPEC_ID)
    for granularity in ("bullet", "trait", "section"):
        out = build_chunker({"granularity": granularity, "min_words": 5}).chunk(spec)
        assert out, granularity
        assert all(c.granularity == granularity for c in out)


def test_bullet_is_finer_than_section():
    spec = load_spec(SPEC_ID)
    bullets = build_chunker({"granularity": "bullet", "min_words": 5}).chunk(spec)
    sections = build_chunker({"granularity": "section", "min_words": 5}).chunk(spec)
    assert len(bullets) > len(sections)


def test_chunk_ids_are_unique_and_structural(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id.startswith(f"{SPEC_ID}/bullet/") for c in chunks)


def test_chunk_meta_flags_prescriptive_text(chunks):
    registers = {c.meta["register"] for c in chunks}
    assert "prescriptive" in registers


# --- groupers -----------------------------------------------------------------


def test_random_grouper_returns_k_distinct(chunks):
    g = build_groupers(["random"], chunks)["random"]
    group = g.group(chunks, 3, random.Random(0))
    assert len(group) == 3
    assert len({c.chunk_id for c in group}) == 3


def test_random_grouper_rejects_k_larger_than_pool(chunks):
    g = build_groupers(["random"], chunks)["random"]
    with pytest.raises(GroupingError):
        g.group(chunks[:2], 5, random.Random(0))


def test_adjacent_grouper_stays_within_a_section(chunks):
    g = build_groupers(["adjacent"], chunks, {"adjacent": {"same_section_only": True}})["adjacent"]
    for seed in range(20):
        group = g.group(chunks, 2, random.Random(seed))
        assert len({c.parent_id for c in group}) == 1
        order = [c.order_idx for c in group]
        assert order == sorted(order)
        assert order[-1] - order[0] == len(order) - 1


def test_adjacent_grouper_errors_when_no_section_is_large_enough(chunks):
    g = build_groupers(["adjacent"], chunks, {"adjacent": {"same_section_only": True}})["adjacent"]
    with pytest.raises(GroupingError, match="same_section_only"):
        g.group(chunks, 99, random.Random(0))


def test_adjacent_grouper_may_cross_sections_when_configured(chunks):
    g = build_groupers(["adjacent"], chunks, {"adjacent": {"same_section_only": False}})["adjacent"]
    crossed = any(
        len({c.parent_id for c in g.group(chunks, 3, random.Random(s))}) > 1 for s in range(40)
    )
    assert crossed


def test_semantic_grouper_respects_the_similarity_ceiling(chunks, index):
    ctx = GroupingContext(index=index)
    g = build_groupers(
        ["semantic"], chunks, {"semantic": {"min_similarity": 0.0, "max_similarity": 0.5}}, ctx
    )["semantic"]
    for seed in range(10):
        anchor, other = g.group(chunks, 2, random.Random(seed))
        assert index.similarity(anchor.chunk_id, other.chunk_id) <= 0.5


def test_semantic_grouper_requires_an_index(chunks):
    with pytest.raises(GroupingError, match="embedding index"):
        build_groupers(["semantic"], chunks, {}, GroupingContext(index=None))


def test_single_grouper_requires_k_one(chunks):
    g = build_groupers([], chunks)["single"]
    assert len(g.group(chunks, 1, random.Random(0))) == 1
    with pytest.raises(GroupingError):
        g.group(chunks, 2, random.Random(0))


# --- sampling and pairing -----------------------------------------------------


def sample(chunks, rec, seed=0):
    """Sample scenarios with the given recipe."""
    groupers = build_groupers(rec.strategies, chunks, rec.grouping_params)
    return list(MixtureSampler(groupers, seed=seed).sample(chunks, rec, rec.n))


def test_sampling_is_reproducible(chunks):
    a = sample(chunks, recipe())
    b = sample(chunks, recipe())
    assert [s.scenario_hash for s in a] == [s.scenario_hash for s in b]


def test_k_one_short_circuits_to_single(chunks):
    scenarios = sample(chunks, recipe(chunks_per_example={1: 1.0}))
    assert {s.grouping_strategy for s in scenarios} == {"single"}
    assert all(len(s.chunks) == 1 for s in scenarios)


def test_changing_one_axis_leaves_the_others_bit_identical(chunks):
    """The paired-sweep property: per-axis RNG streams isolate a mixture change.

    Without this, perturbing any recipe field reshuffles every downstream draw and
    two sweep arms stop being comparable row for row.
    """
    base = sample(chunks, recipe())
    varied = sample(chunks, recipe(doc_type={"difficult_advice": 0.1, "trait_conflict": 0.9}))

    assert [s.chunk_ids for s in base] == [s.chunk_ids for s in varied]
    assert [s.axes for s in base] == [s.axes for s in varied]
    assert [s.grouping_strategy for s in base] == [s.grouping_strategy for s in varied]
    assert [s.doc_type for s in base] != [s.doc_type for s in varied]


def test_changing_an_axis_mixture_leaves_chunks_and_doc_types_alone(chunks):
    base = sample(chunks, recipe())
    varied = sample(chunks, recipe(tools={"none": 0.05, "agentic": 0.95}))
    assert [s.chunk_ids for s in base] == [s.chunk_ids for s in varied]
    assert [s.doc_type for s in base] == [s.doc_type for s in varied]
    assert [s.axes["reasoning"] for s in base] == [s.axes["reasoning"] for s in varied]


def test_scenarios_are_unique_despite_a_small_chunk_pool(chunks):
    scenarios = sample(chunks, recipe(n=200))
    assert len({s.scenario_hash for s in scenarios}) == 200


def test_axis_keys_are_identical_across_every_scenario(chunks):
    """A stable axis key set is what keeps the snapshot schema fixed."""
    scenarios = sample(chunks, recipe())
    keysets = {tuple(sorted(s.axes)) for s in scenarios}
    assert len(keysets) == 1


def test_mixture_weights_are_approximately_respected(chunks):
    scenarios = sample(chunks, recipe(n=2000, doc_type={"difficult_advice": 0.8, "trait_conflict": 0.2}))
    share = sum(1 for s in scenarios if s.doc_type == "difficult_advice") / len(scenarios)
    assert 0.75 < share < 0.85
