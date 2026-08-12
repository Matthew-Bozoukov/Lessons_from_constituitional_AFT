# ABOUTME: Offline tests for constitution chunking and grouping -- every granularity x
# ABOUTME: strategy. Run: uv run pytest tests/test_constitution_chunking.py -q

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data.synth.constitution import (
    GRANULARITIES,
    STRATEGIES,
    Trait,
    chunk,
    full_text,
    group,
    segment,
    units_from_config,
)

# Every constitution in the repo, archived ones included: chunking must survive both
# markdown formats (v1's bolded list items and specgen's numbered H2 headings).
CONSTITUTIONS = sorted(str(p) for p in Path("constitutions").rglob("constitution.md"))
MID = "constitutions/claude_distilled_12_principles_mid/constitution.md"
V1 = "constitutions/archive/claude_distilled_8_principles_v1/constitution.md"

# Strips a chunk's leading "**Title**" so bodies can be compared across granularities.
_TITLE = re.compile(r"^\*\*.*?\*\*\s*", re.S)


def test_constitutions_are_discovered():
    # A glob that silently matched nothing would make every parametrised test vacuous.
    assert len(CONSTITUTIONS) >= 5, CONSTITUTIONS


@pytest.mark.parametrize("path", CONSTITUTIONS)
def test_principle_granularity_reproduces_segment(path):
    """`segment` IS chunk+group at principle/k=1 -- the guarantee that protects the
    production corpus. If this fails, an existing config would generate different data
    under an unchanged YAML."""
    chunks, style = chunk(path, granularity="principle")
    units = group(chunks, size=1, strategy="single")
    traits, seg_style = segment(path)
    assert style == seg_style
    assert [u.as_trait() for u in units] == traits
    assert [t.trait_id for t in traits] == [f"t{i}" for i in range(1, len(traits) + 1)]


@pytest.mark.parametrize("path", CONSTITUTIONS)
@pytest.mark.parametrize("granularity", ["paragraph", "bullet"])
def test_finer_granularity_loses_no_constitution_text(path, granularity):
    """A finer cut must carry exactly the same words per principle as `principle` does.
    Dropping stubs would make a granularity comparison secretly a coverage comparison."""
    base, _ = chunk(path, granularity="principle")
    fine, _ = chunk(path, granularity=granularity)
    assert {c.parent_id for c in fine} == {c.parent_id for c in base}
    for b in base:
        mine = [c for c in fine if c.parent_id == b.parent_id]
        assert mine, f"{b.chunk_id} produced no {granularity} chunks"
        # Every sub-chunk repeats the principle title, so it stays self-contained.
        assert all(_TITLE.match(c.text) for c in mine)
        assert sorted(w for c in mine for w in _TITLE.sub("", c.text).split()) == \
            sorted(_TITLE.sub("", b.text).split())


def test_bullet_is_strictly_finer_than_paragraph():
    # Guards the real bug found while building this: prose between bullets was not
    # being split on blank lines, so `bullet` silently collapsed onto `paragraph`.
    counts = {g: len(chunk(MID, granularity=g)[0])
              for g in ("whole", "principle", "paragraph", "bullet")}
    assert counts["whole"] == 1
    assert counts["whole"] < counts["principle"] < counts["paragraph"] < counts["bullet"]


def test_whole_granularity_is_the_entire_document():
    chunks, _ = chunk(MID, granularity="whole")
    assert len(chunks) == 1
    unit = group(chunks, size=1, strategy="single")[0]
    assert unit.unit_id == "all"
    assert unit.n_chunks == 1
    assert unit.text == full_text(MID)


@pytest.mark.parametrize("path", CONSTITUTIONS)
@pytest.mark.parametrize("granularity", GRANULARITIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_partitions_the_pool(path, granularity, strategy):
    """Each chunk lands in exactly one unit. Total constitution content is therefore
    identical across arms, so group size is never confounded with coverage."""
    chunks, _ = chunk(path, granularity=granularity)
    size = 1 if strategy == "single" else 2
    units = group(chunks, size=size, strategy=strategy, seed=0)
    assigned = sorted(cid for u in units for cid in u.chunk_ids)
    assert assigned == sorted(c.chunk_id for c in chunks)
    assert all(u.n_chunks == len(u.chunk_ids) for u in units)
    assert all(u.granularity == granularity for u in units)
    assert all(u.grouping_strategy == strategy for u in units)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_grouping_is_deterministic(strategy):
    chunks, _ = chunk(MID, granularity="principle")
    size = 1 if strategy == "single" else 3
    a = group(chunks, size=size, strategy=strategy, seed=0)
    b = group(chunks, size=size, strategy=strategy, seed=0)
    assert [u.as_dict() for u in a] == [u.as_dict() for u in b]


def test_random_grouping_responds_to_the_seed():
    chunks, _ = chunk(MID, granularity="principle")
    a = group(chunks, size=2, strategy="random", seed=0)
    b = group(chunks, size=2, strategy="random", seed=1)
    assert [u.unit_id for u in a] != [u.unit_id for u in b]


def test_unit_text_and_id_follow_document_order_not_draw_order():
    """Two arms that happen to select the same members must produce identical text,
    or a paired comparison across strategies is meaningless."""
    chunks, _ = chunk(MID, granularity="principle")
    by_id = {c.chunk_id: c for c in chunks}
    pair = [by_id["t7"], by_id["t3"]]
    unit = group(pair, size=2, strategy="adjacent")[0]
    assert unit.unit_id == "t3+t7"
    assert unit.chunk_ids == ("t3", "t7")
    assert unit.text == f"{by_id['t3'].text}\n\n{by_id['t7'].text}"
    assert unit.index == 3


def test_grouped_unit_carries_every_member_name_once():
    chunks, _ = chunk(MID, granularity="bullet")
    unit = group(chunks[:4], size=4, strategy="adjacent")[0]
    # Sub-chunks of one principle share a name; the unit name must not repeat it.
    assert unit.name == chunks[0].name
    assert unit.n_chunks == 4


def test_cluster_ids_are_stable_and_cover_the_pool():
    chunks, _ = chunk(MID, granularity="paragraph")
    units = group(chunks, strategy="cluster", n_clusters=4)
    assert [u.unit_id for u in units] == ["c1", "c2", "c3", "c4"]
    assert sum(u.n_chunks for u in units) == len(chunks)


@pytest.mark.parametrize("granularity,k", [("principle", 3), ("paragraph", 4),
                                           ("bullet", 4)])
def test_cluster_sizes_are_capacity_bounded(granularity, k):
    """Unconstrained k-means over hashed features collapses onto one huge cluster (the
    real failure this was built against: 3053 words vs 45). An arm whose units differ
    by that much in length is not comparable to any other arm."""
    chunks, _ = chunk(MID, granularity=granularity)
    units = group(chunks, strategy="cluster", n_clusters=k)
    cap = -(-len(chunks) // k)
    assert max(u.n_chunks for u in units) <= cap
    assert sum(u.n_chunks for u in units) == len(chunks)


def test_cluster_count_is_capped_by_the_pool_size():
    chunks, _ = chunk(V1, granularity="principle")
    units = group(chunks, strategy="cluster", n_clusters=99)
    assert 1 <= len(units) <= len(chunks)
    assert sum(u.n_chunks for u in units) == len(chunks)


def test_unit_record_is_a_superset_of_the_trait_record():
    """Stage-1 rows must stay readable as Traits, since every downstream operator
    reconstructs one from them."""
    chunks, _ = chunk(MID, granularity="principle")
    rec = group(chunks, size=2, strategy="adjacent")[0].as_dict()
    assert set(rec) == {"trait_id", "index", "name", "text", "chunk_ids",
                        "granularity", "grouping_strategy", "n_chunks"}
    trait = Trait.from_record(rec)
    assert trait.trait_id == rec["trait_id"] and trait.text == rec["text"]
    with pytest.raises(TypeError):
        Trait(**rec)  # the reason from_record exists


def test_bad_granularity_and_strategy_fail_loudly():
    with pytest.raises(ValueError, match="unknown granularity"):
        chunk(MID, granularity="sentence")
    chunks, _ = chunk(MID, granularity="principle")
    with pytest.raises(ValueError, match="unknown grouping strategy"):
        group(chunks, strategy="kmeans")
    with pytest.raises(ValueError, match="size 1"):
        group(chunks, size=2, strategy="single")
    with pytest.raises(ValueError, match="size must be"):
        group(chunks, size=0, strategy="adjacent")


def test_units_from_config_defaults_to_the_original_recipe():
    units, style = units_from_config({"constitution": MID})
    traits, seg_style = segment(MID)
    assert [u.as_trait() for u in units] == traits
    assert style == seg_style


def test_units_from_config_reads_the_chunking_block():
    units, _ = units_from_config({
        "constitution": MID,
        "chunking": {"granularity": "paragraph",
                     "group": {"size": 3, "strategy": "adjacent"}},
    })
    assert all(u.granularity == "paragraph" for u in units)
    assert all(u.grouping_strategy == "adjacent" for u in units)
    assert max(u.n_chunks for u in units) == 3


def test_units_from_config_rejects_a_typo():
    # A misspelled key silently generating the DEFAULT corpus is the failure mode here.
    with pytest.raises(ValueError, match="unknown key"):
        units_from_config({"constitution": MID, "chunking": {"granularty": "bullet"}})


# --- the study config: the arms must differ ONLY in chunking -------------------------

STUDY = "configs/data/synth/difficult_advice_chunking.yaml"

# The arms the config's header documents, as (label, override dict).
ARMS = [
    ("control_k1", {}),
    ("bullet", {"granularity": "bullet"}),
    ("paragraph", {"granularity": "paragraph"}),
    ("adjacent_k2", {"group": {"size": 2, "strategy": "adjacent"}}),
    ("random_k2", {"group": {"size": 2, "strategy": "random"}}),
    ("lexical_k2", {"group": {"size": 2, "strategy": "lexical"}}),
    ("whole", {"granularity": "whole"}),
    ("cluster", {"granularity": "paragraph", "group": {"strategy": "cluster"}}),
]


def _study_cfg(override: dict) -> dict:
    import yaml

    cfg = yaml.safe_load(open(STUDY))
    chunking = dict(cfg["chunking"])
    chunking.update({k: v for k, v in override.items() if k != "group"})
    chunking["group"] = {**chunking["group"], **override.get("group", {})}
    cfg["chunking"] = chunking
    return cfg


@pytest.mark.parametrize("label,override", ARMS)
def test_every_study_arm_is_size_matched(label, override):
    """THE experimental invariant. `scenarios_per_trait` is per unit, so it would give
    the bullet arm (45 units) ~45x the data of the whole arm (1 unit) and turn a
    chunking comparison into a data-scaling comparison. `total_scenarios` fixes the
    budget instead -- if this ever regresses, every arm comparison is invalid."""
    from src.data.synth.operators import scenario_batches
    from src.data.synth.pipeline import n_examples, n_units

    cfg = _study_cfg(override)
    assert "scenarios_per_trait" not in cfg, \
        "the study config must size by total_scenarios, not per unit"
    batches = scenario_batches(n_units(cfg), cfg)
    assert sum(n for _ti, _bi, n in batches) == cfg["total_scenarios"]
    assert n_examples(cfg) == cfg["total_scenarios"]


@pytest.mark.parametrize("label,override", ARMS)
def test_every_study_arm_builds_its_units_offline(label, override):
    """Stage 1 is deterministic and free, so every arm is provable before any spend."""
    from src.data.synth.core import Ctx
    from src.data.synth.operators import op_segment

    cfg = _study_cfg(override)
    ctx = Ctx(cfg=cfg, usage=None, workers=1, run_dir=None, smoke=False, vars={})
    rows = op_segment({"name": "traits", "kind": "segment"}, cfg).fn(ctx, [], None)
    assert rows and ctx.vars["style_guidance"]
    for r in rows:
        assert r["trait_id"] and r["text"] and r["n_chunks"] >= 1
        assert Trait.from_record(r).text == r["text"]


def test_study_arms_produce_distinct_unit_sets():
    """If two arms happened to yield identical units they would not be an ablation."""
    from src.data.synth.constitution import units_from_config

    seen = {label: tuple(u.unit_id for u in units_from_config(_study_cfg(o))[0])
            for label, o in ARMS}
    assert len(set(seen.values())) == len(seen), \
        f"arms collapsed onto the same unit set: {seen}"


def test_study_config_prompts_are_granularity_neutral():
    """The prompts must not say "one principle": every arm shares them, including the
    k=1 control, so that chunking is the only thing that varies."""
    import yaml

    cfg = yaml.safe_load(open(STUDY))
    prompts = [v for s in cfg["stages"] for v in (s.get("prompts") or {}).values()]
    assert prompts
    for text in prompts:
        low = text.lower()
        assert "one principle" not in low, text
        assert "<principle" not in low and "target_principle" not in low, text
