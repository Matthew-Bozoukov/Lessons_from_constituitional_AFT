# ABOUTME: Offline tests for constitution chunking and grouping -- every granularity x
# ABOUTME: strategy. Run: uv run pytest tests/test_constitution_chunking.py -q

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data.synth.constitution import (
    CHUNKINGS,
    DEFAULT_CHUNKING,
    GRANULARITIES,
    STRATEGIES,
    Trait,
    chunk,
    full_text,
    group,
    resolve_chunking,
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


# --- the named methods a dataset config picks from -----------------------------------


def test_default_chunking_is_the_original_recipe():
    """The default must stay the method every existing corpus was generated with, or a
    config that never mentions chunking would silently produce different data."""
    assert DEFAULT_CHUNKING == "principle"
    units, style = units_from_config({"constitution": MID})
    traits, seg_style = segment(MID)
    assert [u.as_trait() for u in units] == traits
    assert style == seg_style
    # Naming it explicitly must be identical to leaving it out.
    named, _ = units_from_config({"constitution": MID, "chunking": "principle"})
    assert [u.as_dict() for u in named] == [u.as_dict() for u in units]


@pytest.mark.parametrize("name", sorted(CHUNKINGS))
def test_every_registered_method_is_well_formed(name):
    spec = CHUNKINGS[name]
    assert spec.name == name, "CHUNKINGS key must match the method's own name"
    assert spec.granularity in GRANULARITIES
    assert spec.strategy in STRATEGIES
    assert spec.size >= 1 and spec.summary


@pytest.mark.parametrize("name", sorted(CHUNKINGS))
def test_every_registered_method_builds_units(name):
    """Stage 1 is deterministic and free, so every method is provable before any spend."""
    units, style = units_from_config({"constitution": MID, "chunking": name})
    assert units and style
    spec = CHUNKINGS[name]
    assert all(u.granularity == spec.granularity for u in units)
    assert all(u.grouping_strategy == spec.strategy for u in units)


def test_registered_methods_produce_distinct_unit_sets():
    """Two methods yielding identical units would not be a distinct choice."""
    seen = {name: tuple(u.unit_id for u in
                        units_from_config({"constitution": MID, "chunking": name})[0])
            for name in CHUNKINGS}
    assert len(set(seen.values())) == len(seen), \
        f"methods collapsed onto the same unit set: {seen}"


def test_unknown_chunking_name_lists_the_options():
    # A silent fallback to the default would generate the wrong corpus without saying so.
    with pytest.raises(ValueError, match="unknown chunking"):
        units_from_config({"constitution": MID, "chunking": "bullets"})
    with pytest.raises(ValueError, match="unknown chunking"):
        resolve_chunking("per_sentence")


def test_chunking_must_be_a_name_not_an_inline_block():
    """Settings live with the method so a manifest records which recipe ran, rather than
    an anonymous bag of knobs no one can look up later."""
    with pytest.raises(ValueError, match="takes the NAME"):
        units_from_config({"constitution": MID,
                           "chunking": {"granularity": "bullet"}})


@pytest.mark.parametrize("name", sorted(CHUNKINGS))
def test_dataset_config_can_select_any_method(name):
    """The flag path end to end: a real dataset config plus `chunking: <name>` builds
    its units and prices, with corpus size held fixed by total_scenarios."""
    import yaml

    from src.data.synth.stage_runtime import Ctx
    from src.data.synth.stage_operators import op_segment, scenario_batches
    from src.data.synth.pipeline import n_examples, n_units

    cfg = yaml.safe_load(open("configs/data/synth/difficult_advice.yaml"))
    cfg["chunking"] = name
    cfg.pop("n_traits")            # derived from the method, not declared
    cfg.pop("scenarios_per_trait")  # per unit; swap for a fixed budget
    cfg["total_scenarios"] = 180

    ctx = Ctx(cfg=cfg, usage=None, workers=1, run_dir=None, smoke=False, vars={})
    rows = op_segment({"name": "traits", "kind": "segment"}, cfg).fn(ctx, [], None)
    assert rows and ctx.vars["style_guidance"]
    for r in rows:
        assert r["trait_id"] and r["text"] and r["n_chunks"] >= 1
        assert Trait.from_record(r).text == r["text"]

    # Size-matching: `scenarios_per_trait` is PER UNIT, so it would hand `bullet`
    # (45 units) ~45x the data of `whole` (1 unit) -- a data-scaling curve wearing a
    # chunking comparison's clothes. `total_scenarios` splits a fixed budget instead.
    assert sum(n for _t, _b, n in scenario_batches(n_units(cfg), cfg)) == 180
    assert n_examples(cfg) == 180


def test_shipped_dataset_configs_declare_the_default():
    """difficult_advice and self_reflection are the corpora of record: their chunking
    must be stated outright, not inherited from a default that could later move."""
    import yaml

    for path in ("configs/data/synth/difficult_advice.yaml",
                 "configs/data/synth/self_reflection.yaml"):
        cfg = yaml.safe_load(open(path))
        assert cfg.get("chunking") == DEFAULT_CHUNKING, path


def test_n_traits_hint_must_match_the_chosen_method():
    """difficult_advice declares n_traits: 9 for the default. Any other method changes
    the unit count, and a stale hint would misprice the run."""
    import yaml

    from src.data.synth.pipeline import n_units

    cfg = yaml.safe_load(open("configs/data/synth/difficult_advice.yaml"))
    assert n_units(cfg) == cfg["n_traits"] == 9
    with pytest.raises(AssertionError, match="n_traits"):
        n_units({**cfg, "chunking": "bullet"})


def test_unit_provenance_reaches_the_generated_records_and_the_export():
    """Which unit a document came from, and how that unit was cut and grouped, must be
    readable from the DOCUMENT -- not only from the stage-1 snapshot.

    Stage 2 builds a fresh record from the Trait interface, so without this the chunking
    provenance would die there and every downstream consumer (the metadata export, the
    corpus checks, `balance_by`) would have to join back to stage 1 to recover it.
    """
    import yaml

    from src.data.synth.constitution import UNIT_PROVENANCE, units_from_config
    from src.data.synth.stage_operators import op_chat_export

    cfg = yaml.safe_load(open("configs/data/synth/difficult_advice.yaml"))
    units, _ = units_from_config(cfg)
    stage1 = [u.as_dict() for u in units]
    assert all(k in stage1[0] for k in UNIT_PROVENANCE)

    # A scenario record as op_scenarios now builds it: Trait fields + provenance.
    prov = {r["trait_id"]: {k: r[k] for k in UNIT_PROVENANCE if k in r} for r in stage1}
    scenario = {"scenario_id": "t1_b00_s000", "trait_id": "t1",
                "trait_name": units[0].name, "trait_text": units[0].text,
                "domain": "d", "situation": "s", "shortcut": "sc",
                "system": "sys", "user": "u", "response": "r", "reasoning": "why",
                **prov["t1"]}

    sft_stage = next(s for s in cfg["stages"] if s["kind"] == "chat_export")
    assert set(UNIT_PROVENANCE) <= set(sft_stage["metadata"]), \
        "the export must carry the provenance, or the corpus checks cannot see it"

    class _Ctx:
        cfg: dict = {}
        vars: dict = {}

    exported = op_chat_export(sft_stage, cfg).fn(_Ctx(), [scenario], None)
    meta = exported[0]["metadata"]
    assert meta["chunk_ids"] == ["t1"]
    assert meta["n_chunks"] == 1
    assert meta["granularity"] == "principle"
    assert meta["grouping_strategy"] == "single"
