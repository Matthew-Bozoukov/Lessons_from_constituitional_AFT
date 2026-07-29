# ABOUTME: Tests for the stages added from the GDM and Teaching Claude Why pipelines:
# ABOUTME: scenario planning, generation strategies, pattern discovery, and cache policy.

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from synthdoc import load_config, load_snapshot, run_pipeline
from synthdoc.config import ConfigError
from synthdoc.control import loader
from synthdoc.core.cache import SCOPES, Cache, CacheConfig
from synthdoc.pipeline import _stage_names

FULL = "smoke_full.yaml"


def cfg_for(tmp_path, config=FULL, **overrides):
    """Load a config redirected into a temporary directory."""
    base = {
        "output_dir": str(tmp_path / "runs"),
        "cache.dir": str(tmp_path / "cache"),
        "report.enabled": False,
        "name": None,
    }
    base.update(overrides)
    return load_config(config, base)


# --- stage sequencing ---------------------------------------------------------


def test_stage_names_insert_planning_first():
    assert _stage_names(2, planning=False) == [
        "stage_00_generated",
        "stage_01_revised",
        "stage_02_revised",
        "stage_03_filtered",
    ]
    assert _stage_names(2, planning=True) == [
        "stage_00_planned",
        "stage_01_generated",
        "stage_02_revised",
        "stage_03_revised",
        "stage_04_filtered",
    ]


def test_stage_names_with_no_revision():
    assert _stage_names(0, planning=False) == ["stage_00_generated", "stage_01_filtered"]
    assert _stage_names(0, planning=True) == [
        "stage_00_planned",
        "stage_01_generated",
        "stage_02_filtered",
    ]


# --- planning -----------------------------------------------------------------


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    """A completed run with planning, draft_then_align, and every filter."""
    root = tmp_path_factory.mktemp("planned")
    cfg = load_config(
        FULL,
        {
            "output_dir": str(root / "runs"),
            "cache.dir": str(root / "cache"),
            "name": None,
        },
    )
    return run_pipeline(cfg, n=8, run_id="planned_run", progress=False)


def test_planning_is_its_own_stage(planned):
    assert planned.stages[0] == "stage_00_planned"
    assert planned.stages[1] == "stage_01_generated"


def test_plan_stage_snapshot_has_plans_but_no_turns(planned):
    table = pq.read_table(planned.run_dir / "stage_00_planned.parquet").to_pydict()
    assert all(p for p in table["plan"])
    assert all(k == "what_how_why" for k in table["plan_kind"])
    assert sum(table["n_turns"]) == 0


def test_generation_stage_fills_turns(planned):
    table = pq.read_table(planned.run_dir / "stage_01_generated.parquet").to_pydict()
    assert sum(table["n_turns"]) > 0


def test_plan_survives_to_the_final_stage(planned):
    docs = load_snapshot(planned.run_dir / f"{planned.stages[-1]}.jsonl")
    assert all(d.plan for d in docs)
    assert all({"what", "how", "why"} <= set(d.plan) for d in docs)


def test_planning_stage_does_not_report_itself_as_failed(planned):
    """Documents legitimately have no turns yet, which must not read as errors."""
    counts = planned.counts["stage_00_planned"]
    assert counts["n_error"] == 0
    assert counts["n_ok"] == counts["n"]
    assert counts["n_with_turns"] == 0
    assert counts["n_planned"] == counts["n"]


def test_planner_model_is_independently_configurable(planned):
    docs = load_snapshot(planned.run_dir / "stage_00_planned.jsonl")
    kinds = {r.kind: r.model for d in docs for r in d.lineage}
    assert kinds["plan:what_how_why"] == "echo-planner"


def test_schema_is_identical_across_all_five_stages(planned):
    schemas = [pq.read_schema(planned.run_dir / f"{s}.parquet") for s in planned.stages]
    assert all(s.equals(schemas[0]) for s in schemas)


def test_doc_id_is_stable_from_planning_through_filtering(planned):
    first = pq.read_table(planned.run_dir / planned.stages[0] + ".parquet").to_pydict() \
        if False else pq.read_table(planned.run_dir / f"{planned.stages[0]}.parquet").to_pydict()
    last = pq.read_table(planned.run_dir / f"{planned.stages[-1]}.parquet").to_pydict()
    assert set(first["doc_id"]) == set(last["doc_id"])


def test_planning_can_be_disabled(tmp_path):
    cfg = cfg_for(tmp_path, planning={"enabled": False}, **{"generation.strategy": "single_pass"})
    result = run_pipeline(cfg, n=4, run_id="unplanned", progress=False)
    assert result.stages[0] == "stage_00_generated"
    assert all(not d.plan for d in result.corpus)


# --- generation strategies ----------------------------------------------------


def test_draft_then_align_makes_both_calls(planned):
    docs = load_snapshot(planned.run_dir / "stage_01_generated.jsonl")
    kinds = [r.kind for r in docs[0].lineage]
    # Default draft_context is spec_in_system, faithful to GDM's description.
    assert "draft:spec_in_system" in kinds and "align" in kinds


def test_draft_context_puts_the_spec_in_the_drafting_system_prompt(tmp_path):
    """GDM draft with the trait in the system prompt; drafting blind is our variant."""
    from synthdoc.plugins.generators import GenerationContext, seed_document
    from synthdoc.plugins.strategies import DraftThenAlign
    from synthdoc.pipeline import build_scenarios

    cfg = cfg_for(tmp_path)
    scenario = build_scenarios(cfg, 1)[0][0]
    doc = seed_document(scenario, "t")
    doc.plan = {"user_prompt": "help me with this"}

    def system_for(context):
        ctx = GenerationContext(llm=None, model="m", strategy_params={"draft_context": context})
        strategy = DraftThenAlign(ctx)
        entry = __import__("synthdoc.control", fromlist=["loader"]).loader.entry(
            "strategies", "draft_then_align"
        )
        return entry[strategy.DRAFT_CONTEXTS[context]]["system"]

    assert "{{ c.text }}" in system_for("spec_in_system")
    assert "{{ c.text }}" not in system_for("no_spec")


def test_unknown_draft_context_is_reported_on_the_document(tmp_path):
    cfg = cfg_for(tmp_path, **{"generation.strategy_params": {"draft_context": "telepathy"}})
    result = run_pipeline(cfg, n=2, run_id="baddraft", progress=False)
    docs = load_snapshot(result.run_dir / "stage_01_generated.jsonl")
    assert all("draft_context" in d.error for d in docs)


def test_draft_then_align_requires_planning(tmp_path):
    """Without a plan there is no user turn to draft against; fail at config time."""
    with pytest.raises(ConfigError, match="planning.enabled"):
        cfg_for(tmp_path, **{"generation.strategy": "draft_then_align",
                             "planning": {"enabled": False}})


def test_best_of_n_generates_candidates_and_selects(tmp_path):
    cfg = cfg_for(
        tmp_path,
        **{"generation.strategy": "best_of_n",
           "generation.strategy_params": {"n": 3, "selector": "judge"}},
    )
    result = run_pipeline(cfg, n=4, run_id="bon", progress=False)
    doc = result.corpus[0]
    kinds = [r.kind for r in doc.lineage]
    assert sum(1 for k in kinds if k.startswith("generate:cand")) == 3
    assert any(k.startswith("best_of_n:select") for k in kinds)
    assert doc.ok


def test_best_of_n_accounts_for_discarded_candidates(tmp_path):
    """Dropping the unselected candidates' cost would under-report this strategy n-fold."""
    cfg = cfg_for(
        tmp_path,
        **{"generation.strategy": "best_of_n",
           "generation.strategy_params": {"n": 3, "selector": "judge"},
           "pricing": {"echo-gen": {"in": 1000.0, "out": 1000.0}}},
    )
    result = run_pipeline(cfg, n=2, run_id="bon_cost", progress=False)
    doc = result.corpus[0]
    discarded = [r for r in doc.lineage if r.kind.endswith("(discarded)")]
    assert len(discarded) == 2
    assert all(r.cost_usd > 0 for r in discarded)
    assert doc.cost_usd_total >= sum(r.cost_usd for r in discarded)


def test_best_of_n_without_a_judge_skips_selection(tmp_path):
    cfg = cfg_for(
        tmp_path,
        **{"generation.strategy": "best_of_n",
           "generation.strategy_params": {"n": 2, "selector": "first_ok"}},
    )
    result = run_pipeline(cfg, n=4, run_id="bon_nojudge", progress=False)
    kinds = [r.kind for r in result.corpus[0].lineage]
    assert not any(k.startswith("best_of_n:select") for k in kinds)


def test_unknown_strategy_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="strategy"):
        cfg_for(tmp_path, **{"generation.strategy": "telepathy"})


# --- pattern scanning ---------------------------------------------------------


def test_pattern_scan_discovers_and_seeds_patterns(planned):
    summary = planned.manifest["filter_summaries"]["pattern_scan"]
    assert summary["seeded"] == len(loader.load_pack("patterns")["seed_patterns"])
    assert summary["discovered"] >= 1
    assert summary["n_patterns"] == summary["seeded"] + summary["discovered"]


def test_pattern_list_is_recorded_in_the_manifest(planned):
    patterns = planned.manifest["filter_summaries"]["pattern_scan"]["patterns"]
    names = {p["name"] for p in patterns}
    assert "conversion_arc" in names and "bluf" in names
    assert all(p["description"] for p in patterns)


def test_pattern_scores_are_written_to_every_document(planned):
    assert all("pattern_matches" in d.filter_scores for d in planned.corpus)


def test_pattern_scan_can_run_without_discovery(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg["filters"] = [
        {"kind": "pattern_scan", "model": "echo-rater", "discover": False,
         "use_seed_patterns": True, "max_patterns": 2}
    ]
    result = run_pipeline(cfg, n=4, run_id="seeded_only", progress=False)
    summary = result.manifest["filter_summaries"]["pattern_scan"]
    assert summary["discovered"] == 0
    assert summary["seeded"] > 0


# --- export -------------------------------------------------------------------


def test_strip_system_removes_system_turns(planned):
    rows = [
        json.loads(line)
        for line in Path(planned.exports["main"]).read_text().splitlines()
    ]
    assert rows
    assert not any(m["role"] == "system" for r in rows for m in r["messages"])


def test_baseline_mixing_interleaves_an_external_dataset(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        "\n".join(json.dumps({"messages": [{"role": "user", "content": f"b{i}"}]})
                  for i in range(50))
        + "\n"
    )
    cfg = cfg_for(tmp_path, **{"export.baseline": {"path": str(baseline), "ratio": 1.0}})
    result = run_pipeline(cfg, n=6, run_id="mixed", progress=False)
    assert "mixed" in result.exports
    mixed = Path(result.exports["mixed"]).read_text().splitlines()
    ours = Path(result.exports["main"]).read_text().splitlines()
    assert len(mixed) == 2 * len(ours)


def test_missing_baseline_path_fails_loudly(tmp_path):
    cfg = cfg_for(tmp_path, **{"export.baseline": {"path": str(tmp_path / "nope.jsonl")}})
    with pytest.raises(FileNotFoundError, match="baseline"):
        run_pipeline(cfg, n=4, run_id="nobaseline", progress=False)


# --- cache policy -------------------------------------------------------------


def test_cache_config_defaults_to_every_scope():
    cfg = CacheConfig.from_config({"cache": {}})
    assert set(cfg.scope) == set(SCOPES)
    assert cfg.enabled is True


def test_cache_scope_accepts_shorthand():
    assert CacheConfig.from_config({"cache": {"scope": "all"}}).scope == SCOPES
    assert CacheConfig.from_config({"cache": {"scope": "none"}}).scope == ()
    assert CacheConfig.from_config({"cache": {"scope": "generate"}}).scope == ("generate",)


def test_unknown_cache_scope_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="scope"):
        cfg_for(tmp_path, **{"cache.scope": ["generate", "telepathy"]})


def test_scope_bypasses_uncached_call_sites(tmp_path):
    """Narrowing scope must skip the cache, not silently cache anyway."""
    cfg = cfg_for(tmp_path, **{"cache.scope": ["generate"], "resume": False})
    first = run_pipeline(cfg, n=4, run_id="scoped1", progress=False)
    second = run_pipeline(cfg, n=4, run_id="scoped2", progress=False)
    assert first.manifest["cache"]["bypassed"] > 0
    assert second.manifest["cache"]["hits"] > 0
    # Plan, revise and filter calls were never cached, so they are paid for again.
    assert second.manifest["cache"]["bypassed"] == first.manifest["cache"]["bypassed"]


def test_disabled_cache_never_hits(tmp_path):
    cfg = cfg_for(tmp_path, **{"cache.enabled": False, "resume": False})
    run_pipeline(cfg, n=4, run_id="nocache1", progress=False)
    second = run_pipeline(cfg, n=4, run_id="nocache2", progress=False)
    assert second.manifest["cache"]["hits"] == 0


def test_namespace_isolates_runs_sharing_a_directory(tmp_path):
    cfg_a = cfg_for(tmp_path, **{"cache.namespace": "a", "resume": False})
    cfg_b = cfg_for(tmp_path, **{"cache.namespace": "b", "resume": False})
    run_pipeline(cfg_a, n=4, run_id="ns_a", progress=False)
    result_b = run_pipeline(cfg_b, n=4, run_id="ns_b", progress=False)
    assert result_b.manifest["cache"]["hits"] == 0


def test_cache_is_invariant_to_run_id(tmp_path):
    """Every cache key must be content-addressed, or sweep arms pay twice."""
    cfg = cfg_for(tmp_path, **{"resume": False})
    run_pipeline(cfg, n=4, run_id="first", progress=False)
    second = run_pipeline(cfg, n=4, run_id="totally_different", progress=False)
    assert second.manifest["cache"]["misses"] == 0
    assert second.manifest["cache"]["hits"] > 0


def test_max_bytes_evicts_oldest_entries(tmp_path):
    cache = Cache(CacheConfig(dir=str(tmp_path / "c"), max_bytes=400))
    for i in range(40):
        cache.put(f"{i:032x}", {"payload": "x" * 100})
    cache.prune()
    assert cache.size_bytes() <= 400
    assert cache.evicted > 0


def test_embeddings_dir_follows_the_cache_dir(tmp_path):
    cfg = CacheConfig.from_config({"cache": {"dir": str(tmp_path / "c")}})
    assert cfg.embeddings_path() == str(tmp_path / "c" / "embeddings")
    off = CacheConfig.from_config({"cache": {"dir": str(tmp_path), "embeddings": False}})
    assert off.embeddings_path() is None


def test_cache_stats_report_the_policy(planned):
    stats = planned.manifest["cache"]
    assert set(stats["scope"]) == set(SCOPES)
    assert stats["enabled"] is True
    assert stats["size_bytes"] > 0
