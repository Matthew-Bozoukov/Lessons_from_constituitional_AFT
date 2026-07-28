# ABOUTME: End-to-end pipeline tests on the offline echo provider: stage snapshots,
# ABOUTME: schema stability, doc_id joins, cache reuse, filter retention, and sweeps.

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from synthdoc import load_config, load_snapshot, run_pipeline, run_sweep
from synthdoc.pipeline import BudgetExceeded


@pytest.fixture
def cfg(tmp_path):
    """A smoke config redirected into a temporary directory."""
    return load_config(
        "smoke.yaml",
        {
            "output_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
            "report.plot": False,
        },
    )


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """One completed run shared by the read-only assertions below."""
    root = tmp_path_factory.mktemp("shared")
    cfg = load_config(
        "smoke.yaml",
        {"output_dir": str(root / "runs"), "cache_dir": str(root / "cache")},
    )
    return run_pipeline(cfg, n=10, run_id="shared_run", progress=False)


# --- stages and snapshots -----------------------------------------------------


def test_stage_sequence_matches_the_revision_dose(shared):
    assert shared.stages == ["stage_00_generated", "stage_01_revised", "stage_02_filtered"]


def test_every_stage_writes_a_complete_snapshot(shared):
    for stage in shared.stages:
        table = pq.read_table(shared.run_dir / f"{stage}.parquet")
        assert table.num_rows == 10, stage


def test_schema_is_identical_across_stages(shared):
    """Identical schemas are what make stage-over-stage comparison a groupby."""
    schemas = [pq.read_schema(shared.run_dir / f"{s}.parquet") for s in shared.stages]
    assert all(s.equals(schemas[0]) for s in schemas)


def test_filter_columns_exist_before_the_filter_stage(shared):
    """stage_00 must already carry the filter columns, or splits stop being comparable."""
    schema = pq.read_schema(shared.run_dir / "stage_00_generated.parquet")
    names = [f.name for f in schema.field("filter_scores").type]
    assert "autorater_overall" in names and "dedup_max_sim" in names


def test_doc_id_joins_stages_row_for_row(shared):
    first = pq.read_table(shared.run_dir / "stage_00_generated.parquet").to_pydict()
    last = pq.read_table(shared.run_dir / "stage_02_filtered.parquet").to_pydict()
    assert set(first["doc_id"]) == set(last["doc_id"])
    assert set(first["scenario_hash"]) == set(last["scenario_hash"])


def test_input_doc_id_points_at_the_previous_stage_row(shared):
    docs = load_snapshot(shared.run_dir / "stage_01_revised.jsonl")
    assert all(d.input_doc_id == d.doc_id for d in docs)


def test_lineage_records_every_model_call(shared):
    docs = load_snapshot(shared.run_dir / "stage_02_filtered.jsonl")
    doc = next(d for d in docs if d.ok)
    kinds = [r.kind for r in doc.lineage]
    assert kinds[0] == "generate"
    assert any(k == "critique_rewrite" for k in kinds)
    assert any(k.startswith("autorater") for k in kinds)
    assert all(r.prompt_hash for r in doc.lineage)


def test_dropped_documents_are_retained_with_a_verdict(shared):
    """The filter's own effect must stay inspectable, so nothing is deleted."""
    table = pq.read_table(shared.run_dir / "stage_02_filtered.parquet").to_pydict()
    assert len(table["doc_id"]) == 10
    assert set(table["filter_verdict"]) <= {"keep", "drop"}


def test_manifest_records_provenance_and_thresholds(shared):
    manifest = json.loads((shared.run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "shared_run"
    assert manifest["spec"]["spec_sha"]
    assert manifest["git_sha"]
    assert "embedding_dedup" in manifest["thresholds"]
    assert manifest["agreement"]["autorater"]["n_raters"] == 2


def test_coverage_report_and_index_are_emitted(shared):
    assert (shared.run_dir / "coverage_report.md").exists()
    assert (shared.run_dir / "coverage_index.parquet").exists()
    text = (shared.run_dir / "coverage_report.md").read_text()
    assert "Spec coverage" in text and "chunk_id x doc_type" in text


def test_coverage_heatmap_file_is_written(shared):
    assert (shared.run_dir / "coverage_heatmap.png").stat().st_size > 0


def test_exports_are_written(shared):
    main = Path(shared.exports["main"])
    assert main.exists()
    rows = [json.loads(line) for line in main.read_text().splitlines()]
    assert rows and all("messages" in r for r in rows)


def test_shard_assignment_is_uniform_and_deterministic():
    """Shard routing is a pure function of doc_id, so two runs shard identically."""
    from synthdoc.plugins.exporters import _assignment

    ids = [f"doc{i:05d}" for i in range(4000)]
    picked = [i for i in ids if _assignment(i, "pretrain_shard") < 0.4]
    assert 0.37 < len(picked) / len(ids) < 0.43
    assert _assignment(ids[0], "pretrain_shard") == _assignment(ids[0], "pretrain_shard")
    assert _assignment(ids[0], "pretrain_shard") != _assignment(ids[0], "other_shard")


def test_export_keeps_reasoning_in_its_own_field(shared):
    rows = [json.loads(line) for line in Path(shared.exports["main"]).read_text().splitlines()]
    assistant = [m for r in rows for m in r["messages"] if m["role"] == "assistant"]
    assert any("reasoning_content" in m for m in assistant)


# --- determinism, caching, resume --------------------------------------------


def test_two_runs_with_the_same_seed_produce_the_same_scenarios(tmp_path):
    cfg = load_config(
        "smoke.yaml",
        {"output_dir": str(tmp_path / "r"), "cache_dir": str(tmp_path / "c"), "report.enabled": False},
    )
    a = run_pipeline(cfg, n=6, run_id="a", progress=False)
    b = run_pipeline(cfg, n=6, run_id="b", progress=False)
    assert [d.scenario.scenario_hash for d in a.corpus] == [
        d.scenario.scenario_hash for d in b.corpus
    ]


def test_second_run_is_served_from_cache(tmp_path):
    """Cache hits are what make revision dose sweeps affordable."""
    cfg = load_config(
        "smoke.yaml",
        {"output_dir": str(tmp_path / "r"), "cache_dir": str(tmp_path / "c"), "report.enabled": False},
    )
    first = run_pipeline(cfg, n=6, run_id="first", progress=False)
    second = run_pipeline(cfg, n=6, run_id="second", progress=False)
    assert first.manifest["cache"]["hits"] == 0
    assert second.manifest["cache"]["hits"] > 0
    assert second.manifest["cache"]["misses"] == 0


def test_resume_reloads_a_completed_stage(tmp_path):
    cfg = load_config(
        "smoke.yaml",
        {
            "output_dir": str(tmp_path / "r"),
            "cache_dir": str(tmp_path / "c"),
            "resume": True,
            "report.enabled": False,
        },
    )
    run_pipeline(cfg, n=6, run_id="resumed", progress=False)
    again = run_pipeline(cfg, n=6, run_id="resumed", progress=False)
    assert again.counts["stage_00_generated"]["n"] == 6


def test_snapshot_roundtrip_preserves_the_document(shared):
    docs = load_snapshot(shared.run_dir / "stage_02_filtered.jsonl")
    doc = docs[0]
    assert doc.scenario.chunk_ids
    assert doc.scenario.axes
    assert doc.stage_name == "stage_02_filtered"


def test_budget_stops_the_run(tmp_path, monkeypatch):
    cfg = load_config(
        "smoke.yaml",
        {
            "output_dir": str(tmp_path / "r"),
            "cache_dir": str(tmp_path / "c"),
            "budget_usd": 0.0000001,
            "pricing": {"echo-gen": {"in": 1000.0, "out": 1000.0}},
            "report.enabled": False,
        },
    )
    with pytest.raises(BudgetExceeded):
        run_pipeline(cfg, n=4, run_id="broke", progress=False)


# --- sweeps -------------------------------------------------------------------


def test_sweep_over_a_post_sampling_axis_is_fully_paired(tmp_path):
    sweep_cfg = {
        "base": "smoke.yaml",
        "axis": "generation.model",
        "id": "paired_sweep",
        "arms": [{"name": "a", "value": "echo-a"}, {"name": "b", "value": "echo-b"}],
    }
    result = run_sweep(sweep_cfg, n=6, output_dir=tmp_path / "sweeps", dry_run=True)
    assert result.pairing["paired"] is True
    assert result.pairing["shared_fraction"] == 1.0
    assert Path(result.report_path).exists()


def test_sweep_over_a_sampler_axis_reports_partial_pairing(tmp_path):
    sweep_cfg = {
        "base": "smoke.yaml",
        "axis": "recipe.grouping",
        "id": "unpaired_sweep",
        "base_overrides": {"recipe.chunks_per_example": {2: 1.0}},
        "arms": [
            {"name": "random", "value": {"random": 1.0}},
            {"name": "adjacent", "value": {"adjacent": 1.0}},
        ],
    }
    result = run_sweep(sweep_cfg, n=6, output_dir=tmp_path / "sweeps", dry_run=True)
    assert result.pairing["paired"] is False
    assert "partially paired" in result.pairing["note"]


def test_sweep_runs_every_arm_and_writes_a_report(tmp_path):
    sweep_cfg = {
        "base": "smoke.yaml",
        "axis": "generation.model",
        "id": "real_sweep",
        "arms": [{"name": "a", "value": "echo-a"}, {"name": "b", "value": "echo-b"}],
    }
    result = run_sweep(sweep_cfg, n=4, output_dir=tmp_path / "sweeps")
    assert set(result.runs) == {"a", "b"}
    report = Path(result.report_path).read_text()
    assert "Arm comparison" in report and "echo-a" in report
    a_hashes = {d.scenario.scenario_hash for d in result.runs["a"].corpus}
    b_hashes = {d.scenario.scenario_hash for d in result.runs["b"].corpus}
    assert a_hashes == b_hashes  # joinable row for row on scenario_hash
