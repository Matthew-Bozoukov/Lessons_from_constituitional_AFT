# ABOUTME: Tests for saving named corpora and ablating arbitrary axes: config inheritance,
# ABOUTME: the axis catalogue, the corpus catalogue, and paired corpus comparison.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from synthdoc import load_config, run_pipeline
from synthdoc.ablations import catalog
from synthdoc.config import CONTROL_CONFIGS, ConfigError
from synthdoc.control import loader
from synthdoc.core.specs import available_specs, load_spec
from synthdoc.corpora import compare, format_index, load_index, summarize
from synthdoc.snapshots import SnapshotConfig, SnapshotWriter

CORPORA_DIR = CONTROL_CONFIGS / "corpora"


def write(tmp_path: Path, name: str, data: dict) -> str:
    """Write a YAML config and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


# --- spec resolution ----------------------------------------------------------


def test_specs_resolve_by_id_alone():
    """`axis: spec.id` sweeps need id-only resolution, including via index.yaml."""
    ids = available_specs()
    assert "demo_spec" in ids
    assert "claude_constitution_principles" in ids
    for spec_id in ids:
        assert load_spec(spec_id).text.strip()


def test_indexed_spec_loads_from_outside_the_control_dir():
    spec = load_spec("claude_constitution_principles")
    assert "docs/claude_constitution_principles.md" in spec.path
    assert spec.sha


def test_unknown_spec_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="demo_spec"):
        load_spec("no_such_spec")


# --- config inheritance -------------------------------------------------------


def test_extends_inherits_the_parent(tmp_path):
    path = write(tmp_path, "child.yaml", {"extends": "smoke.yaml", "seed": 99})
    cfg = load_config(path)
    assert cfg["seed"] == 99
    assert cfg["spec"]["id"] == "demo_spec"          # inherited
    assert cfg["llm"]["provider"] == "echo"          # inherited


def test_extends_replaces_mixtures_rather_than_merging(tmp_path):
    """A 100% single-type corpus must not silently retain the parent's other types."""
    path = write(
        tmp_path,
        "child.yaml",
        {"extends": "smoke.yaml", "recipe": {"doc_type": {"trait_conflict": 1.0}}},
    )
    cfg = load_config(path)
    assert cfg["recipe"]["doc_type"] == {"trait_conflict": 1.0}
    assert cfg["recipe"]["n"] == 12                  # other recipe keys still inherited


def test_extends_deep_merges_non_mixture_blocks(tmp_path):
    path = write(tmp_path, "child.yaml", {"extends": "smoke.yaml", "generation": {"temperature": 0.4}})
    cfg = load_config(path)
    assert cfg["generation"]["temperature"] == 0.4
    assert cfg["generation"]["model"] == "echo-gen"   # sibling key survives


def test_extends_merges_grouping_params_but_not_grouping(tmp_path):
    path = write(
        tmp_path,
        "child.yaml",
        {
            "extends": "smoke.yaml",
            "recipe": {
                "grouping": {"random": 1.0},
                "grouping_params": {"semantic": {"min_similarity": 0.1}},
            },
        },
    )
    cfg = load_config(path)
    assert cfg["recipe"]["grouping"] == {"random": 1.0}
    assert cfg["recipe"]["grouping_params"]["adjacent"] == {"same_section_only": False}
    assert cfg["recipe"]["grouping_params"]["semantic"]["min_similarity"] == 0.1


def test_extends_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text(f"extends: {tmp_path / 'a.yaml'}\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(str(tmp_path / "a.yaml"))


def test_name_becomes_a_stable_run_id(tmp_path):
    from synthdoc.config import make_run_id

    cfg = load_config(write(tmp_path, "c.yaml", {"extends": "smoke.yaml", "name": "my_corpus"}))
    assert make_run_id(cfg) == "my_corpus"
    assert make_run_id(cfg) == make_run_id(cfg)      # no timestamp drift


def test_unnamed_runs_get_distinct_ids():
    from synthdoc.config import make_run_id

    cfg = load_config("smoke.yaml")
    assert make_run_id(cfg) != "demo_spec"
    assert cfg["spec"]["id"] in make_run_id(cfg)


# --- shipped corpus presets ---------------------------------------------------


def test_every_corpus_preset_is_valid():
    presets = sorted(CORPORA_DIR.glob("*.yaml"))
    assert presets, "no corpus presets found"
    for preset in presets:
        cfg = load_config(str(preset))
        assert cfg["name"], f"{preset.name} has no name"


def test_all_multiturn_preset_is_actually_all_multiturn():
    cfg = load_config(str(CORPORA_DIR / "all_multiturn.yaml"))
    assert cfg["recipe"]["doc_type"] == {"multiturn_adversarial": 1.0}
    assert "single" not in cfg["recipe"]["turns"]


def test_single_spec_preset_pins_one_spec():
    cfg = load_config(str(CORPORA_DIR / "single_spec_constitution.yaml"))
    assert cfg["spec"]["id"] == "claude_constitution_principles"
    assert cfg["recipe"]["chunks_per_example"] == {1: 1.0}


def test_control_preset_disables_revision_and_filtering():
    cfg = load_config(str(CORPORA_DIR / "no_revision_control.yaml"))
    assert cfg["revision"] == []
    assert cfg["filters"] == []


# --- the axis catalogue -------------------------------------------------------


def test_catalogue_is_populated_and_well_formed():
    axes = catalog()
    assert len(axes) > 20
    for axis in axes:
        assert axis.key and axis.varies
        assert axis.pairing in ("paired", "unpaired")


def test_catalogue_covers_the_headline_questions():
    keys = {a.key for a in catalog()}
    for expected in (
        "spec.id",
        "recipe.doc_type",
        "recipe.grouping",
        "recipe.chunks_per_example",
        "generation.model",
        "generation.template",
        "revision",
        "filters",
        "seed",
    ):
        assert expected in keys, expected


def test_catalogue_values_track_the_live_registry():
    """The catalogue is generated, so a new plugin or prompt entry shows up for free."""
    by_key = {a.key: a for a in catalog()}
    assert set(by_key["recipe.doc_type"].values) == set(loader.declared_doc_types())
    assert set(by_key["generation.template"].values) == set(loader.load_pack("generation"))
    for axis_name in loader.declared_axes():
        assert f"recipe.{axis_name}" in by_key


def test_sampler_axes_are_marked_unpaired():
    by_key = {a.key: a for a in catalog()}
    assert by_key["recipe.grouping"].pairing == "unpaired"
    assert by_key["generation.model"].pairing == "paired"


# --- corpus catalogue and comparison -----------------------------------------


@pytest.fixture(scope="module")
def two_corpora(tmp_path_factory):
    """Two named corpora over identical scenarios, differing only in generator model."""
    root = tmp_path_factory.mktemp("corpora")
    runs = {}
    for name, model in (("baseline", "echo-a"), ("variant", "echo-b")):
        cfg = load_config(
            "smoke.yaml",
            {
                "output_dir": str(root / "runs"),
                "cache_dir": str(root / "cache"),
                "name": name,
                "generation.model": model,
                "report.plot": False,
            },
        )
        runs[name] = run_pipeline(cfg, n=8, progress=False)
    return runs


def test_named_corpus_lands_in_a_predictable_directory(two_corpora):
    assert two_corpora["baseline"].run_dir.name == "baseline"
    assert two_corpora["variant"].run_dir.name == "variant"


def test_corpora_are_catalogued(two_corpora):
    output_dir = Path(two_corpora["baseline"].config["output_dir"])
    entries = load_index(output_dir)
    names = {e["name"] for e in entries}
    assert {"baseline", "variant"} <= names
    listing = format_index(entries)
    assert "baseline" in listing and "echo-a" in listing


def test_catalogue_entry_records_the_distinguishing_fields(two_corpora):
    entry = summarize(two_corpora["variant"])
    assert entry["name"] == "variant"
    assert entry["generator_model"] == "echo-b"
    assert entry["spec_id"] == "demo_spec"
    assert entry["revision_dose"] == 1
    assert "difficult_advice" in entry["doc_type"]


def test_compare_reports_paired_deltas(two_corpora):
    result = compare(
        two_corpora["baseline"].run_dir, two_corpora["variant"].run_dir, "baseline", "variant"
    )
    assert result["n_shared_scenarios"] == 8
    assert result["paired"] is True
    assert "n_words" in result["paired_deltas"]
    assert result["paired_deltas"]["n_words"]["n"] == 8
    assert "by_doc_type" in result


def test_comparison_renders_as_markdown(two_corpora):
    from synthdoc.corpora import format_comparison

    text = format_comparison(
        compare(two_corpora["baseline"].run_dir, two_corpora["variant"].run_dir, "a", "b")
    )
    assert "Paired deltas" in text and "Marginals" in text


def test_compare_pairs_on_sample_index_when_the_recipe_differs(tmp_path):
    runs = {}
    for name, doc_types in (
        ("only_advice", {"difficult_advice": 1.0}),
        ("only_conflict", {"trait_conflict": 1.0}),
    ):
        cfg = load_config(
            "smoke.yaml",
            {
                "output_dir": str(tmp_path / "runs"),
                "cache_dir": str(tmp_path / "cache"),
                "name": name,
                "recipe.doc_type": doc_types,
                "report.enabled": False,
            },
        )
        runs[name] = run_pipeline(cfg, n=6, progress=False)

    # Changing the recipe means no scenario_hash can match, but example i in each arm
    # still differs only in the swept axis - so the comparison stays paired.
    result = compare(runs["only_advice"].run_dir, runs["only_conflict"].run_dir)
    assert result["n_shared_scenarios"] == 0
    assert result["join_key"] == "sample_index"
    assert result["paired_deltas"]["n_words"]["n"] == 6
    assert "sample_index" in result["note"]


def test_single_type_corpus_really_contains_one_type(tmp_path):
    cfg = load_config(
        "smoke.yaml",
        {
            "output_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
            "name": "multiturn_only",
            "recipe.doc_type": {"multiturn_adversarial": 1.0},
            "report.enabled": False,
        },
    )
    result = run_pipeline(cfg, n=10, progress=False)
    assert {d.scenario.doc_type for d in result.corpus} == {"multiturn_adversarial"}


# --- local cleanup guard ------------------------------------------------------


def test_cleanup_refuses_without_a_remote_copy(tmp_path):
    """Deleting local snapshots with no Hub copy would be unrecoverable data loss."""
    writer = SnapshotWriter(
        run_dir=tmp_path / "run",
        cfg=SnapshotConfig(backend="local"),
        run_id="r",
        axis_names=["tools"],
        filter_fields=["length_words"],
    )
    with pytest.raises(ValueError, match="huggingface"):
        writer.cleanup()


def test_cleanup_is_skipped_when_a_push_failed(tmp_path, recwarn):
    writer = SnapshotWriter(
        run_dir=tmp_path / "run",
        cfg=SnapshotConfig(backend="huggingface", org="someone"),
        run_id="r",
        axis_names=["tools"],
        filter_fields=["length_words"],
    )
    writer.push_errors.append("simulated failure")
    assert writer.cleanup() == []


def test_local_run_keeps_its_files(two_corpora):
    """With backend=local nothing is deleted, so smoke runs stay inspectable."""
    run_dir = two_corpora["baseline"].run_dir
    assert (run_dir / "stage_00_generated.parquet").exists()
    assert (run_dir / "stage_00_generated.jsonl").exists()
