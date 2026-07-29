# ABOUTME: Tests that config and sweep validation fail loudly BEFORE a paid run starts,
# ABOUTME: including the review-blocking rule that a sweep may vary exactly one axis.

from __future__ import annotations

import pytest
import yaml

from synthdoc.config import ConfigError, filter_score_fields, load_config, load_config_dict
from synthdoc.control.loader import PromptError
from synthdoc.sweep import load_sweep

SMOKE = "smoke.yaml"


def write(tmp_path, name, data):
    """Write a YAML file and return its path as a string."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_smoke_config_loads_and_defaults_are_applied():
    cfg = load_config(SMOKE)
    assert cfg["spec"]["id"] == "demo_spec"
    assert cfg["report"]["enabled"] is True
    assert cfg["cache_enabled"] is True


def test_base_config_is_valid():
    """The reference config must stay runnable as prompts and plugins evolve."""
    cfg = load_config("base.yaml")
    assert cfg["recipe"]["n"] > 0
    assert len(cfg["revision"]) == 2


def test_overrides_replace_rather_than_merge():
    cfg = load_config(SMOKE, {"recipe.grouping": {"random": 1.0}})
    assert cfg["recipe"]["grouping"] == {"random": 1.0}


def test_dotted_override_reaches_nested_keys():
    cfg = load_config(SMOKE, {"generation.model": "other-model"})
    assert cfg["generation"]["model"] == "other-model"


def test_unknown_doc_type_is_rejected():
    with pytest.raises(ConfigError, match="doc_type"):
        load_config(SMOKE, {"recipe.doc_type": {"not_a_real_type": 1.0}})


def test_undeclared_axis_is_rejected():
    with pytest.raises(ConfigError, match="axes.yaml"):
        load_config(SMOKE, {"recipe.vibes": {"good": 1.0}})


def test_undeclared_axis_value_is_rejected():
    with pytest.raises(PromptError):
        load_config(SMOKE, {"recipe.tools": {"telekinesis": 1.0}})


def test_unknown_grouping_strategy_is_rejected():
    with pytest.raises(ConfigError, match="grouping"):
        load_config(SMOKE, {"recipe.grouping": {"vibes": 1.0}})


def test_unknown_filter_is_rejected():
    with pytest.raises(ConfigError, match="filters"):
        load_config(SMOKE, {"filters": [{"kind": "nonexistent"}]})


def test_unknown_reviser_kind_is_rejected():
    with pytest.raises(PromptError):
        load_config(SMOKE, {"revision": [{"kind": "vibe_pass", "model": "m"}]})


def test_bad_revision_context_is_rejected():
    with pytest.raises(ConfigError, match="context"):
        load_config(SMOKE, {"revision": [{"kind": "realism_pass", "context": "telepathy"}]})


def test_unknown_chunker_is_rejected():
    with pytest.raises(ConfigError, match="chunker"):
        load_config(SMOKE, {"spec.chunker.granularity": "vibes"})


def test_huggingface_backend_requires_an_org():
    with pytest.raises(ConfigError, match="snapshots.org"):
        load_config(SMOKE, {"snapshots.backend": "huggingface", "snapshots.org": ""})


def test_filter_score_fields_are_known_before_the_run():
    """The snapshot schema is declared up front, so the fields must be derivable."""
    fields = filter_score_fields(load_config(SMOKE))
    assert "dedup_max_sim" in fields
    assert "autorater_overall" in fields
    assert "autorater_spec_fidelity" in fields


def test_load_config_dict_clears_a_baked_in_run_id():
    base = load_config(SMOKE, {"run_id": "fixed"})
    arm = load_config_dict(base, {"generation.model": "m2"})
    assert arm["run_id"] is None
    assert arm["generation"]["model"] == "m2"


# --- sweep validation ---------------------------------------------------------


def test_sweep_loads(tmp_path):
    path = write(
        tmp_path,
        "s.yaml",
        {
            "base": SMOKE,
            "axis": "generation.model",
            "arms": [{"name": "a", "value": "m1"}, {"name": "b", "value": "m2"}],
        },
    )
    assert load_sweep(path)["axis"] == "generation.model"


def test_multi_axis_sweep_is_rejected(tmp_path):
    """Review-blocking rule: an unattributable arm difference is not a result."""
    path = write(
        tmp_path,
        "s.yaml",
        {
            "base": SMOKE,
            "axis": ["generation.model", "generation.temperature"],
            "arms": [{"name": "a", "value": "m1"}, {"name": "b", "value": "m2"}],
        },
    )
    with pytest.raises(ConfigError, match="Multi-axis"):
        load_sweep(path)


def test_arm_with_extra_keys_is_rejected(tmp_path):
    path = write(
        tmp_path,
        "s.yaml",
        {
            "base": SMOKE,
            "axis": "generation.model",
            "arms": [
                {"name": "a", "value": "m1", "generation.temperature": 0.2},
                {"name": "b", "value": "m2"},
            ],
        },
    )
    with pytest.raises(ConfigError, match="multi-axis"):
        load_sweep(path)


def test_base_overrides_may_not_touch_the_swept_axis(tmp_path):
    path = write(
        tmp_path,
        "s.yaml",
        {
            "base": SMOKE,
            "axis": "generation.model",
            "base_overrides": {"generation.model": "m0"},
            "arms": [{"name": "a", "value": "m1"}, {"name": "b", "value": "m2"}],
        },
    )
    with pytest.raises(ConfigError, match="swept axis"):
        load_sweep(path)


def test_sweep_needs_at_least_two_arms(tmp_path):
    path = write(
        tmp_path,
        "s.yaml",
        {"base": SMOKE, "axis": "generation.model", "arms": [{"name": "a", "value": "m1"}]},
    )
    with pytest.raises(ConfigError, match="at least two"):
        load_sweep(path)


def test_duplicate_arm_names_are_rejected(tmp_path):
    path = write(
        tmp_path,
        "s.yaml",
        {
            "base": SMOKE,
            "axis": "generation.model",
            "arms": [{"name": "a", "value": "m1"}, {"name": "a", "value": "m2"}],
        },
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        load_sweep(path)


def test_shipped_sweep_configs_are_valid():
    for name in ("generator_model.yaml", "revision_dose.yaml", "grouping_strategy.yaml"):
        sweep = load_sweep(name)
        assert len(sweep["arms"]) >= 2
