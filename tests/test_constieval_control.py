# ABOUTME: Tests for the control plane: the frozen clause set, rubric declarations, pressure
# ABOUTME: wrappers, OOD distance ordering, and config validation. No network.

from __future__ import annotations

import pytest

from constieval.config import ConfigError, load_config, validate
from constieval.control import loader
from constieval.core import registry


@pytest.fixture(scope="module")
def clauses():
    """The shipped clause set."""
    return loader.clause_set("approved_constitution_v1")


class TestClauseSet:
    def test_loads_and_is_non_trivial(self, clauses):
        assert len(clauses) >= 15
        assert clauses.spec_id == "approved_constitution_v1"

    def test_every_clause_states_a_rationale(self, clauses):
        # The justification axis grades against the stated rationale; a clause without one
        # would silently make that axis ungradeable.
        missing = [c.clause_id for c in clauses if not c.rationale.strip()]
        assert missing == []

    def test_held_out_subset_exists_and_is_a_minority(self, clauses):
        assert 0 < len(clauses.held_out) < len(clauses)

    def test_priority_order_and_note_are_present(self, clauses):
        assert clauses.priority_order.strip()
        # The ordering is holistic, not lexical; a judge told otherwise grades the wrong thing.
        assert "HOLISTIC" in clauses.priority_note.upper()

    def test_priority_order_keeps_its_line_structure(self, clauses):
        assert clauses.priority_order.count("\n") >= 3

    def test_distractors_target_real_clauses(self, clauses):
        assert clauses.fakes
        for fake in clauses.fakes:
            assert clauses.find(fake.near_clause_id) is not None
            assert fake.why_fake.strip()

    def test_find_returns_none_for_synthetic_ids(self, clauses):
        assert clauses.find("_global") is None
        with pytest.raises(KeyError):
            clauses.get("_global")

    def test_fingerprint_tracks_content(self, clauses):
        assert clauses.fingerprint == loader.clause_set("approved_constitution_v1").fingerprint

    def test_conflict_items_are_constructible(self, clauses):
        # Needs at least one pair with differing tiers AND differing principles.
        pairs = [
            (a, b)
            for i, a in enumerate(clauses.clauses)
            for b in clauses.clauses[i + 1 :]
            if a.priority_tier != b.priority_tier and a.principle != b.principle
        ]
        assert len(pairs) >= 12


class TestRubrics:
    def test_every_axis_has_a_registered_judge(self):
        import constieval.judges  # noqa: F401

        for axis in loader.declared_axes():
            assert registry.has("judge", axis), f"no judge registered for axis {axis}"

    def test_every_axis_declares_families_and_a_direction(self):
        for axis in loader.declared_axes():
            spec = loader.rubric(axis)
            assert spec["applies_to"], f"{axis} is bound to no family"
            assert spec["direction"] in ("higher_better", "lower_better")
            assert 0 <= spec["pass_at"] <= spec["scale_max"]

    def test_compliance_and_tension_are_separate_axes(self):
        # Scored by separate calls on purpose: one rubric covering both would let a strong
        # compliance score hide a model that never noticed anything was at stake.
        assert "compliance" in loader.declared_axes()
        assert "tension_recognition" in loader.declared_axes()
        for axis in ("compliance", "tension_recognition"):
            assert "application" in loader.rubric(axis)["applies_to"]

    def test_over_refusal_is_lower_better(self):
        assert loader.rubric("over_refusal")["direction"] == "lower_better"

    def test_axes_for_family_is_driven_by_the_rubrics(self):
        assert set(loader.axes_for_family("application")) >= {
            "compliance",
            "tension_recognition",
            "justification_quality",
        }
        assert loader.axes_for_family("nonexistent_family") == []

    def test_unknown_axis_raises(self):
        with pytest.raises(loader.PromptError):
            loader.rubric("no_such_axis")


class TestPressureAndOOD:
    def test_wrappers_declare_a_known_kind(self):
        assert loader.declared_wrappers()
        for name in loader.declared_wrappers():
            spec = loader.wrapper(name)
            assert spec["kind"] in ("system", "prefix", "history")
            assert spec[spec["kind"]]

    def test_ood_axes_start_at_distance_zero_and_increase(self):
        assert loader.declared_ood_axes()
        for name in loader.declared_ood_axes():
            distances = [int(v["distance"]) for v in loader.ood_axis(name)["values"]]
            assert distances[0] == 0, f"{name} has no anchor for its decay curve"
            assert distances == sorted(distances)

    def test_unknown_wrapper_and_axis_raise(self):
        with pytest.raises(loader.PromptError):
            loader.wrapper("no_such_wrapper")
        with pytest.raises(loader.PromptError):
            loader.ood_axis("no_such_axis")


class TestRender:
    def test_strict_undefined_fails_loudly(self):
        # A template referencing a field the caller forgot must fail rather than silently
        # render a prompt with a hole in it.
        with pytest.raises(loader.PromptError):
            loader.render("{{ missing_variable }}")

    def test_renders_with_context(self):
        assert loader.render("hello {{ who }}", who="world") == "hello world"


class TestConfig:
    def test_base_and_smoke_configs_validate(self):
        for name in ("base.yaml", "smoke.yaml", "checkpoint.yaml", "hf_local.yaml"):
            load_config(name)

    def test_smoke_extends_base(self):
        cfg = load_config("smoke.yaml")
        assert cfg["target"]["provider"] == "echo"
        assert cfg["clause_set"] == "approved_constitution_v1"  # inherited from base

    def test_overrides_replace_rather_than_merge(self):
        cfg = load_config("smoke.yaml", {"itemset.transforms.ood.axes": ["language"]})
        assert cfg["itemset"]["transforms"]["ood"]["axes"] == ["language"]

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ConfigError, match="not registered"):
            load_config("smoke.yaml", {"target.provider": "telepathy"})

    def test_unknown_family_is_rejected(self):
        cfg = load_config("smoke.yaml")
        cfg["itemset"]["families"]["not_a_family"] = {"enabled": True}
        with pytest.raises(ConfigError, match="unknown families"):
            validate(cfg)

    def test_undeclared_pressure_wrapper_is_rejected(self):
        with pytest.raises(ConfigError, match="undeclared wrappers"):
            load_config("smoke.yaml", {"itemset.transforms.pressure.wrappers": ["nope"]})

    def test_retrieval_without_application_is_rejected(self):
        # Retrieval items are derived from application scenarios; enabling one without the
        # other would quietly produce an empty family.
        with pytest.raises(ConfigError, match="derived from application"):
            load_config("smoke.yaml", {"itemset.families.application.enabled": False})

    def test_missing_clause_set_is_rejected(self):
        with pytest.raises(ConfigError, match="Available clause sets"):
            load_config("smoke.yaml", {"clause_set": "no_such_spec"})
