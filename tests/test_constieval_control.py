# ABOUTME: Tests for the control plane: the 8-clause set, the four binary rubrics, and config
# ABOUTME: validation. These encode the design decisions that fixed v1's unusable metrics.

from __future__ import annotations

import pytest

from constieval.config import ConfigError, load_config, validate
from constieval.control import loader


@pytest.fixture(scope="module")
def clauses():
    """The shipped clause set."""
    return loader.clause_set("principles_v2")


class TestClauseSet:
    def test_is_coarse_on_purpose(self, clauses):
        """Fine-grained clauses made `knows` unanswerable.

        v1 split these same principles into 21 overlapping claims and judges then disagreed 41% of
        the time about which one governed. Coarse clauses are the fix, not a shortcut.
        """
        assert 6 <= len(clauses) <= 10, f"expected ~8 distinct clauses, got {len(clauses)}"

    def test_every_clause_has_a_distinct_principle(self, clauses):
        principles = [c.principle for c in clauses]
        assert len(principles) == len(set(principles)), "two clauses share a principle - they overlap"

    def test_every_clause_has_distractors(self, clauses):
        for clause in clauses:
            fakes = clauses.fakes_for(clause.clause_id)
            assert len(fakes) >= 3, f"{clause.clause_id} has {len(fakes)} distractors, want >=3"
            for fake in fakes:
                assert fake.why_fake.strip(), f"{fake.fake_id} has no why_fake for the judge"

    def test_no_conflict_axis_is_claimed(self, clauses):
        # The source document states no ordering; claiming one would grade against an invented rule.
        assert clauses.priority_order == ""

    def test_find_tolerates_synthetic_ids(self, clauses):
        assert clauses.find("_global") is None


class TestRubrics:
    def test_exactly_the_four_plot_metrics(self):
        assert set(loader.declared_axes()) == {"acts", "notices", "knows", "discriminates"}

    def test_every_axis_is_binary(self):
        """Binary is the load-bearing decision.

        v1's 0-3 scales scored 2 only 9 times in 1324 rows while costing judge reliability
        (kappa 0.17 on the worst axis). The one binary rubric got perfect agreement.
        """
        for axis in loader.declared_axes():
            spec = loader.rubric(axis)
            assert spec["scale_max"] == 1, f"{axis} is not binary"
            assert spec["pass_at"] == 1
            assert spec["direction"] == "higher_better"

    def test_every_axis_is_bound_to_a_family(self):
        for axis in loader.declared_axes():
            assert loader.rubric(axis)["applies_to"], f"{axis} would never be scored"

    def test_families_map_to_the_right_metrics(self):
        assert set(loader.axes_for_family("application")) == {"acts", "notices"}
        assert loader.axes_for_family("retrieval") == ["knows"]
        assert loader.axes_for_family("fake_clause") == ["discriminates"]

    def test_knows_rubric_presents_the_whole_list(self):
        # The reliability fix: a matching task with one right answer, not a similarity judgement.
        template = loader.rubric("knows")["template"]
        assert "all_clauses" in template and "target_index" in template


class TestItemPrompts:
    def test_generator_forbids_narrating_the_problem(self):
        """Without this constraint `notices` sits at 100% and measures nothing.

        v1's scenarios said things like "we actually did double-bill", so noticing was free.
        """
        system = loader.pack("items")["system"]
        assert "NOT narrate the problem" in system
        assert "routine" in system.lower()

    def test_clear_difficulty_is_gone(self):
        # `clear` items drove `acts` to 86-90% at ceiling.
        assert set(loader.pack("items")["difficulty"]) == {"edge", "ambiguous"}

    def test_only_one_pressure_wrapper(self):
        # The other four moved neither model in v1 and cost a generation each.
        assert loader.declared_wrappers() == ["system_override"]
        assert loader.wrapper("system_override")["kind"] == "system"


class TestConfig:
    def test_shipped_configs_validate(self):
        for name in ("base.yaml", "smoke.yaml", "compare.yaml"):
            load_config(name)

    def test_smoke_is_fully_offline(self):
        cfg = load_config("smoke.yaml")
        assert cfg["target"]["provider"] == "echo"
        assert cfg["judge"]["provider"] == "echo"
        assert cfg["itemset"]["generator"]["provider"] == "echo"

    def test_base_and_compare_differ_only_in_the_target(self):
        """The comparison is only valid if everything except the weights is held fixed."""
        base, compare = load_config("base.yaml"), load_config("compare.yaml")
        assert base["itemset"]["families"] == compare["itemset"]["families"]
        assert base["itemset"]["transforms"] == compare["itemset"]["transforms"]
        assert base["judge"] == compare["judge"]
        assert base["target"]["temperature"] == compare["target"]["temperature"]
        assert base["target"]["max_tokens"] == compare["target"]["max_tokens"]

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ConfigError, match="not registered"):
            load_config("smoke.yaml", {"target.provider": "telepathy"})

    def test_undeclared_wrapper_is_rejected(self):
        with pytest.raises(ConfigError, match="undeclared wrappers"):
            load_config("smoke.yaml", {"itemset.transforms.pressure.wrappers": ["nope"]})

    def test_a_graded_rubric_would_be_rejected(self, monkeypatch):
        cfg = load_config("smoke.yaml")
        real = loader.rubric

        def graded(axis):
            spec = dict(real(axis))
            if axis == "acts":
                spec["scale_max"] = 3
            return spec

        monkeypatch.setattr(loader, "rubric", graded)
        with pytest.raises(ConfigError, match="must be\n?\\s*binary|binary"):
            validate(cfg)
