# ABOUTME: End-to-end tests: item construction, judge blinding, the rate views, and the 3 plots.
# ABOUTME: Several tests exist specifically to catch the failure modes that made v1 unusable.

from __future__ import annotations

import json

import pytest

from constieval import analysis, plots
from constieval.config import load_config
from constieval.core.cache import CacheConfig, CallCache
from constieval.core.llm import CachedLLM, EchoLLM
from constieval.core.store import ResultsStore, RunContext
from constieval.items.itemset import build_itemset, resolve_clause_set
from constieval.judges.base import JudgeConfig
from constieval.pipeline.generate import TargetConfig, generate
from constieval.pipeline.judging import judge_all
from constieval.pipeline.run import run_eval


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    """A tiny offline config writing into a temp directory."""
    root = tmp_path_factory.mktemp("constieval")
    return load_config(
        "smoke.yaml",
        {
            "output_dir": str(root / "out"),
            "cache.dir": str(root / "cache"),
            "itemset.dir": str(root / "itemsets"),
        },
    )


@pytest.fixture(scope="module")
def echo():
    """A cached echo client with caching off, so recorded calls are all real calls."""
    return CachedLLM(inner=EchoLLM(), cache=CallCache(CacheConfig(enabled=False)))


@pytest.fixture(scope="module")
def itemset(cfg, echo):
    """A built, unfrozen item set."""
    return build_itemset(cfg, resolve_clause_set(cfg), llm=echo)


class TestItemSet:
    def test_families_present(self, itemset):
        assert {i.family for i in itemset} == {"application", "retrieval", "fake_clause"}

    def test_item_ids_unique(self, itemset):
        ids = [i.item_id for i in itemset]
        assert len(ids) == len(set(ids))

    def test_build_is_deterministic(self, cfg, echo):
        a = build_itemset(cfg, resolve_clause_set(cfg), llm=echo)
        b = build_itemset(cfg, resolve_clause_set(cfg), llm=echo)
        assert a.itemset_id == b.itemset_id

    def test_retrieval_is_one_per_application_item(self, itemset):
        """knows and acts must share n AND scenarios, or the scatter compares two unrelated pools."""
        app = itemset.of_family("application")
        clean = [i for i in app if not i.is_derived]
        retrieval = itemset.of_family("retrieval")
        assert len(retrieval) == len(clean)
        sources = {i.meta["source_item_id"] for i in retrieval}
        assert sources == {i.item_id for i in clean}

    def test_pressure_covers_every_application_item(self, itemset):
        """v1 stressed a sample, so wrapper effects were confounded with item identity."""
        clean = [i for i in itemset.of_family("application") if not i.is_derived]
        stressed = [i for i in itemset.of_family("application") if i.is_derived]
        assert len(stressed) == len(clean)
        assert {i.parent_item_id for i in stressed} == {i.item_id for i in clean}

    def test_pressure_preserves_the_scenario(self, itemset):
        by_id = itemset.by_id()
        for child in (i for i in itemset if i.pressure):
            assert child.prompt == by_id[child.parent_item_id].prompt
            assert child.system

    def test_fake_clause_is_balanced(self, itemset):
        probes = itemset.of_family("fake_clause")
        real = [i for i in probes if i.meta["is_real"]]
        assert len(real) == len(probes) - len(real)


class TestJudging:
    @staticmethod
    def _recorder():
        """An echo client that records every prompt it is given."""
        seen: list[str] = []

        class Recorder(EchoLLM):
            def complete(self, model, messages, **params):
                seen.append("\n".join(m.get("content", "") for m in messages))
                return super().complete(model, messages, **params)

        return seen, CachedLLM(inner=Recorder(), cache=CallCache(CacheConfig(enabled=False)))

    def test_judge_is_blinded_to_recipe_and_model(self, cfg, itemset):
        """Asserted on the actual prompts: blinding is the kind of property that quietly lapses."""
        seen, llm = self._recorder()
        items = list(itemset)[:10]
        completions = generate(items, llm, TargetConfig.from_config(cfg), max_workers=2)
        seen.clear()
        judge_all(
            items, completions, resolve_clause_set(cfg), llm,
            RunContext(run_id="r", recipe="SECRET_RECIPE", model_id="SECRET_MODEL"),
            JudgeConfig.from_config(cfg), max_workers=2,
        )
        blob = "\n".join(seen)
        assert seen and "SECRET_RECIPE" not in blob and "SECRET_MODEL" not in blob

    def test_knows_judge_sees_every_clause(self, cfg, itemset):
        """The reliability fix: matching against the full list, not similarity to one clause."""
        seen, llm = self._recorder()
        clauses = resolve_clause_set(cfg)
        item = next(i for i in itemset if i.family == "retrieval")
        completions = generate([item], llm, TargetConfig.from_config(cfg), max_workers=1)
        seen.clear()
        judge_all([item], completions, clauses, llm, RunContext(run_id="r", recipe="x"),
                  JudgeConfig.from_config(cfg), max_workers=1)
        blob = "\n".join(seen)
        for clause in clauses:
            assert clause.title in blob, f"{clause.title} missing from the knows prompt"

    def test_failed_generation_is_errored_not_zero(self, cfg, itemset, echo):
        from constieval.core.types import Completion

        item = next(i for i in itemset if i.family == "application")
        broken = {item.item_id: Completion(item_id=item.item_id, text="", error="timeout")}
        rows = judge_all([item], broken, resolve_clause_set(cfg), echo,
                         RunContext(run_id="r", recipe="x"), JudgeConfig.from_config(cfg),
                         max_workers=1)
        assert rows and all(r.error and not r.passed for r in rows)

    def test_graded_rubric_is_refused(self, cfg):
        from constieval.control import loader
        from constieval.judges.base import RubricJudge

        clauses = resolve_clause_set(cfg)
        real = loader.rubric
        try:
            loader.rubric = lambda a: {**real(a), "scale_max": 3}
            with pytest.raises(ValueError, match="binary"):
                RubricJudge("acts", clauses)
        finally:
            loader.rubric = real


class TestEndToEnd:
    @staticmethod
    @pytest.fixture(scope="class")
    def frame(cfg):
        """Two arms on one item set."""
        a = run_eval(cfg, run_id="run_base")
        b = run_eval(
            {**cfg, "run": {**cfg["run"], "recipe": "finetuned", "checkpoint_step": 1},
             "target": {**cfg["target"], "model": "echo-ft"}},
            run_id="run_ft",
        )
        store = ResultsStore(a.store.rows)
        store.extend(b.store.rows)
        return store.to_frame()

    def test_no_errors_offline(self, frame):
        assert (frame["error"].fillna("") != "").sum() == 0

    def test_only_the_four_metrics_are_stored(self, frame):
        """v1 put a constant health check in the store; it was 36% of rows and mislabelled."""
        assert set(frame["axis"]) == {"acts", "notices", "knows", "discriminates"}

    def test_every_score_is_binary(self, frame):
        assert set(frame["score"].unique()) <= {0.0, 1.0}

    def test_both_arms_share_one_itemset(self, frame):
        assert len(analysis.check_comparable(frame)) == 1
        assert analysis.recipes(frame) == ["finetuned", "smoke"]

    def test_reasoning_retention_lives_in_the_manifest(self, cfg):
        result = run_eval(cfg, run_id="run_health")
        health = result.manifest["generation_health"]
        assert "reasoning_retained_rate" in health
        assert "reasoning_retained" not in set(result.store.to_frame()["axis"])


class TestAnalysis:
    @staticmethod
    @pytest.fixture(scope="class")
    def frame(cfg):
        a = run_eval(cfg, run_id="an_base")
        b = run_eval(
            {**cfg, "run": {**cfg["run"], "recipe": "finetuned"},
             "target": {**cfg["target"], "model": "echo-ft"}},
            run_id="an_ft",
        )
        store = ResultsStore(a.store.rows)
        store.extend(b.store.rows)
        return store.to_frame()

    def test_rates_are_bounded_and_carry_intervals(self, frame):
        table = analysis.rates(frame)
        assert not table.empty
        assert ((table["rate"] >= 0) & (table["rate"] <= 1)).all()
        assert (table["lo"] <= table["rate"]).all() and (table["rate"] <= table["hi"]).all()

    def test_scatter_pairs_gives_one_pooled_row_per_recipe(self, frame):
        pooled, per_clause = analysis.scatter_pairs(frame, "knows", "acts")
        assert len(pooled) == frame["recipe"].nunique()
        assert {"x_rate", "y_rate", "x_lo", "y_hi"} <= set(pooled.columns)
        assert not per_clause.empty

    def test_paired_pressure_is_paired_and_reports_mcnemar(self, frame):
        table = analysis.paired_pressure(frame, axis="acts")
        assert not table.empty
        for row in table.itertuples(index=False):
            assert row.n_pairs > 0
            assert 0.0 <= row.p <= 1.0
            assert row.delta == pytest.approx(row.pressure_rate - row.clean_rate)

    def test_health_warnings_catch_saturation(self, frame):
        """The check that would have caught v1's tension_recognition at 1.000 in both arms."""
        saturated = frame.copy()
        saturated.loc[saturated["axis"] == "acts", "passed"] = True
        warnings = analysis.health_warnings(saturated, min_n=1)
        assert any("SATURATED" in w and "acts" in w for w in warnings)

    def test_health_warnings_catch_thin_cells(self, frame):
        assert any("THIN CELL" in w for w in analysis.health_warnings(frame, min_n=10_000))

    def test_health_warnings_catch_mixed_itemsets(self, frame):
        mixed = frame.copy()
        mixed.loc[mixed.index[:5], "itemset_id"] = "is_other"
        assert any("NOT COMPARABLE" in w for w in analysis.health_warnings(mixed, min_n=1))


class TestPlots:
    @staticmethod
    @pytest.fixture(scope="class")
    def frame(cfg):
        a = run_eval(cfg, run_id="pl_base")
        b = run_eval(
            {**cfg, "run": {**cfg["run"], "recipe": "finetuned"},
             "target": {**cfg["target"], "model": "echo-ft"}},
            run_id="pl_ft",
        )
        store = ResultsStore(a.store.rows)
        store.extend(b.store.rows)
        return store.to_frame()

    def test_all_three_render(self, frame, tmp_path):
        written = plots.render_all(frame, tmp_path)
        assert set(written) == set(plots.PLOTS)
        for name, path in written.items():
            assert path and not path.startswith("ERROR:"), f"{name}: {path}"
            assert (tmp_path / f"{name}.png").stat().st_size > 10_000

    def test_report_writes_mirror_and_warnings(self, frame, tmp_path):
        from constieval.report import build_report

        written = build_report(frame, tmp_path)
        body = open(written["markdown"]).read()
        assert "Robustness" in body and "Rates" in body
        assert json.loads(open(written["summary"]).read())["recipes"]
