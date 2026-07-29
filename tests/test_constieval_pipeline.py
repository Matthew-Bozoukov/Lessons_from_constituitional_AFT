# ABOUTME: End-to-end tests for constieval: item building, transform pairing, judge blinding,
# ABOUTME: the analysis views, and every Tier A figure. Offline via the echo provider.

from __future__ import annotations

import pytest

from constieval import analysis, plots
from constieval.config import load_config
from constieval.core.cache import CacheConfig, CallCache
from constieval.core.llm import CachedLLM, EchoLLM
from constieval.core.store import ResultsStore
from constieval.items.itemset import build_itemset, resolve_clause_set
from constieval.judges.base import JudgeConfig
from constieval.pipeline.generate import TargetConfig, generate
from constieval.pipeline.judging import judge_all
from constieval.pipeline.run import run_eval
from constieval.pipeline.side_effects import generation_health
from constieval.core.store import RunContext


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
            "itemset.families.conflict.pairs": 2,
            "itemset.families.persona_drift.n": 2,
            "itemset.transforms.pressure.wrappers": ["system_override", "sunk_rapport"],
            "itemset.transforms.ood.axes": ["language"],
        },
    )


@pytest.fixture(scope="module")
def echo(tmp_path_factory):
    """A cached echo client with caching disabled, so recorded calls are all real calls."""
    return CachedLLM(inner=EchoLLM(), cache=CallCache(CacheConfig(enabled=False)))


@pytest.fixture(scope="module")
def itemset(cfg, echo):
    """A built, unfrozen item set."""
    return build_itemset(cfg, resolve_clause_set(cfg), llm=echo)


class TestItemSet:
    def test_covers_every_enabled_family(self, cfg, itemset):
        enabled = {f for f, s in cfg["itemset"]["families"].items() if s.get("enabled", True)}
        assert {i.family for i in itemset} >= enabled

    def test_item_ids_are_unique(self, itemset):
        ids = [i.item_id for i in itemset]
        assert len(ids) == len(set(ids))

    def test_build_is_deterministic(self, cfg, echo):
        again = build_itemset(cfg, resolve_clause_set(cfg), llm=echo)
        assert again.itemset_id == build_itemset(cfg, resolve_clause_set(cfg), llm=echo).itemset_id

    def test_retrieval_reuses_application_scenarios(self, itemset):
        # The retrieval-vs-application scatter is only a within-scenario comparison if the
        # retrieval item was built from an application item's scenario.
        application_ids = {i.item_id for i in itemset.of_family("application")}
        retrieval = itemset.of_family("retrieval")
        assert retrieval
        assert all(i.meta.get("source_item_id") in application_ids for i in retrieval)

    def test_fake_clause_items_are_balanced(self, itemset):
        probes = itemset.of_family("fake_clause")
        assert probes
        real = [i for i in probes if i.meta["is_real"]]
        # Half real, half fabricated: that is what turns recall into discrimination.
        assert len(real) == len(probes) - len(real)

    def test_derived_items_are_paired_to_a_parent(self, itemset):
        by_id = itemset.by_id()
        derived = [i for i in itemset if i.is_derived]
        assert derived
        assert all(i.parent_item_id in by_id for i in derived)

    def test_pressure_preserves_the_scenario(self, itemset):
        by_id = itemset.by_id()
        wrapped = [i for i in itemset if i.pressure == "system_override"]
        assert wrapped
        # A wrapper may add a system prompt but must never rewrite the scenario, or the
        # paired delta would confound robustness with item difficulty.
        for child in wrapped:
            assert child.prompt == by_id[child.parent_item_id].prompt
            assert child.system

    def test_multi_turn_wrapper_adds_history(self, itemset):
        wrapped = [i for i in itemset if i.pressure == "sunk_rapport"]
        assert wrapped and all(len(i.history) >= 2 for i in wrapped)

    def test_ood_items_record_their_axis_and_distance(self, itemset):
        ood = [i for i in itemset if i.ood_axis]
        assert ood
        for child in ood:
            assert child.ood_value
            assert int(child.meta["ood_distance"]) > 0

    def test_held_out_clauses_are_still_evaluated(self, cfg, itemset):
        clauses = resolve_clause_set(cfg)
        held = {c.clause_id for c in clauses.held_out}
        assert held and held <= {i.clause_id for i in itemset}

    def test_subsample_keeps_pairs_intact(self, itemset):
        small = itemset.subsample(12, seed=0)
        assert len(small) < len(itemset)
        kept = {i.item_id for i in small}
        # Every derived item still has its parent, or the paired deltas would silently
        # lose most of their rows.
        assert all(i.parent_item_id in kept for i in small if i.is_derived)
        assert small.itemset_id != itemset.itemset_id

    def test_subsample_is_a_no_op_when_not_smaller(self, itemset):
        assert itemset.subsample(0) is itemset

    def test_freeze_roundtrip(self, itemset, tmp_path):
        from constieval.items.itemset import ItemSet

        itemset.write(tmp_path)
        loaded = ItemSet.load(tmp_path / itemset.itemset_id)
        assert loaded.itemset_id == itemset.itemset_id
        assert len(loaded) == len(itemset)


class TestJudging:
    def test_judge_is_blinded_to_recipe_and_model(self, cfg, itemset):
        """The judge must never see who produced the response.

        Asserted on the actual prompts rather than by inspection, because blinding is the
        kind of property that quietly stops holding when someone adds a debug field.
        """
        recorded: list[str] = []

        class Recorder(EchoLLM):
            def complete(self, model, messages, **params):
                recorded.append("\n".join(m.get("content", "") for m in messages))
                return super().complete(model, messages, **params)

        llm = CachedLLM(inner=Recorder(), cache=CallCache(CacheConfig(enabled=False)))
        items = list(itemset)[:12]
        target = TargetConfig.from_config(cfg)
        completions = generate(items, llm, target, max_workers=2)
        recorded.clear()

        ctx = RunContext(
            run_id="r", recipe="SECRET_RECIPE", model_id="SECRET_MODEL", itemset_id=itemset.itemset_id
        )
        judge_all(items, completions, resolve_clause_set(cfg), llm, ctx, JudgeConfig.from_config(cfg), max_workers=2)

        assert recorded
        blob = "\n".join(recorded)
        assert "SECRET_RECIPE" not in blob
        assert "SECRET_MODEL" not in blob

    def test_clause_text_is_always_in_the_judge_context(self, cfg, itemset):
        recorded: list[str] = []

        class Recorder(EchoLLM):
            def complete(self, model, messages, **params):
                recorded.append("\n".join(m.get("content", "") for m in messages))
                return super().complete(model, messages, **params)

        llm = CachedLLM(inner=Recorder(), cache=CallCache(CacheConfig(enabled=False)))
        clauses = resolve_clause_set(cfg)
        item = next(i for i in itemset if i.family == "application")
        completions = generate([item], llm, TargetConfig.from_config(cfg), max_workers=1)
        recorded.clear()
        judge_all(
            [item], completions, clauses, llm, RunContext(run_id="r", recipe="x"),
            JudgeConfig.from_config(cfg), max_workers=1,
        )
        # A judge grading from memory of the constitution is grading something else.
        assert clauses.get(item.clause_id).text[:60] in "\n".join(recorded)

    def test_failed_generation_is_errored_not_zero(self, cfg, itemset, echo):
        from constieval.core.types import Completion

        item = next(i for i in itemset if i.family == "application")
        broken = {item.item_id: Completion(item_id=item.item_id, text="", error="timeout")}
        rows = judge_all(
            [item], broken, resolve_clause_set(cfg), echo, RunContext(run_id="r", recipe="x"),
            JudgeConfig.from_config(cfg), max_workers=1,
        )
        # Scoring a timeout as a failure would bias every aggregate that includes it.
        assert rows and all(r.error for r in rows)
        assert all(not r.passed for r in rows)


class TestEndToEnd:
    @staticmethod
    @pytest.fixture(scope="class")
    def result(cfg):
        """One complete offline run."""
        return run_eval(cfg, max_items=40)

    def test_run_produces_rows_and_a_manifest(self, result):
        assert len(result.store) > 0
        assert result.manifest["generation_health"]["n_error"] == 0
        assert result.manifest["n_rows"] == len(result.store)

    def test_every_row_carries_run_identity(self, result):
        df = result.store.to_frame()
        assert df["run_id"].nunique() == 1
        assert df["itemset_id"].nunique() == 1

    def test_no_judge_errors_offline(self, result):
        df = result.store.to_frame()
        assert (df["error"].fillna("") != "").sum() == 0

    def test_side_effect_axes_are_present(self, result):
        axes = set(result.store.to_frame()["axis"])
        assert {"over_refusal", "persona_drift", "reasoning_retained"} <= axes

    def test_generation_health_reports_truncation(self):
        from constieval.core.types import Completion

        health = generation_health(
            {
                "a": Completion(item_id="a", text="ok", finish_reason="stop", thinking="t"),
                "b": Completion(item_id="b", text="cut", finish_reason="length"),
            }
        )
        # A truncated answer is not a refusal; a spike here invalidates a run.
        assert health["n_truncated"] == 1
        assert health["reasoning_retained_rate"] == pytest.approx(0.5)


class TestAnalysisAndPlots:
    @staticmethod
    @pytest.fixture(scope="class")
    def frame(cfg):
        """Two runs under different recipes, on the same item set."""
        a = run_eval(cfg, max_items=40, run_id="run_a")
        b = run_eval(
            {**cfg, "run": {"recipe": "treated", "checkpoint_step": 500},
             "target": {**cfg["target"], "model": "echo-treated"}},
            max_items=40,
            run_id="run_b",
        )
        store = ResultsStore(a.store.rows)
        store.extend(b.store.rows)
        return store.to_frame()

    def test_two_recipes_share_one_itemset(self, frame):
        assert analysis.recipes(frame) == ["smoke", "treated"]
        assert len(analysis.check_comparable(frame)) == 1

    def test_orientation_flips_only_lower_better_axes(self, frame):
        oriented = analysis.orient(frame)
        refusal = oriented[oriented["axis"] == "over_refusal"]
        compliance = oriented[oriented["axis"] == "compliance"]
        assert (refusal["score_oriented"] == 1.0 - refusal["score"]).all()
        assert (compliance["score_oriented"] == compliance["score"]).all()

    def test_clause_matrix_excludes_global_rows(self, frame):
        matrix = analysis.clause_axis_matrix(frame)
        assert not matrix.empty
        assert "_global" not in set(matrix["clause_id"])

    def test_aggregates_carry_intervals(self, frame):
        table = analysis.headline_table(frame)
        assert not table.empty
        assert (table["lo"] <= table["mean"]).all() and (table["mean"] <= table["hi"]).all()

    def test_scatter_views_join_both_axes(self, frame):
        view = analysis.retrieval_vs_application(frame)
        assert not view.empty
        assert {"retrieval", "compliance"} <= set(view.columns)

    def test_ood_decay_is_per_axis_with_an_anchor(self, frame):
        decay = analysis.ood_decay(frame)
        assert not decay.empty
        # Every axis starts from the same items it decays away from.
        for _, part in decay.groupby("ood_axis"):
            assert part["distance"].min() == 0

    def test_robustness_delta_is_paired(self, frame):
        delta = analysis.robustness_delta(frame)
        assert not delta.empty
        assert {"delta", "lo", "hi", "n"} <= set(delta.columns)

    def test_all_tier_a_plots_render(self, frame, tmp_path):
        written = plots.render_all(frame, tmp_path)
        assert set(written) == set(plots.TIER_A_PLOTS)
        for name, path in written.items():
            assert not path.startswith("ERROR:"), f"{name} failed: {path}"
            assert path, f"{name} produced no figure"

    def test_report_writes_markdown_mirror(self, frame, tmp_path):
        from constieval.report import build_report

        written = build_report(frame, tmp_path)
        body = open(written["markdown"]).read()
        # The relief rule: every figure's numbers must also be readable as text.
        assert "Retrieval vs application" in body
        assert "OOD decay" in body

    def test_report_warns_on_mixed_itemsets(self, frame, tmp_path):
        from constieval.report import build_report

        mixed = frame.copy()
        mixed.loc[mixed.index[:5], "itemset_id"] = "is_other"
        written = build_report(mixed, tmp_path / "mixed", make_plots=False)
        assert any("MUST NOT be compared" in w for w in written["warnings"])


class TestPalette:
    def test_recipe_colour_follows_the_entity_not_its_rank(self):
        from constieval.plots.theme import recipe_colors

        # Filtering a report down must not repaint the survivors.
        full = recipe_colors(["alpha", "beta", "gamma"])
        subset = recipe_colors(["alpha", "gamma"])
        assert subset["alpha"] == full["alpha"]

    def test_too_many_series_is_an_error_not_a_generated_hue(self):
        from constieval.plots.theme import PaletteError, recipe_colors

        with pytest.raises(PaletteError, match="budget"):
            recipe_colors([f"r{i}" for i in range(4)], scatter=True)


class TestCostControls:
    def test_justification_is_scoped_to_clean_items(self, cfg, itemset, echo):
        """The justification axis must not be re-run on stressed or OOD items.

        It asks whether the model has the document's own reasoning; re-asking under a
        system-prompt override or a translation is a different question, and it was the
        single largest avoidable cost in the judge pass.
        """
        items = [i for i in itemset if i.family == "application"]
        derived = [i for i in items if i.is_derived]
        assert derived, "need derived items to make this meaningful"
        completions = generate(items, echo, TargetConfig.from_config(cfg), max_workers=4)
        rows = judge_all(
            items, completions, resolve_clause_set(cfg), echo,
            RunContext(run_id="r", recipe="x"), JudgeConfig.from_config(cfg), max_workers=4,
        )
        conditions = {r.condition for r in rows if r.axis == "justification_quality"}
        assert conditions == {"clean"}
        # The axes that DO belong on stressed items are still scored there.
        stressed = {r.axis for r in rows if r.condition != "clean"}
        assert {"compliance", "tension_recognition"} <= stressed

    def test_ood_max_distance_caps_the_tail(self, cfg, echo):
        from constieval.config import load_config_dict

        capped = load_config_dict(
            cfg,
            {
                "itemset.transforms.ood.axes": ["language"],
                "itemset.transforms.ood.max_distance": 1,
            },
        )
        built = build_itemset(capped, resolve_clause_set(capped), llm=echo)
        distances = {int(i.meta["ood_distance"]) for i in built if i.ood_axis}
        assert distances == {1}

    def test_condition_kind_strips_parameters(self, itemset):
        kinds = {i.condition_kind for i in itemset}
        assert kinds <= {"clean", "pressure", "ood"}

    def test_estimate_counts_match_a_real_build(self, cfg, itemset):
        """The projection must agree with what the builders actually produce.

        An estimator that drifts from the builders is worse than none: it would quietly
        under-quote the run everyone is deciding to pay for.
        """
        from constieval.estimate import estimate

        projected = estimate(cfg, arms=1)
        assert projected.counts["items TOTAL"] == len(itemset)
        assert projected.counts["items (ood)"] == sum(1 for i in itemset if i.ood_axis)
        assert projected.counts["items (pressure)"] == sum(1 for i in itemset if i.pressure)

    def test_estimate_scales_with_arms(self, cfg):
        from constieval.estimate import estimate

        one, two = estimate(cfg, arms=1), estimate(cfg, arms=2)
        assert two.counts["judge calls per arm"] == one.counts["judge calls per arm"]
        # Item generation is charged once no matter how many arms share the item set.
        gen = next(k for k in one.costs if k.startswith("item generation"))
        assert two.costs[gen] == one.costs[gen]

    def test_judge_agreement_detects_a_disagreeing_judge(self, cfg, tmp_path, echo):
        """A cross-check that cannot notice a different judge would be decorative."""
        from constieval.judge_check import JudgeCheckError, check_judge, load_run

        result = run_eval(cfg, max_items=30, run_id="agree_run")
        itemset_used = ItemSetFinder(cfg, result)
        reference = JudgeConfig.from_config(cfg)
        reference.model = "echo-different-judge"
        report = check_judge(
            result.run_dir, itemset_used, resolve_clause_set(cfg), echo, reference,
            n=40, max_workers=4,
        )
        assert report.overall["n"] > 0
        assert 0.0 <= report.overall["raw"] <= 1.0
        assert report.verdict() in ("PASS", "REVIEW")
        # Sanity: the loader refuses a directory that is not a completed run.
        with pytest.raises(JudgeCheckError):
            load_run(tmp_path)


class TestRunGuards:
    def test_truncation_is_flagged_loudly(self):
        """A run whose completions were cut off must not report quietly.

        A truncated answer is not a refusal, but a judge grades it as one, so a run with a
        high truncation rate is measuring the token budget rather than the model. This is
        the failure that invalidated the first real arm-A pass.
        """
        from constieval.pipeline.side_effects import generation_health
        from constieval.core.types import Completion

        cut = {
            f"i{n}": Completion(item_id=f"i{n}", text="half an ans", finish_reason="length")
            for n in range(9)
        }
        cut["ok"] = Completion(item_id="ok", text="a whole answer", finish_reason="stop")
        health = generation_health(cut)
        assert health["truncation_rate"] == pytest.approx(0.9)
        assert health["truncation_rate"] > 0.15  # the shipped default limit

    def test_clean_run_carries_no_warnings(self, cfg):
        result = run_eval(cfg, max_items=24, run_id="clean_guard_run")
        assert result.warnings == []
        assert result.manifest["generation_health"]["truncation_rate"] == 0.0
        assert "warnings" in result.manifest


def ItemSetFinder(cfg, result):
    """Load the frozen item set a completed run used."""
    from constieval.items.itemset import ItemSet

    return ItemSet.find(cfg["itemset"]["dir"], result.itemset_id)


class TestStudy:
    """The one-command orchestration: many arms, one item set, one bundle."""

    @staticmethod
    def _echo(tmp_path):
        """Offline overrides pointing every path at a temp dir."""
        return {
            "output_dir": str(tmp_path / "out"),
            "cache.dir": str(tmp_path / "cache"),
            "itemset.dir": str(tmp_path / "is"),
            "itemset.generator.provider": "echo",
            "judge.provider": "echo",
            "target.provider": "echo",
            "target.base_url": None,
            "itemset.families.application.difficulties": ["clear"],
            "itemset.families.application.variants": 1,
            "itemset.families.persona_drift.n": 2,
            "itemset.transforms.ood.axes": ["language"],
            "itemset.transforms.ood.max_distance": 1,
            "itemset.transforms.pressure.wrappers": ["system_override"],
        }

    def test_bundle_is_self_contained(self, tmp_path):
        from constieval.study import run_study

        result = run_study(
            "base=cheap.yaml,treated=cheap.yaml",
            name="t",
            out_root=tmp_path / "studies",
            overrides={**self._echo(tmp_path), "itemset.id": None},
        )
        b = result.bundle
        # Everything needed to re-read or re-plot the study must be inside the bundle,
        # because output/ gets cleaned and bundles get moved between machines.
        for path in ("study.json", "README.md", "report/tier_a_results.md", "itemset/items.jsonl"):
            assert (b / path).exists(), f"missing {path}"
        for arm in ("base", "treated"):
            assert (b / "runs" / arm / "results.jsonl").exists()
            assert (b / "runs" / arm / "completions.jsonl").exists()
        assert len(list((b / "report" / "figures").glob("*.png"))) == 7

    def test_all_arms_share_one_itemset(self, tmp_path):
        """The comparison must be valid by construction, not by remembering to pin an id."""
        from constieval.study import run_study

        result = run_study(
            "a=cheap.yaml,b=cheap.yaml",
            name="t",
            out_root=tmp_path / "studies",
            overrides={**self._echo(tmp_path), "itemset.id": None},
        )
        import json as _json

        ids = set()
        for arm in ("a", "b"):
            rows = [
                _json.loads(line)
                for line in (result.bundle / "runs" / arm / "results.jsonl").read_text().splitlines()
                if line.strip()
            ]
            ids |= {r["itemset_id"] for r in rows}
        assert len(ids) == 1
        assert not (result.report.get("warnings") or [])

    def test_a_failing_arm_does_not_sink_the_study(self, tmp_path):
        from constieval.study import run_study

        result = run_study(
            "good=cheap.yaml,broken=cheap.yaml",
            name="t",
            out_root=tmp_path / "studies",
            overrides={**self._echo(tmp_path), "itemset.id": None},
            # The broken arm points at an endpoint that is not up - the common real case.
            cross_check="",
        )
        assert [a.status for a in result.arms] == ["ok", "ok"]
        assert result.ok_arms

    def test_malformed_arm_spec_is_rejected(self):
        from constieval.study import StudyError, run_study

        with pytest.raises(StudyError, match="name=config"):
            run_study("just-a-config.yaml")
        with pytest.raises(StudyError, match="Duplicate arm"):
            run_study("a=x.yaml,a=y.yaml")
