# ABOUTME: Offline tests for the config-driven engine: caching, checkpoints, ablation,
# ABOUTME: skip semantics, stage numbering and estimates. Fake operator kind, no network.

from __future__ import annotations

import json

import pytest
import yaml

from src.data.synth.stage_runtime import Stage
from src.data.synth.stage_operators import OPERATORS
from src.data.synth.pipeline import build_stages, estimate, run, snapshot_positions

CONSTITUTION = "constitutions/archive/claude_distilled_8_principles_v1/constitution.md"

CALLS: list[str] = []  # which stage fns actually ran, for cache assertions


def _fake_operator(sc, cfg):
    def fn(ctx, records, ckpt):
        CALLS.append(sc["name"])
        if sc.get("use_checkpoint"):
            assert ckpt is not None and ckpt.key == sc["checkpoint"]
            for i in range(2):
                ckpt.record({sc["checkpoint"]: f"s{i}", "v": i})
            return list(ckpt.done.values())
        return [{"id": f"s{i}", sc["name"]: True,
                 **(records[i] if i < len(records) else {})} for i in range(3)]

    return Stage(sc["name"], fn, checkpoint_key=sc.get("checkpoint"),
                 skip=(lambda ctx, rs: True) if sc.get("always_skip") else None)


@pytest.fixture(autouse=True)
def _register_fake():
    OPERATORS["fake"] = _fake_operator
    CALLS.clear()
    yield
    OPERATORS.pop("fake", None)


def _cfg(tmp_path, stages, **extra):
    return {"pipeline": "fake_type", "constitution": CONSTITUTION,
            "output_dir": str(tmp_path), "workers": 2,
            "stages": stages, **extra}


def _st(name, **kw):
    return {"name": name, "kind": "fake", **kw}


def test_runner_snapshots_every_stage_and_reuses_them_on_resume(tmp_path):
    cfg = _cfg(tmp_path, [_st("alpha"), _st("beta")])
    m = run(cfg)
    assert (tmp_path / m["run_id"] / "stage_1_alpha.jsonl").exists()
    assert (tmp_path / m["run_id"] / "stage_2_beta.jsonl").exists()
    assert m["counts"] == {"alpha": 3, "beta": 3}
    assert CALLS == ["alpha", "beta"]

    CALLS.clear()
    m2 = run(cfg, resume=m["run_dir"])
    assert CALLS == [], "resume must reuse every snapshot"
    assert m2["counts"] == m["counts"]

    CALLS.clear()
    (tmp_path / m["run_id"] / "stage_2_beta.jsonl").unlink()
    run(cfg, resume=m["run_dir"])
    assert CALLS == ["beta"], "only the deleted stage re-runs"


def test_ablation_applies_the_field_copy_null_op_and_lands_in_the_manifest(tmp_path):
    cfg = _cfg(tmp_path,
               [_st("draft"),
                _st("revise", ablate_with={"final": "draft"}),
                _st("export")],
               ablate=["revise"])
    m = run(cfg)
    assert "revise" not in CALLS, "ablated stage must not run its fn"
    assert m["ablated"] == ["revise"]
    # The null-op output is still snapshotted in the stage's slot, so arms diff cleanly.
    rows = [json.loads(line) for line in
            (tmp_path / m["run_id"] / "stage_2_revise.jsonl").open()]
    assert all(r["final"] == r["draft"] for r in rows)


def test_ablation_fails_fast_on_typos_and_load_bearing_stages(tmp_path):
    with pytest.raises(ValueError, match="not in this pipeline's stages"):
        run(_cfg(tmp_path, [_st("draft")], ablate=["revsie"]))
    with pytest.raises(ValueError, match="cannot be ablated"):
        run(_cfg(tmp_path, [_st("draft")], ablate=["draft"]))


def test_unknown_stage_kind_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        build_stages(_cfg(tmp_path, [{"name": "x", "kind": "nope"}]))


def test_skipped_stage_keeps_its_slot(tmp_path):
    cfg = _cfg(tmp_path, [_st("plan"), _st("perturb", always_skip=True),
                          _st("generate")])
    m = run(cfg)
    assert CALLS == ["plan", "generate"]
    assert "perturb" not in m["counts"]
    assert not (tmp_path / m["run_id"] / "stage_2_perturb.jsonl").exists()
    assert (tmp_path / m["run_id"] / "stage_3_generate.jsonl").exists(), \
        "later stages keep their positional slot when one is skipped"


def test_checkpointed_stage_gets_a_partial_file_at_its_slot(tmp_path):
    cfg = _cfg(tmp_path, [_st("work", checkpoint="id", use_checkpoint=True)])
    m = run(cfg)
    assert (tmp_path / m["run_id"] / "stage_1_work.partial.jsonl").exists()
    assert m["counts"]["work"] == 2


def test_smoke_merges_overrides_and_never_mutates_the_input(tmp_path):
    seen = []
    OPERATORS["probe"] = lambda sc, cfg: Stage(
        sc["name"], lambda ctx, rs, ck: seen.append(ctx.cfg["knob"]) or [{"id": "s0"}])
    try:
        cfg = _cfg(tmp_path, [{"name": "only", "kind": "probe"}],
                   knob=100, smoke={"knob": 1})
        m = run(cfg, smoke=True)
        assert seen == [1], "stages must see the smoke-merged config"
        assert cfg["knob"] == 100, "the caller's config must never be mutated"
        assert "smoke_" in m["run_dir"] and m["smoke"] is True
        assert m["effective"] == {"knob": 1}
    finally:
        OPERATORS.pop("probe", None)


# --- the real configs against the engine --------------------------------------------


def _real(name):
    return yaml.safe_load(open(f"configs/data/synth/{name}.yaml"))


def _archived(name):
    """A frozen config from `configs/data/synth/archive/`.

    Archived configs are the exact record of a published corpus, so they never change --
    which makes them the stable fixtures for the engine contracts below (snapshot names,
    priced totals) that a live config would keep invalidating.
    """
    return yaml.safe_load(open(f"configs/data/synth/archive/{name}.yaml"))


def test_real_configs_keep_historical_snapshot_names():
    # Existing run dirs and HF mirrors must stay resumable: positions and names of
    # every snapshot are part of the on-disk contract.
    # Renamed 2026-08-13 so each stage states its action (traits -> chunk_constitution,
    # final -> revise_responses, ...). A stage name IS its snapshot filename, so run dirs
    # from before that date no longer cache-hit; the published HF mirror keeps the old
    # names, and consumers pin them via `source.snapshot`. The corpus checks are OBSERVERS:
    # they take no position, so they sit anywhere in the list without moving a snapshot.
    #
    # difficult_advice BROKE the observer contract on 2026-08-13, deliberately.
    # `dedupe_scenarios` is a corpus_filter: it removes records, so it cannot be an
    # observer, and it takes position 3 -- shifting every snapshot after it by one. The
    # break is acceptable only because the same commit rewrote the scenario, response and
    # rewrite prompts: a run dir from before it cannot be resumed into this config at ANY
    # numbering without mixing two recipes into one corpus. Its output belongs in a new
    # dated HF repo, not appended to the v1 mirror. Do not renumber a config to dodge a
    # test; if the recipe has not changed, put the new stage last, where nothing follows
    # it to move.
    assert [s.name for s in build_stages(_real("2026-08-01_difficult_advice"))] == \
        ["chunk_constitution", "write_scenarios", "corpus_scenarios", "dedupe_scenarios",
         "draft_prompts", "revise_prompts", "draft_responses", "revise_responses",
         "export_sft", "corpus"]
    assert snapshot_positions(build_stages(_real("2026-08-01_difficult_advice"))) == \
        {"chunk_constitution": 1, "write_scenarios": 2, "corpus_scenarios": 2,
         "dedupe_scenarios": 3, "draft_prompts": 4, "revise_prompts": 5,
         "draft_responses": 6, "revise_responses": 7, "export_sft": 8, "corpus": 8}
    # The archived configs are frozen, which makes them the stable fixture for this.
    assert [s.name for s in build_stages(_archived("model_eval_model"))] == \
        ["source", "plan", "perturbed", "generated", "final", "sft", "corpus"]


def test_estimate_prices_real_configs_and_ablation_out():
    da = _real("2026-08-01_difficult_advice")
    full = estimate(da)
    # $47.68 pre-refactor at n_traits=12; re-cut to ten units on 2026-08-04 ($39.73 at
    # 770 records), then on 2026-08-05 matched byte-exact to the nine-principle document
    # the 2026-08-04 source corpus was generated against, so the same per-call priors
    # now price 693 records. $35.76 until 2026-08-13, when `pattern_scan` was enabled:
    # +$19.71 for the judged tier (31 long-context scan calls on Sonnet 5 plus the
    # classifier sweep on Haiku), which is the cost this config now pays on every run to
    # get an open-ended read on its own recurring tics.
    #
    # NOTE this whole number is priced from `assumed_tokens`, and those priors are ~3x
    # low: the 2026-08-04 run of this config actually cost $166.72, against a rewrite
    # stage assumed at 3,200 in / 1,800 out per call that really ran 10,370 / 2,869.
    # The assertion pins the estimator's arithmetic, NOT the real cost of a run.
    # $56.18 at 693 records; the config moved to `total_scenarios: 2000` on 2026-08-13 to
    # size the v2 corpus against v1's 2,203.
    assert full["total_usd"] == 131.16
    ablated = estimate({**da, "ablate": ["revise_responses"]})
    calls = {r["stage"]: r["calls"] for r in full["per_stage"]}
    assert calls["rewrite"] == 2000
    assert "rewrite" not in {r["stage"] for r in ablated["per_stage"]}
    assert ablated["total_usd"] < full["total_usd"]

    mem = _archived("model_eval_model")
    est = estimate(mem)
    # $100.32 at 300/cell; cells raised to 420 on 2026-08-05 for the 10k-example
    # 20/80 SFT run (docs/plan_full_sft_20_80.md); $140.45 before the `final` rewrite
    # pass (added 2026-08-07: one Sonnet 5 call per verdict-carrying document).
    assert est["total_usd"] == 269.81
    calls = {r["stage"]: r["calls"] for r in est["per_stage"]}
    assert calls["rewrite"] == 4 * 420, "control documents are not rewritten"
    ablated_mem = estimate({**mem, "ablate": ["final"]})
    assert "rewrite" not in {r["stage"] for r in ablated_mem["per_stage"]}
    assert ablated_mem["total_usd"] == 140.45, \
        "--ablate final prices back to the pre-rewrite pipeline"


def test_provider_routing_is_not_a_synth_config_concern():
    # One provider per model, globally, in configs/endpoints/providers.yaml (applied
    # inside OpenRouterClient). A synth config carrying `provider:` anywhere — stage
    # block or defaults — fails loudly instead of being silently ignored.
    from src.data.synth.stage_runtime import model_cfg

    pin = {"order": ["anthropic"], "allow_fallbacks": False}
    base = {"defaults": {}, "models": {"draft": {"model": "anthropic/claude-haiku-4.5"}}}
    assert "extra_body" not in model_cfg(base, "draft")

    stage = {"defaults": {}, "models": {"draft": {"model": "x", "provider": pin}}}
    with pytest.raises(ValueError, match="providers.yaml"):
        model_cfg(stage, "draft")

    run = {"defaults": {"provider": pin}, "models": {"draft": {"model": "x"}}}
    with pytest.raises(ValueError, match="providers.yaml"):
        model_cfg(run, "draft")
