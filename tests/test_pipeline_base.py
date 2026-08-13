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


def test_real_configs_keep_historical_snapshot_names():
    # Existing run dirs and HF mirrors must stay resumable: positions and names of
    # every snapshot are part of the on-disk contract.
    # The corpus checks (2026-08-12) are OBSERVERS: they take no position, so they can
    # sit anywhere in the list -- including mid-pipeline -- without moving a snapshot.
    assert [s.name for s in build_stages(_real("difficult_advice"))] == \
        ["traits", "scenarios", "corpus_scenarios", "draft_prompts", "refined_prompts",
         "draft_responses", "final", "export", "corpus"]
    assert snapshot_positions(build_stages(_real("difficult_advice"))) == \
        {"traits": 1, "scenarios": 2, "corpus_scenarios": 2, "draft_prompts": 3,
         "refined_prompts": 4, "draft_responses": 5, "final": 6, "export": 7, "corpus": 7}
    # `final` (the rewrite pass) added 2026-08-07: stages 1-4 keep their positions, so
    # completed run dirs still cache-hit everything already paid for; only the free sft
    # assembly moves (5 -> 6) and re-runs.
    assert [s.name for s in build_stages(_real("model_eval_model"))] == \
        ["source", "plan", "perturbed", "generated", "final", "export", "corpus"]
    assert snapshot_positions(build_stages(_real("model_eval_model"))) == \
        {"source": 1, "plan": 2, "perturbed": 3, "generated": 4, "final": 5, "export": 6,
         "corpus": 6}


def test_estimate_prices_real_configs_and_ablation_out():
    da = _real("difficult_advice")
    full = estimate(da)
    # $47.68 pre-refactor at n_traits=12; re-cut to ten units on 2026-08-04 ($39.73 at
    # 770 records), then on 2026-08-05 matched byte-exact to the nine-principle document
    # the 2026-08-04 source corpus was generated against, so the same per-call priors
    # now price 693 records.
    assert full["total_usd"] == 35.76
    ablated = estimate({**da, "ablate": ["final"]})
    calls = {r["stage"]: r["calls"] for r in full["per_stage"]}
    assert calls["rewrite"] == 693
    assert "rewrite" not in {r["stage"] for r in ablated["per_stage"]}
    assert ablated["total_usd"] < full["total_usd"]

    mem = _real("model_eval_model")
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
