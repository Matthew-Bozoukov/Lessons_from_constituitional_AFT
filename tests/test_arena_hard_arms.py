# ABOUTME: Offline tests for Arena-Hard's two arm forms — a model that generates and a
# ABOUTME: prior run whose answers are reused — and for the pooled `vs-<baseline>` artifact.

import json
import pathlib

import pytest
from omegaconf import OmegaConf

from src.eval import EVALS
from src.eval.capabilities.arena_hard.runner import register
from src.eval.capabilities.arena_hard.arena_hard_judge import _arm
from src.naming import eval_name


def _arm_run(tmp_path, key, target, *, reference=False):
    """A published arm as run_eval hands it to pool(): answers + its own metadata."""
    d = tmp_path / key
    (d / "rollouts").mkdir(parents=True)
    (d / "metadata").mkdir(parents=True)
    (d / "rollouts" / "answers.jsonl").write_text("{}\n")
    (d / "metadata" / "sources.json").write_text(json.dumps(
        {"arm": key, "answers": {"generated": target}, "reference_arm": reference}))
    return {"target": target, "model_key": key, "mode": "think",
            "out_dir": str(d), "repo": f"LASR-Callum/2026-09-05-ah-{key}"}


def test_a_dynamic_cli_arm_gets_the_prompt_counts_the_judge_reads():
    # Without arm_defaults an appended --target dies in the judge with
    # `Missing key n_hard_prompt`, because only the static ladder spells them out.
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.arms = register(OmegaConf.to_container(cfg.arms, resolve=True),
                         "qwen36_difficult_advice_0", "org/model", "target", cfg)
    assert int(_arm(cfg, "qwen36_difficult_advice_0").n_hard_prompt) > 0


def test_registering_an_arm_twice_does_not_duplicate_it():
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    arms = OmegaConf.to_container(cfg.arms, resolve=True)
    once = register(arms, "a", "org/a", "target", cfg)
    assert register(once, "a", "org/a", "baseline", cfg) == once


def test_arena_hard_declares_the_arm_forms_it_supports():
    spec = EVALS["arena_hard"]
    # A prior run is a valid target; the baseline is an ordinary arm that runs first.
    assert spec.reads_answers and spec.arm_kwargs == ("reference",) and spec.pools
    # A behaviour eval must always generate — reusing responses reuses the experiment.
    assert not EVALS["psychosis"].reads_answers
    assert not EVALS["agentic_misalignment"].reads_answers
    assert not EVALS["odcv"].reads_answers


def test_an_arm_publishes_no_results_because_a_win_rate_belongs_to_the_comparison():
    # The arm's runner writes nothing under results/; the epilogue then files the summary
    # as metadata and drops the empty dir (src/eval/run_eval.py::_publish).
    src = pathlib.Path("src/eval/capabilities/arena_hard/runner.py").read_text()
    assert "results_dir" not in src, "an arm must not write results"
    assert "arena_hard_judge" not in src, "judging belongs to pool.py"


def test_the_comparison_is_named_for_the_one_thing_every_arm_shares(tmp_path, monkeypatch):
    from src.eval.capabilities.arena_hard import pool as pool_mod

    runs = [
        _arm_run(tmp_path, "qwen36_tulu_100_0", "org/r", reference=True),
        _arm_run(tmp_path, "qwen36_difficult_advice_0", "org/a"),
        _arm_run(tmp_path, "qwen36_courtroom_716_0", "org/b"),
    ]
    wins = {"qwen36_difficult_advice_0": 0.6, "qwen36_courtroom_716_0": 0.4}

    def fake_judge(config, mode, arm):
        cfg = OmegaConf.load(config)
        judge_dir = pathlib.Path(cfg.output_dir) / "judging" / "20260906_000000"
        judge_dir.mkdir(parents=True, exist_ok=True)
        (judge_dir / f"judgment_{arm}.json").write_text(json.dumps(
            {"arm": arm, "baseline": "qwen36_tulu_100_0", "by_slice": {
                "hard_prompt": {"n_prompts": 400, "win_rate": wins[arm],
                                "tie_rate": 0.2, "loss_rate": 0.2}}}))

    monkeypatch.setattr(pool_mod.arena_hard_judge, "main", fake_judge)
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.vendor_dir = str(tmp_path / "vendor")

    summary = pool_mod.pool(runs, cfg, tmp_path / "pooled")
    assert summary["model_key"] == "vs_qwen36_tulu_100_0"
    assert eval_name("arena_hard", summary["model_key"], date="2026-09-06") == (
        "2026-09-06-ah-vs-qwen36-tulu-100-0")
    # Ranked; the reference is pooled-from but has no row of its own.
    assert [r["model_key"] for r in summary["leaderboard"]] == [
        "qwen36_difficult_advice_0", "qwen36_courtroom_716_0"]
    assert sum(r["reference_arm"] for r in summary["pooled_from"]) == 1
    # The comparison owns the results, and points at the arms rather than copying them.
    assert (tmp_path / "pooled" / "results" / "leaderboard.json").exists()
    pointers = json.loads((tmp_path / "pooled" / "metadata" / "sources.json").read_text())
    assert [p["repo"] for p in pointers] == [r["repo"] for r in runs]


def test_the_comparison_refuses_an_invocation_it_cannot_be_one_of(tmp_path):
    from src.eval.capabilities.arena_hard.pool import pool as pool_fn

    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    only_reference = [_arm_run(tmp_path, "r", "org/r", reference=True)]
    with pytest.raises(AssertionError, match="nothing to compare"):
        pool_fn(only_reference, cfg, tmp_path / "p1")

    no_reference = [_arm_run(tmp_path, "a", "org/a"), _arm_run(tmp_path, "b", "org/b")]
    with pytest.raises(AssertionError, match="exactly one baseline"):
        pool_fn(no_reference, cfg, tmp_path / "p2")
