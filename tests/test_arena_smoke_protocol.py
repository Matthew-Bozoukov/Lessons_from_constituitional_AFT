# ABOUTME: Arena scoring requires complete primary judgments and reports auxiliary diagnostics.
# ABOUTME: Smoke and diagnostic comparisons do not certify scientific judge calibration.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.capabilities.arena_hard import pool as pool_mod


def _fixture(tmp_path, monkeypatch, *, smoke=True, n_compared=4, passes=False,
             policy="diagnostic", primary_complete=True):
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.smoke = smoke
    cfg.judge_validation.policy = policy
    cfg.vendor_dir = str(tmp_path / "vendor")
    Path(cfg.vendor_dir).mkdir()
    runs = []
    for name in ("reference", "candidate"):
        path = tmp_path / name
        (path / "metadata").mkdir(parents=True)
        (path / "rollouts").mkdir()
        (path / "rollouts/answers.jsonl").write_text("{}\n")
        (path / "metadata/sources.json").write_text(json.dumps({
            "arm": name, "reference_arm": name == "reference", "answers": {"generated": f"org/{name}"}}))
        runs.append({"model_key": name, "target": f"org/{name}", "mode": "think", "out_dir": str(path)})
    calls = []
    # Protocol certification and score parsing have their own concrete fixture tests.
    # These stubs isolate primary completion and the diagnostic-vs-gate policy.
    monkeypatch.setattr(pool_mod.arena_hard_runner, "validate_generation_protocols",
                        lambda *args: {"status": "compatible"}, raising=False)
    def validation(cfg):
        calls.append(("validation", int(cfg.judge_validation.n_questions)))
        return {"n_compared": n_compared, "passes": passes,
                "primary_complete": primary_complete,
                "thresholds_met_on_sample": passes,
                "verdict_agreement": .5, "win_rate_gap_pp": 25.0}
    monkeypatch.setattr(pool_mod.arena_hard_judge, "validate_judge", validation)
    def judge(config, mode, arm):
        calls.append(("judge", arm))
        resolved = OmegaConf.load(config)
        out = Path(resolved.output_dir) / "judging/20260929_000000"
        out.mkdir(parents=True)
        (out / f"judgment_{arm}.json").write_text(json.dumps({"limits": {"hard_prompt": 4, "creative_writing": 4}}))
        raw = (Path(resolved.vendor_dir) / "data" / resolved.bench_name /
               "model_judgment" / resolved.judge.model / f"{arm}.jsonl")
        raw.parent.mkdir(parents=True)
        raw.write_text("{}\n")
    monkeypatch.setattr(pool_mod.arena_hard_judge, "main", judge)
    monkeypatch.setattr(pool_mod, "_score_slices", lambda *args: {"hard_prompt": {
        "controlled": {"mean": .5, "ci_lower": .2, "ci_upper": .8, "n": 4},
        "passes": None if smoke else False}})
    return cfg, runs, calls


def test_smoke_disagreement_is_not_a_false_calibration_failure(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch)
    summary = pool_mod.pool(runs, cfg, tmp_path / "comparison")
    report = summary["judge_validation"]
    assert report["coverage_status"] == "complete"
    assert report["calibration_status"] == "not_assessed"
    assert report["passes"] is None
    assert report["thresholds_met_on_sample"] is False
    assert report["calibration_gate_applies"] is False
    assert report["diagnostic_only"] is True
    assert summary["report_version"] == 3
    assert summary["judge_policy"] == {
        "primary_judge": "openai/gpt-4.1", "primary_completion_required": True,
        "auxiliary_judge": "google/gemini-3-flash-preview",
        "auxiliary_enabled": True, "auxiliary_policy": "diagnostic"}
    assert ("judge", "candidate") in calls
    assert summary["leaderboard"][0]["passes"] is None
    saved = json.loads((tmp_path / "comparison/results/judge_validation.json").read_text())
    assert saved == report


def test_smoke_still_requires_every_primary_validation_question(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, n_compared=3, passes=True,
                               primary_complete=False)
    with pytest.raises(ValueError, match="primary judge has incomplete coverage"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == [("validation", 4)]
    saved = json.loads((tmp_path / "comparison/results/judge_validation.json").read_text())
    assert saved["coverage_status"] == "incomplete"
    assert saved["passes"] is None


def test_full_calibration_failure_still_blocks_comparison(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False, n_compared=100,
                               passes=False, policy="gate")
    with pytest.raises(ValueError, match="full calibration gate"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == [("validation", 100)]


def test_full_calibration_cannot_silently_shrink_to_four(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False, policy="gate")
    cfg.judge_validation.n_questions = 4
    with pytest.raises(ValueError, match="requires 100"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == []


@pytest.mark.parametrize("n_compared", [0, 98, 100])
def test_full_diagnostic_auxiliary_failure_or_disagreement_does_not_block_primary(
        tmp_path, monkeypatch, n_compared):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False,
                               n_compared=n_compared, passes=False)
    summary = pool_mod.pool(runs, cfg, tmp_path / "comparison")
    report = summary["judge_validation"]
    assert report["primary_complete"] is True
    assert report["n_expected"] == 100
    assert report["coverage_status"] == ("complete" if n_compared == 100 else "incomplete")
    assert report["calibration_status"] == "not_assessed"
    assert report["passes"] is None
    assert report["calibration_gate_applies"] is False
    assert ("judge", "candidate") in calls


def test_full_diagnostic_requires_complete_primary_coverage(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False, n_compared=99,
                               primary_complete=False)
    with pytest.raises(ValueError, match="primary judge has incomplete coverage"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == [("validation", 100)]


def test_primary_failure_keeps_report_and_raw_judgments(tmp_path, monkeypatch):
    cfg, runs, _ = _fixture(tmp_path, monkeypatch)
    class ValidationFailure(ValueError):
        report = {"n_compared": 3, "passes": None, "primary_complete": False,
                  "thresholds_met_on_sample": None,
                  "coverage": {"openai/gpt-4.1": {"status": "incomplete"}}}
    monkeypatch.setattr(pool_mod.arena_hard_judge, "JudgeValidationError",
                        ValidationFailure, raising=False)
    def fail(resolved):
        raw = (Path(resolved.vendor_dir) / "data" / resolved.bench_name /
               "model_judgment" / resolved.judge.model / "candidate.jsonl")
        raw.parent.mkdir(parents=True)
        raw.write_text('{"diagnostic": "retained failed attempt"}\n')
        raise ValidationFailure("primary judge has incomplete coverage")
    monkeypatch.setattr(pool_mod.arena_hard_judge, "validate_judge", fail)
    out = tmp_path / "comparison"
    with pytest.raises(ValidationFailure, match="primary judge has incomplete coverage"):
        pool_mod.pool(runs, cfg, out)
    saved = json.loads((out / "results/judge_validation.json").read_text())
    assert saved["primary_complete"] is False
    assert saved["coverage"] == ValidationFailure.report["coverage"]
    assert saved["calibration_status"] == "not_assessed"
    assert saved["passes"] is None
    assert (out / "rollouts/judge_validation/openai/gpt-4.1/candidate.jsonl").exists()
    assert not (out / "results/leaderboard.json").exists()


def test_gate_execution_failure_with_complete_cached_panel_cannot_report_a_pass(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False,
                               n_compared=100, policy="gate")
    report = {
        "n_compared": 100, "primary_complete": True, "passes": False,
        "thresholds_met_on_sample": True,
        "coverage": {
            str(cfg.judge.model): {"status": "complete", "n_complete": 100},
            str(cfg.judge_validation.reference_judge): {
                "status": "execution_failed", "n_complete": 100,
                "execution_error": {"type": "CalledProcessError", "returncode": 1}},
        },
    }
    def fail(resolved):
        raise pool_mod.arena_hard_judge.JudgeValidationError(
            "Arena-Hard auxiliary judge has incomplete paired coverage", report)
    monkeypatch.setattr(pool_mod.arena_hard_judge, "validate_judge", fail)
    out = tmp_path / "comparison"
    with pytest.raises(pool_mod.arena_hard_judge.JudgeValidationError):
        pool_mod.pool(runs, cfg, out)
    saved = json.loads((out / "results/judge_validation.json").read_text())
    assert saved["coverage_status"] == "complete"
    assert saved["coverage"] == report["coverage"]
    assert saved["thresholds_met_on_sample"] is True
    assert saved["passes"] is False
    assert saved["calibration_status"] == "not_assessed"
    assert calls == []
    assert not (out / "results/leaderboard.json").exists()


def test_gate_smoke_still_requires_complete_auxiliary_coverage(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, n_compared=3, policy="gate")
    with pytest.raises(ValueError, match="validation has incomplete coverage"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == [("validation", 4)]


def test_same_primary_and_auxiliary_judge_is_rejected(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch)
    cfg.judge_validation.reference_judge = cfg.judge.model
    with pytest.raises(ValueError, match="judges must be different"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == []


def test_unknown_judge_policy_is_rejected(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, policy="optional")
    with pytest.raises(ValueError, match="Unknown Arena-Hard judge validation policy"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == []


def test_report_carries_health_without_discarding_outputs_or_claiming_legacy_health(tmp_path, monkeypatch):
    cfg, runs, _ = _fixture(tmp_path, monkeypatch)
    metrics = {"by_slice": {"hard_prompt": {"degeneracy": {"truncation_rate": .75, "empty_answer_rate": .25}}}}
    (Path(runs[1]["out_dir"]) / "metadata/gen_gen_metrics.json").write_text(json.dumps(metrics))
    summary = pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert summary["generation_health"]["reference"]["status"] == "unavailable"
    health = summary["generation_health"]["candidate"]
    assert health["diagnostic_only"] is True
    assert health["by_slice"] == metrics["by_slice"]
    assert summary["leaderboard"][0]["n_prompts"] == 4


def test_active_config_preserves_the_effective_protocol_without_old_launch_instructions():
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    assert list(cfg.arms) == [] and cfg.baseline_arm is None
    assert dict(cfg.arm_defaults) == {"n_hard_prompt": 500, "n_creative_writing": 250}
    assert cfg.generation.max_tokens == 6000 and cfg.serving.context_window == 16384
    assert cfg.judge.model == "openai/gpt-4.1"
    assert dict(cfg.judge.extra_body) == {}
    assert cfg.judge_validation.reference_judge == "google/gemini-3-flash-preview"
    assert cfg.judge_validation.extra_body.reasoning.effort == "low"
    assert cfg.judge_validation.policy == "diagnostic"
    assert cfg.judge_validation.n_questions == 100
    assert "staging" not in cfg and "absolute_benchmarks" not in cfg
    assert list(cfg.statistics.control_features) == ["length", "markdown"]
