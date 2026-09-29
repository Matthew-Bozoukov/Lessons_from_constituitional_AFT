# ABOUTME: Arena smoke validates complete wiring without pretending four prompts calibrate judges.
# ABOUTME: Full calibration thresholds and generation diagnostics retain their separate meanings.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.capabilities.arena_hard import pool as pool_mod


def _fixture(tmp_path, monkeypatch, *, smoke=True, n_compared=4, passes=False):
    cfg = OmegaConf.load("configs/eval/arena_hard.yaml")
    cfg.smoke = smoke
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
    # These stubs isolate the four-prompt-vs-full calibration decision in pool().
    monkeypatch.setattr(pool_mod.arena_hard_runner, "validate_generation_protocols",
                        lambda *args: {"status": "compatible"}, raising=False)
    def validation(cfg):
        calls.append(("validation", int(cfg.judge_validation.n_questions)))
        return {"n_compared": n_compared, "passes": passes,
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
    assert ("judge", "candidate") in calls
    assert summary["leaderboard"][0]["passes"] is None
    saved = json.loads((tmp_path / "comparison/results/judge_validation.json").read_text())
    assert saved == report


def test_smoke_still_requires_every_validation_question(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, n_compared=3, passes=True)
    with pytest.raises(ValueError, match="incomplete coverage"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == [("validation", 4)]
    saved = json.loads((tmp_path / "comparison/results/judge_validation.json").read_text())
    assert saved["coverage_status"] == "incomplete"
    assert saved["passes"] is None


def test_full_calibration_failure_still_blocks_comparison(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False, n_compared=100, passes=False)
    with pytest.raises(ValueError, match="full calibration gate"):
        pool_mod.pool(runs, cfg, tmp_path / "comparison")
    assert calls == [("validation", 100)]


def test_full_calibration_cannot_silently_shrink_to_four(tmp_path, monkeypatch):
    cfg, runs, calls = _fixture(tmp_path, monkeypatch, smoke=False)
    cfg.judge_validation.n_questions = 4
    with pytest.raises(ValueError, match="requires 100"):
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
    assert cfg.judge.model == "google/gemini-3-flash-preview"
    assert cfg.judge_validation.reference_judge == "openai/gpt-4.1"
    assert cfg.judge_validation.n_questions == 100
    assert "staging" not in cfg and "absolute_benchmarks" not in cfg
    assert list(cfg.statistics.control_features) == ["length", "markdown"]
