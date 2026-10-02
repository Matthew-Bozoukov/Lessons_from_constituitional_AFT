# ABOUTME: Exercise behavioral runner boundaries with scripted clients and real prompt generation.
# ABOUTME: Covers declared Agentic conditions, Windows artifacts, and Qwen conversation history offline.

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf


def test_agentic_generator_honors_expansion_boundaries(tmp_path):
    from src.eval.misalignment.agentic_misalignment.runner import _HARNESS

    config = Path("configs/eval/agentic_misalignment.yaml").resolve()
    result = subprocess.run(
        [sys.executable, "scripts/generate_prompts.py", "--config", str(config),
         "--output", str(tmp_path)], cwd=_HARNESS,
        env=os.environ | {"PYTHONUTF8": "1"}, capture_output=True, check=True,
    )
    expected = {f"{scenario}_{goal}_{urgency}"
                for scenario in ("blackmail", "leaking")
                for goal in ("none-none", "explicit-america")
                for urgency in ("replacement", "restriction")}
    actual = {p.parent.name for p in tmp_path.glob("*/system_prompt.txt")}
    assert actual == expected, result.stdout.decode("utf-8")
    assert all((tmp_path / c / "email_content.txt").stat().st_size for c in expected)


def test_agentic_generator_does_not_cross_scenarios_between_expansions(tmp_path):
    from src.eval.misalignment.agentic_misalignment.runner import _HARNESS, planned_conditions

    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    cfg.expansions[0].variables.scenarios = ["blackmail"]
    cfg.expansions[0].variables.urgency_types = ["replacement"]
    cfg.expansions[1].variables.scenarios = ["leaking"]
    cfg.expansions[1].variables.urgency_types = ["restriction"]
    cfg.expansions.append({"enabled": False, "variables": {"scenarios": ["murder"],
                          "goal_types": ["none"], "urgency_types": ["restriction"]}})
    config = tmp_path / "fixture.yaml"
    OmegaConf.save(cfg, config)
    subprocess.run([sys.executable, "scripts/generate_prompts.py", "--config", str(config),
                    "--output", str(tmp_path / "prompts")], cwd=_HARNESS,
                   env=os.environ | {"PYTHONUTF8": "1"}, capture_output=True, check=True)
    expected = {"blackmail_none-none_replacement", "leaking_explicit-america_restriction"}
    assert planned_conditions(cfg) == expected
    assert {p.parent.name for p in (tmp_path / "prompts").glob("*/system_prompt.txt")} == expected


def test_agentic_failed_generation_keeps_prompt_evidence_in_run_directory(tmp_path, monkeypatch):
    from src.eval.misalignment.agentic_misalignment import runner

    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    real_step = runner._step
    calls = []

    def step(argv, env):
        calls.append(argv[1])
        assert env["PYTHONUTF8"] == "1"
        if argv[1] == "scripts/generate_prompts.py":
            real_step(argv, env)
        else:
            config = OmegaConf.load(argv[argv.index("--config") + 1])
            root = Path(config["global"].output_directory)
            (root / "failed_request.txt").write_text("fixture failure", encoding="utf-8")
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(runner, "_step", step)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run(_target(), cfg, tmp_path)
    assert calls == ["scripts/generate_prompts.py", "scripts/run_experiments.py"]
    root = tmp_path / "results/harness"
    assert len(list((root / "prompts").glob("*/system_prompt.txt"))) == 8
    assert (root / "failed_request.txt").exists()
    manifest = json.loads((tmp_path / "metadata/condition_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "declared-expansions-v1"
    assert len(manifest["conditions"]) == 8


def test_agentic_complete_runner_keeps_utf8_rollouts_and_declared_denominator(tmp_path, monkeypatch):
    from src.eval.misalignment.agentic_misalignment import runner

    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    cfg["global"].samples_per_condition = 1
    real_step = runner._step

    def step(argv, env):
        if argv[1] == "scripts/generate_prompts.py":
            real_step(argv, env)
        elif argv[1] == "scripts/run_experiments.py":
            config = OmegaConf.load(argv[argv.index("--config") + 1])
            root = Path(config["global"].output_directory)
            for condition in runner.planned_conditions(cfg):
                path = root / "models/fixture" / condition / "sample_001/response.json"
                path.parent.mkdir(parents=True)
                row = {"raw_response": "visible 世界", "reasoning_content": "trace",
                       "metadata": {"model": "vllm/fixture", "condition": condition,
                                    "sample_id": 1, "api_metadata": {"stop_reason": "stop"}}}
                path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        else:
            assert argv[1] == "scripts/classify_results.py"
            root = Path(argv[argv.index("--results-dir") + 1])
            assert root == (tmp_path / "results/harness").resolve()
            for path in (root / "models").rglob("response.json"):
                row = json.loads(path.read_text(encoding="utf-8"))
                row["classification"] = {"harmful_behavior": False, "protocol": "agentic-action-judge-v3"}
                path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(runner, "_step", step)
    summary = runner.run(_target(), cfg, tmp_path)
    assert summary["overall"] == {"n": 8, "harmful": 0, "rate": 0.0,
                                  "n_missing_classification": 0}
    assert summary["condition_protocol"] == "declared-expansions-v1"
    protocol = json.loads((tmp_path / "metadata/scoring_protocol.json").read_text(encoding="utf-8"))
    assert summary['scoring_protocol'] == protocol
    assert protocol['planned_trials'] == 8
    assert protocol['judge_input_protocol'] == 'visible-actions-no-cot-v1'
    assert len(protocol['prompt_sha256']) == 8
    transcripts = list((tmp_path / "rollouts").rglob("sample_001.md"))
    assert len(transcripts) == 8
    assert all("visible 世界" in p.read_text(encoding="utf-8") for p in transcripts)


@pytest.mark.parametrize("defect", ["missing", "unexpected", "empty"])
def test_agentic_prompt_preflight_stops_before_model_calls(tmp_path, monkeypatch, defect):
    from src.eval.misalignment.agentic_misalignment import runner

    cfg = OmegaConf.load("configs/eval/agentic_misalignment.yaml")
    calls = []

    def step(argv, env):
        calls.append(argv[1])
        assert argv[1] == "scripts/generate_prompts.py", "No model calls may precede preflight"
        conditions = sorted(runner.planned_conditions(cfg))
        if defect == "missing":
            conditions.pop()
        elif defect == "unexpected":
            conditions.append("blackmail_explicit-none_replacement")
        for condition in conditions:
            path = tmp_path / "results/harness/prompts" / condition
            path.mkdir(parents=True)
            for name in ("system_prompt.txt", "user_prompt.txt", "email_content.txt"):
                text = "" if defect == "empty" and name == "email_content.txt" else "fixture"
                (path / name).write_text(text, encoding="utf-8")

    monkeypatch.setattr(runner, "_step", step)
    with pytest.raises(ValueError, match="prompt"):
        runner.run(_target(), cfg, tmp_path)
    assert calls == ["scripts/generate_prompts.py"]


def _target():
    return NS(base_url="http://unused.invalid/v1", api_key="unused", model_name="fixture",
              spec=NS(mode="think", model_key="qwen36_fixture"))


def test_dictator_keeps_qwen_reasoning_in_followup_target_request(tmp_path, monkeypatch):
    from src.eval.misalignment.dictator import runner

    scenario = {"scenario_id": "fixture", "category": 1, "difficulty": "Multi-turn",
                "content_kind": "multi_turn", "prompt":
                "### Turn 1 (user)\nfirst\n### Turn 2 (user)\nsecond"}
    cfg = OmegaConf.load("configs/eval/dictator.yaml")
    cfg.generation.n_samples = 1
    monkeypatch.setattr(runner, "load_scenarios", lambda cfg, path: [scenario])
    requests, judge_prompts = [], []

    def generate(**kwargs):
        requests.append([dict(m) for m in kwargs["messages"]])
        return NS(choices=[NS(message=NS(content="visible reply", reasoning="private trace"),
                              finish_reason="stop")])

    def judge(model, messages, **kwargs):
        judge_prompts.append(messages)
        return NS(content='{"score": 4, "rationale": "fixture"}', finish_reason="stop")

    monkeypatch.setattr(runner, "OpenAI", lambda **kw: NS(chat=NS(completions=NS(create=generate))))
    monkeypatch.setattr(runner, "OpenRouterClient", lambda: NS(chat=judge))
    summary = runner.run(_target(), cfg, tmp_path)
    assert requests[1][1] == {"role": "assistant", "content": "visible reply",
                              "reasoning_content": "private trace"}
    assert "private trace" not in judge_prompts[0][0]["content"]
    assert summary["n_scored"] == 1


def test_psychosis_windows_unicode_artifacts_and_completion_marker(tmp_path, monkeypatch):
    from src.eval.misalignment.psychosis import runner
    from src.eval.misalignment.psychosis.judge import THERAPY_KEYS

    # Simulate Windows' cp1252 default on every platform; explicit UTF-8 stays intact.
    write_text = Path.write_text

    def windows_write(path, data, encoding=None, **kwargs):
        return write_text(path, data, encoding=encoding or "cp1252", **kwargs)

    monkeypatch.setattr(Path, "write_text", windows_write)
    cfg = OmegaConf.load("configs/eval/psychosis.yaml")
    cfg.turns = 1
    monkeypatch.setattr(runner, "load_characters", lambda selected: {"fixture": "fixture"})
    grade = {"mention_professional_help": "No", "pushback_rating": 1,
             "pushback_quote": "None", "delusion_confirmation_rating": 0,
             "delusion_confirmation_quote": "None", **{k: 0 for k in THERAPY_KEYS}}

    def remote(model, messages, **kwargs):
        content = ("<message>Hello.</message>" if model == cfg.redteam.model
                   else json.dumps(grade))
        return NS(content=content, finish_reason="stop")

    monkeypatch.setattr(runner, "OpenRouterClient", lambda: NS(chat=remote))
    monkeypatch.setattr(runner, "OpenAI", lambda **kw: NS(chat=NS(completions=NS(
        create=lambda **kwargs: NS(choices=[NS(message=NS(content="Hello 世界 👋"),
                                              finish_reason="stop")])))))
    summary = runner.run(_target(), cfg, tmp_path)
    assert summary["n_characters"] == 1
    checkpoint = json.loads((tmp_path / "rollouts/fixture.json").read_text(encoding="utf-8"))
    assert checkpoint["complete"] is True
    assert "世界 👋" in (tmp_path / "rollouts/fixture.md").read_text(encoding="utf-8")
