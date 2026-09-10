# ABOUTME: Offline integration tests for complete deliberative generation, checkpoints, and Hub layout.
# ABOUTME: Mock paid sampling and Hugging Face while exercising real context/export and cache code.

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.data.synth.deliberative_alignment import pipeline
from src.infra.endpoints.openrouter import ChatResult


MODEL = "qwen/qwen3.6-27b"
PIN = {"order": ["alibaba"], "allow_fallbacks": False}
SOURCE = {"repo": "source/final-synth", "revision": "a" * 40, "file": "dataset.jsonl"}


def response(**changes):
    return ChatResult(**{
        "content": "A thoughtful answer.", "reasoning_content": "Constitution principle 1 applies.",
        "prompt_tokens": 100, "completion_tokens": 50, "finish_reason": "stop",
        "provider": "Alibaba", "cost": 0.002, "response_id": "gen-1", "response_model": MODEL,
        **changes,
    })


@pytest.fixture
def environment(monkeypatch, tmp_path):
    constitution = tmp_path / "constitution.md"
    constitution.write_text("FULL CONSTITUTION α", encoding="utf-8")
    cfg = {
        "method": "deliberative_alignment", "pipeline": "delib",
        "source": {"repo": SOURCE["repo"], "revision": SOURCE["revision"]},
        "constitution": str(constitution), "model": MODEL,
        "sampling": {"temperature": 0.8, "max_tokens": 8192,
                     "reasoning": {"enabled": True, "exclude": False}, "seed": 7},
        "generation_prompt": "Consult the entire constitution:\n{constitution}\nThen reason and answer.",
        "workers": 2, "budget_usd": 1.0, "output_dir": str(tmp_path / "runs"),
        "hf_push": False, "hf_private": True,
    }
    records = [{"id": str(i), "source_row": i,
                "messages": [{"role": "system", "content": "Original system context."},
                             {"role": "user", "content": f"Final prompt {i % 2}"}],
                "metadata": {"scenario_id": f"scenario-{i}", "trait_id": "t1"}}
               for i in range(3)]
    calls = []
    behavior = {"fn": lambda kwargs: response()}

    class FakeClient:
        def chat(self, **kwargs):
            calls.append(deepcopy(kwargs))
            return behavior["fn"](kwargs)

    monkeypatch.setattr(pipeline, "load_prompts", lambda source: (deepcopy(records), deepcopy(SOURCE)))
    monkeypatch.setattr(pipeline, "OpenRouterClient", FakeClient)
    monkeypatch.setattr(pipeline, "git_sha", lambda: "b" * 40)
    monkeypatch.setattr(pipeline, "origin_url", lambda: "https://github.com/example/lasr")
    monkeypatch.setattr(pipeline, "provider_pin", lambda model: deepcopy(PIN))
    monkeypatch.setattr(pipeline, "provider_price", lambda model: {"in": 0.45, "out": 2.7})
    monkeypatch.setattr("src.infra.huggingface.hf_token", lambda: None)
    return SimpleNamespace(cfg=cfg, records=records, calls=calls, behavior=behavior,
                           constitution=constitution, tmp=tmp_path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_complete_run_preserves_exact_membership_context_and_resume(environment, monkeypatch):
    env = environment
    before = deepcopy(env.records)
    manifest = pipeline.run(env.cfg)
    run_dir = Path(manifest["run_dir"])
    rows = read_rows(run_dir / "dataset.jsonl")
    assert manifest["status"] == "complete"
    assert manifest["source"] == SOURCE
    assert manifest["source_rows"] == manifest["selected_rows"] == manifest["completed_rows"] == 3
    assert len(env.calls) == len(rows) == 3  # duplicate user text is still a separate final source row
    assert env.records == before
    for original, row in zip(env.records, rows, strict=True):
        assert row["messages"][:-1] == original["messages"]
        assert row["messages"][-1]["reasoning_content"] == "Constitution principle 1 applies."
        assert row["supervise"] == "final"
        assert "FULL CONSTITUTION" not in json.dumps(row["messages"][:-1])
    for call in env.calls:
        assert "FULL CONSTITUTION α" in call["messages"][0]["content"]
        assert call["model"] == MODEL
        assert call["extra_body"] == {"reasoning": {"enabled": True, "exclude": False}}
        assert call["seed"] == 7
    assert manifest["usage"]["total_usd"] == pytest.approx(0.006)
    assert manifest["usage"]["calls_with_usage"] == 3

    monkeypatch.setattr(pipeline, "load_prompts", lambda source: pytest.fail("resume refetched prompts"))
    env.behavior["fn"] = lambda _: pytest.fail("resume regenerated successful rows")
    resumed = pipeline.run(env.cfg, resume=str(run_dir))
    assert resumed["status"] == "complete"
    assert read_rows(run_dir / "dataset.jsonl") == rows
    assert resumed["usage"] == manifest["usage"]
    assert len(resumed["commands"]) == 2


def test_invalid_completion_checkpoints_usage_then_resume_only_missing(environment):
    env = environment
    env.behavior["fn"] = lambda kwargs: response(finish_reason="length") \
        if kwargs["messages"][-1]["content"] == "Final prompt 1" else response()
    with pytest.raises(RuntimeError, match="Generation failed"):
        pipeline.run(env.cfg)
    run_dir = next((env.tmp / "runs").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["completed_rows"] == 1
    assert manifest["usage"]["total_usd"] == pytest.approx(0.004)
    assert not (run_dir / "dataset.jsonl").exists()
    attempts = read_rows(run_dir / "generations.partial.jsonl")
    assert len(attempts) == 2
    assert any(attempt.get("error", "").startswith("Incomplete completion") for attempt in attempts)

    env.behavior["fn"] = lambda _: response()
    resumed = pipeline.run({**env.cfg, "workers": 1, "budget_usd": 2}, resume=str(run_dir))
    assert resumed["completed_rows"] == 3
    assert len(env.calls) == 4  # one failed call, three distinct successful source rows
    assert resumed["usage"]["calls_with_usage"] == 4
    assert resumed["usage"]["total_usd"] == pytest.approx(0.008)
    assert len(read_rows(run_dir / "dataset.jsonl")) == 3


def test_budget_failure_preserves_success_and_allows_budget_raise(environment):
    env = environment
    env.cfg.update(workers=1, budget_usd=0.001)
    with pytest.raises(RuntimeError, match="budget reached"):
        pipeline.run(env.cfg)
    run_dir = next((env.tmp / "runs").iterdir())
    assert len(env.calls) == 1
    assert not (run_dir / "dataset.jsonl").exists()
    resumed = pipeline.run({**env.cfg, "budget_usd": 1}, resume=str(run_dir))
    assert resumed["completed_rows"] == 3
    assert len(env.calls) == 3


def test_resume_refuses_constitution_or_prompt_snapshot_changes(environment):
    env = environment
    manifest = pipeline.run(env.cfg)
    env.constitution.write_text("CHANGED CONSTITUTION", encoding="utf-8")
    with pytest.raises(ValueError, match="config/constitution/provider changed"):
        pipeline.run(env.cfg, resume=manifest["run_dir"])
    env.constitution.write_text("FULL CONSTITUTION α", encoding="utf-8")
    prompt_file = Path(manifest["run_dir"]) / "stage_1_prompts.jsonl"
    altered = deepcopy(env.records)
    altered[0]["messages"][-1]["content"] = "Wrong prompt"
    prompt_file.write_text("".join(json.dumps(record) + "\n" for record in altered))
    with pytest.raises(ValueError, match="prompt snapshot changed"):
        pipeline.run(env.cfg, resume=manifest["run_dir"])
    assert len(env.calls) == 3


def test_wrong_provider_pin_fails_before_generation(environment, monkeypatch):
    monkeypatch.setattr(pipeline, "provider_pin", lambda model: {
        "order": ["alibaba"], "allow_fallbacks": True})
    with pytest.raises(ValueError, match="pinned to Alibaba"):
        pipeline.run(environment.cfg)
    assert environment.calls == []


@pytest.mark.parametrize("changes,message", [
    ({"provider": "DeepInfra"}, "Unexpected serving provider"),
    ({"finish_reason": "length"}, "Incomplete completion"),
    ({"reasoning_content": ""}, "Missing native reasoning"),
    ({"content": ""}, "Missing final answer"),
    ({"finish_reason": "tool_calls"}, "tool_calls finish reason"),
])
def test_invalid_completion_is_not_a_training_target(environment, changes, message):
    with pytest.raises(ValueError, match=message):
        pipeline._completion(response(**changes), environment.records[0])


def test_separate_native_trace_wins_over_think_example_in_answer(environment):
    answer = "An example of markup is <think>example content</think>. Keep it literal."
    completion = pipeline._completion(response(content=answer), environment.records[0])
    assert completion["reasoning_content"] == "Constitution principle 1 applies."
    assert completion["content"] == answer


def test_usage_prefers_actual_cost_and_marks_estimates():
    attempts = [
        {"response": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0}},
        {"response": {"prompt_tokens": 100, "completion_tokens": 50, "cost": None}},
        {"error": "APITimeoutError"},
    ]
    usage = pipeline._usage(attempts, {"in": 1, "out": 2})
    assert usage == {"calls_with_usage": 2, "calls_without_usage": 1,
                     "prompt_tokens": 200, "completion_tokens": 100,
                     "total_usd": pytest.approx(0.0002), "estimated_cost_calls": 1}


def test_hub_final_export_has_contract_layout_and_all_prompts(environment, monkeypatch):
    env = environment
    env.cfg["hf_push"] = True
    remote = {}

    def upload_file(*, path_or_fileobj, path_in_repo, **kwargs):
        remote[path_in_repo] = path_or_fileobj if isinstance(path_or_fileobj, bytes) \
            else Path(path_or_fileobj).read_bytes()

    def create_commit(*, operations, **kwargs):
        for operation in operations:
            upload_file(path_or_fileobj=operation.path_or_fileobj,
                        path_in_repo=operation.path_in_repo)

    hub = SimpleNamespace(repo_exists=lambda *a, **kw: False,
                          upload_file=upload_file, create_commit=create_commit)
    monkeypatch.setenv("HF_ORG", "offline-tests")
    monkeypatch.setattr(pipeline, "hf_api", lambda: hub)
    monkeypatch.setattr(pipeline.StageCache, "_hf", lambda self: hub)
    manifest = pipeline.run(env.cfg)
    assert manifest["hf_repo"].startswith("offline-tests/")
    assert {"stages/stage_1_prompts.jsonl", "stages/stage_2_responses.jsonl",
            "stages/stage_3_export_sft.jsonl", "dataset.jsonl", "manifest.json",
            "generation_prompt.txt", "generations.partial.jsonl", "README.md"} <= remote.keys()
    rows = [json.loads(line) for line in remote["dataset.jsonl"].decode().splitlines()]
    assert len(rows) == len(env.records) == 3
    assert [row["metadata"]["scenario_id"] for row in rows] == ["scenario-0", "scenario-1", "scenario-2"]
    assert all("FULL CONSTITUTION" not in json.dumps(row["messages"][:-1]) for row in rows)
    card = yaml.safe_load(remote["README.md"].decode().split("---", 2)[1])
    assert next(config for config in card["configs"] if config.get("default")) == {
        "config_name": "dataset", "data_files": "dataset.jsonl", "default": True}
    assert json.loads(remote["manifest.json"])["status"] == "complete"
