# ABOUTME: Offline tests for the swebench eval: the pinned instance list, the config-to-inspect
# ABOUTME: mapping, and the .eval unpacking into the contract layout — no docker, no model.
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval import EVALS
from src.eval.capabilities.swebench import runner

CFG = "configs/eval/swebench/lite.yaml"


def _cfg(**over):
    return OmegaConf.merge(OmegaConf.load(CFG), OmegaConf.create(over))


def test_registry_entry_and_config():
    spec = EVALS["swebench"]
    assert spec.needs_docker and spec.networks_per_scenario == 1 and spec.key == "swe"
    cfg = OmegaConf.load(spec.config)
    assert cfg.serving.context_window == 131072 and cfg.serving.needs_tool_calls
    assert cfg.concurrency <= cfg.serving.concurrency


def test_instance_list_is_200_unique_ids_allocated_per_repo():
    spec = json.loads(Path(_cfg().instance_ids_file).read_text())
    ids = spec["instance_ids"]
    assert len(ids) == 200 and len(set(ids)) == 200
    assert sum(spec["per_repo"].values()) == 200
    for repo, n in spec["per_repo"].items():
        assert n <= spec["available_per_repo"][repo]
        assert sum(i.startswith(repo + "__") for i in ids) == n
    assert spec["arch"] == "arm64"


def test_instance_ids_limit_and_explicit_override():
    cfg = _cfg(limit=5)
    assert runner.instance_ids(cfg) == runner.instance_ids(_cfg())[:5]
    assert runner.instance_ids(_cfg(instance_ids="a__b-1,c__d-2")) == ["a__b-1", "c__d-2"]
    with pytest.raises(AssertionError, match="duplicates"):
        runner.instance_ids(_cfg(instance_ids="a__b-1,a__b-1"))


def test_eval_kwargs_carry_the_protocol(tmp_path):
    cfg = _cfg(limit=3, concurrency=7, arch="x86_64")
    ids = runner.instance_ids(cfg)
    kw = runner.eval_kwargs(cfg, ids, tmp_path)
    assert kw["sample_id"] == ids
    assert kw["max_sandboxes"] == kw["max_connections"] == kw["max_samples"] == 7
    assert kw["task_args"]["arch"] == "x86_64"
    assert kw["task_args"]["dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert kw["message_limit"] == cfg.message_limit and kw["time_limit"] == cfg.time_limit
    assert kw["token_limit"] is None, "no cumulative-token cap by default (it caps turns in disguise)"
    assert runner.eval_kwargs(_cfg(token_limit=500000), ids, tmp_path)["token_limit"] == 500000
    assert kw["reasoning_history"] == "all" and kw["temperature"] == 0.0
    assert runner.resolved_arch(_cfg(arch="auto")) in ("arm64", "x86_64")


def _sample(iid, *, value=None, error=None, limit=None, total=100.0, out_tok=1000):
    scores = {} if value is None else {"swe_bench_scorer": SimpleNamespace(
        value=value, explanation="tests", metadata={"model_patch": "diff --git a b"})}
    return SimpleNamespace(
        id=iid, epoch=1, scores=scores,
        error=SimpleNamespace(message=error) if error else None,
        limit=SimpleNamespace(type=limit) if limit else None,
        messages=[{"role": "user", "content": "fix it"},
                  {"role": "assistant", "content": [{"type": "reasoning", "reasoning": "think"},
                                                    {"type": "text", "text": "ok"}],
                   "tool_calls": [{"id": "1", "function": "bash", "arguments": {"cmd": "ls"}}]},
                  {"role": "tool", "content": "a.py"}],
        total_time=total, working_time=total * 0.9,
        model_usage={"m": SimpleNamespace(input_tokens=5000, output_tokens=out_tok, total_tokens=6000)})


def test_unpack_and_summarise(tmp_path):
    samples = [_sample("x__y-1", value=1.0, total=60), _sample("x__y-2", value=0.0, total=120),
               _sample("x__y-3", error="sandbox died", total=30),
               _sample("x__y-4", value=0.0, limit="time", total=1800, out_tok=60000)]
    rows = [runner.unpack_sample(s, tmp_path) for s in samples]
    assert (tmp_path / "x__y-1" / "transcript.json").exists()
    assert (tmp_path / "x__y-1" / "transcript.md").exists()
    assert (tmp_path / "x__y-1" / "model.patch").read_text() == "diff --git a b"
    assert not (tmp_path / "x__y-3" / "model.patch").exists()
    s = runner.summarise(rows, status="success", model="m", arch="arm64", requested=4, concurrency=8)
    assert s["n_attempted"] == 4 and s["n_scored"] == 3 and s["n_resolved"] == 1
    assert s["resolved_rate"] == 0.25 and s["resolved_rate_scored_only"] == pytest.approx(1 / 3)
    assert s["resolved_rate_ci95"]["ci_lower"] < 0.25 < s["resolved_rate_ci95"]["ci_upper"]
    assert s["n_errors"] == 1 and s["limits"] == {"time": 1}
    assert s["timing_s"]["max"] == 1800 and s["tokens"]["output_max"] == 60000
    p = runner.projection(s, n_total=200, concurrency=8)
    # work / concurrency: mean 502.5 s * 200 / 8 = 12,562 s = 209.4 min; the tail floor is 30 min.
    assert p["ideal_minutes"] == pytest.approx(209.4, abs=0.1) and p["tail_floor_minutes"] == 30.0
    assert p["estimate_minutes"] == p["ideal_minutes"]


def test_run_sets_the_endpoint_env_for_inspect_only_and_publishes(monkeypatch, tmp_path):
    seen = {}

    def fake_eval(task, *, model, **kw):
        seen["task"], seen["model"], seen["kw"] = task, model, kw
        seen["env"] = (os.environ.get("LASR_BASE_URL"), os.environ.get("LASR_API_KEY"))
        log_path = tmp_path / "fake.eval"
        log_path.write_text("log")
        return [SimpleNamespace(status="success", location=str(log_path),
                                samples=[_sample(i, value=1.0) for i in kw["sample_id"]])]

    import inspect_ai
    monkeypatch.setattr(inspect_ai, "eval", fake_eval)
    monkeypatch.delenv("LASR_BASE_URL", raising=False)
    target = SimpleNamespace(model_name="da-25", base_url="http://127.0.0.1:8000/v1", api_key="k")
    out = tmp_path / "run"
    summary = runner.run(target, _cfg(limit=2), out)
    assert seen["task"] == runner.TASK and seen["model"] == "openai-api/lasr/da-25"
    assert seen["env"] == ("http://127.0.0.1:8000/v1", "k")
    assert "LASR_BASE_URL" not in os.environ, "the endpoint env must not outlive the call"
    assert len(seen["kw"]["sample_id"]) == 2
    assert summary["n_resolved"] == 2 and "per_instance" not in summary
    assert (out / "results" / "results.json").exists() and (out / "results" / "per_instance.json").exists()
    assert (out / "metadata" / "protocol.json").exists() and (out / "metadata" / "fake.eval").exists()
    assert sorted(p.name for p in out.iterdir()) == ["metadata", "results", "rollouts"]
    assert len(list((out / "rollouts").iterdir())) == 2
