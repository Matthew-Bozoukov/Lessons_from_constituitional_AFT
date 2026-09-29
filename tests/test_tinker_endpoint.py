# ABOUTME: Offline shim lifecycle and GPT-OSS SWE integration tests, independent of the optional SDK environment.
# ABOUTME: Check checkpoint identity, process ownership and unchanged default SWE overlays without model calls.

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.infra.endpoints import tinker


def test_resolved_tinker_port_is_the_port_that_run_eval_starts(monkeypatch):
    from src.eval.run_eval import _tinker_endpoint
    monkeypatch.setenv("TINKER_SHIM_PORT", "23456")
    spec = tinker.resolve_tinker_target("tinker://run/sampler_weights/model")
    captured = {}
    def start(checkpoint, **kwargs):
        captured.update(checkpoint=checkpoint, **kwargs)
        return "context"
    monkeypatch.setattr(tinker, "tinker_shim", start)
    cfg = OmegaConf.create({"tinker": {"reasoning": "high", "context_window": 64000, "max_tokens": 16000}})
    assert _tinker_endpoint(spec, cfg) == "context"
    assert captured["port"] == 23456
    assert captured["context_window"] == 64000 and captured["max_tokens"] == 16000
    assert captured["reasoning"] == "high"


@pytest.mark.parametrize("wrong_server", [False, True])
def test_readiness_is_bound_to_started_instance_and_checkpoint(monkeypatch, tmp_path, wrong_server):
    monkeypatch.setenv("TINKER_API_KEY", "synthetic-secret")
    state = {"terminated": False, "yielded": False}
    class Process:
        def __init__(self, argv, **kwargs):
            state.update(argv=argv, **kwargs)
        def poll(self):
            return None
        def terminate(self):
            state["terminated"] = True
        def wait(self, **kwargs):
            return 0
    def get(url, **kwargs):
        env = state["env"]
        assert kwargs["headers"]["Authorization"] == "Bearer synthetic-secret"
        entry = {"id": tinker.DEFAULT_BASE_MODEL, "checkpoint": env["TINKER_CKPT"],
                 "instance_id": "other-instance" if wrong_server else env["TINKER_SHIM_INSTANCE"],
                 "reasoning_effort": "medium", "context_window": 131072}
        return SimpleNamespace(status_code=200, json=lambda: {"data": [entry]})
    monkeypatch.setattr(tinker.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(tinker.subprocess, "Popen", Process)
    monkeypatch.setattr(tinker.requests, "get", get)
    def serve():
        with tinker.tinker_shim("tinker://run/sampler_weights/model", port=23457, log_dir=tmp_path):
            state["yielded"] = True
    if wrong_server:
        with pytest.raises(RuntimeError, match="another server identity"):
            serve()
        assert not state["yielded"]
    else:
        serve()
        assert state["yielded"]
    assert state["terminated"]
    assert "python" in Path(state["argv"][0]).name
    assert state["cwd"] == tinker.REPO_ROOT
    assert "synthetic-secret" not in (tmp_path / "shim_23457.log").read_text()


def test_swe_default_overlay_is_unchanged_and_tinker_controls_are_explicit(tmp_path):
    from src.eval.capabilities.swebench_mini.agent import build_overlay, rollout_env
    standard = OmegaConf.to_container(OmegaConf.load(build_overlay("http://model/v1", tmp_path)))
    assert standard == {"model": {"model_kwargs": {"api_base": "http://model/v1"}},
                        "environment": {"run_args": ["--rm", "--network", "none"]}}
    cfg = OmegaConf.load("configs/eval/swebench_mini/gptoss_tinker.yaml")
    tinker_overlay = OmegaConf.load(build_overlay("http://shim/v1", tmp_path, model_kwargs=dict(cfg.tinker_model_kwargs)))
    assert tinker_overlay.model.model_kwargs.max_tokens == 16384
    assert tinker_overlay.model.model_kwargs.drop_params is False
    assert cfg.dataset == "princeton-nlp/SWE-bench_Lite" and cfg.subset.fraction == 1.0
    env = rollout_env(registry=tmp_path / "registry.json", global_config_dir=tmp_path / "empty", api_key="fake-key")
    assert env["HOSTED_VLLM_API_KEY"] == "fake-key" and env["OPENAI_API_KEY"] == "fake-key"
