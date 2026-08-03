# ABOUTME: Offline tests for the eval framework's pure logic: target resolution, thinking-mode
# ABOUTME: template pinning, registry shape, and dataset-card field enforcement.

import pytest

from src.endpoints.vllm_server import TargetSpec, _spec_from_files, pin_template
from src.eval import EVALS, EvalSpec
from src.eval.publish import REQUIRED_FIELDS, card_markdown

ADAPTER_CONFIG = {"base_model_name_or_path": "Qwen/Qwen3-32B", "r": 16}


def test_full_model_uses_template_default_mode():
    spec = _spec_from_files("Qwen/Qwen3-32B", None, None)
    assert spec == TargetSpec(hf_path="Qwen/Qwen3-32B", base_model="Qwen/Qwen3-32B",
                              adapter=False, mode="default", model_key="Qwen3-32B",
                              lora_rank=None)


def test_adapter_mode_comes_from_training_meta():
    think = _spec_from_files("org/arm-lora", ADAPTER_CONFIG, {"thinking": True})
    nothink = _spec_from_files("org/arm-lora", ADAPTER_CONFIG, {"thinking": False})
    assert (think.mode, nothink.mode) == ("think", "nothink")
    assert think.base_model == "Qwen/Qwen3-32B" and think.lora_rank == 16


def test_adapter_without_stamp_is_a_hard_error():
    with pytest.raises(RuntimeError, match="training_meta.json"):
        _spec_from_files("org/legacy-lora", ADAPTER_CONFIG, None)
    with pytest.raises(AssertionError, match="thinking"):
        _spec_from_files("org/bad-lora", ADAPTER_CONFIG, {"mode": "think"})


def test_pin_template_shadows_request_kwargs():
    template = "{%- if enable_thinking %}T{% else %}N{% endif %}"
    pinned = pin_template(template, "nothink")
    # The pin is a top-level set BEFORE the original template, so it wins over any
    # enable_thinking a client passes per request.
    assert pinned.startswith("{%- set enable_thinking = false -%}\n")
    assert pin_template(template, "think").startswith("{%- set enable_thinking = true -%}\n")
    with pytest.raises(AssertionError):
        pin_template(template, "default")


def test_registry_specs_are_wellformed():
    assert EVALS, "registry is empty"
    for name, spec in EVALS.items():
        assert isinstance(spec, EvalSpec), name
        module, _, func = spec.runner.partition(":")
        assert module.startswith("src.eval.") and func, (name, spec.runner)
        assert spec.config.startswith("configs/eval/"), (name, spec.config)


def test_card_markdown_enforces_required_fields():
    fields = {f: "x" for f in REQUIRED_FIELDS}
    text = card_markdown(fields)
    assert all(f"| `{f}` |" in text for f in REQUIRED_FIELDS)
    with pytest.raises(AssertionError, match="constitution"):
        card_markdown({f: "x" for f in REQUIRED_FIELDS if f != "constitution"})


def test_registry_runner_modules_and_configs_exist():
    import importlib.util
    from pathlib import Path

    for name, spec in EVALS.items():
        module = spec.runner.split(":")[0]
        assert importlib.util.find_spec(module) is not None, (name, module)
        assert Path(spec.config).exists(), (name, spec.config)


def test_odcv_bridge_url_rewrite():
    from src.eval.misalignment import odcv_bench

    assert (_ := odcv_bench._bridge_url("http://localhost:8000/v1", "172.17.0.1")
            ) == "http://172.17.0.1:8000/v1"
    assert (odcv_bench._bridge_url("http://127.0.0.1:9000/v1", "host.docker.internal")
            == "http://host.docker.internal:9000/v1")


def test_odcv_container_host_address_is_platform_aware(monkeypatch):
    from src.eval.misalignment import odcv_bench

    monkeypatch.setattr(odcv_bench.sys, "platform", "linux")
    assert odcv_bench.container_host_address() == "172.17.0.1"
    monkeypatch.setattr(odcv_bench.sys, "platform", "darwin")
    assert odcv_bench.container_host_address() == "host.docker.internal"


def test_odcv_preflight_fails_clearly_without_docker(monkeypatch):
    from src.eval.misalignment import odcv_bench

    monkeypatch.setattr(odcv_bench.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as e:
        odcv_bench.docker_preflight()
    message = str(e.value)
    # The error must say what broke, why ODCV needs it, and where TO run instead.
    assert "docker" in message and "vast.ai" in message and "Fix:" in message


def test_odcv_preflight_network_failure_names_the_runpod_trap(monkeypatch):
    import subprocess as sp

    from src.eval.misalignment import odcv_bench

    monkeypatch.setattr(odcv_bench.shutil, "which", lambda _: "/usr/bin/docker")

    def fake_run(argv, **kwargs):
        ok = argv[:3] != ["docker", "network", "create"]
        return sp.CompletedProcess(argv, 0 if ok else 1, stdout="",
                                   stderr="operation not permitted")

    monkeypatch.setattr(odcv_bench.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as e:
        odcv_bench.docker_preflight()
    message = str(e.value)
    assert "CANNOT create networks" in message and "RunPod" in message


def test_agentic_misalignment_config_rewrite():
    from omegaconf import OmegaConf

    from src.eval.misalignment.agentic_misalignment import _harness_config

    cfg = OmegaConf.create({
        "global": {"models": ["vllm/qwen3"],
                   "concurrency": {"providers": {"vllm": 32}, "models": {"vllm/qwen3": 32}}},
        "expansions": [],
    })
    d = _harness_config(cfg, "vllm/my_arm", "my_arm_20260803")
    assert d["experiment_id"] == "my_arm_20260803"
    assert d["global"]["models"] == ["vllm/my_arm"]
    assert d["global"]["concurrency"]["models"] == {"vllm/my_arm": 32}


def test_sshexec_remote_commands_source_the_hosts_own_env():
    from src.endpoints.vllm_server import SshExec

    ex = SshExec("somehost", port=8000)
    wrapped = ex._with_env("uv run python -c x")
    # The pod's .env, sourced shell-convention style; the driver's env is never sent.
    assert wrapped.startswith("set -a; [ -f /root/work/.env ]")
    assert wrapped.endswith("uv run python -c x")
