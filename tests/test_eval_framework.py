# ABOUTME: Offline tests for the eval framework's pure logic: target resolution, thinking-mode
# ABOUTME: template pinning, registry shape, and dataset-card field enforcement.

import pytest

from src.endpoints.vllm_server import TargetSpec, _spec_from_files, pin_template
from src.eval import EVALS, EvalSpec
from src.huggingface import REQUIRED_FIELDS, card_markdown
from src.model_profile import QWEN36_PROFILE

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
        # package is relative to src.eval and names its subarea (vulnerabilities/ is exempt
        # from the contract, so nothing may register under it).
        assert spec.package.split(".")[0] in ("capabilities", "misalignment"), (name, spec.package)
        assert spec.config.startswith("configs/eval/"), (name, spec.config)


def test_card_markdown_enforces_required_fields():
    fields = {f: "x" for f in REQUIRED_FIELDS}
    text = card_markdown(fields)
    assert all(f"| `{f}` |" in text for f in REQUIRED_FIELDS)
    with pytest.raises(AssertionError, match="constitution"):
        card_markdown({f: "x" for f in REQUIRED_FIELDS if f != "constitution"})


def test_registry_runners_fulfill_the_contract_and_configs_exist():
    from inspect import signature
    from pathlib import Path

    from src.eval import resolve

    for name, spec in EVALS.items():
        run = resolve(name)  # imports the runner: a missing module or run() fails right here
        params = signature(run).parameters
        assert list(params)[:3] == ["target", "cfg", "out_dir"], name
        # Anything beyond the contract trio must be keyword-only WITH a default (the
        # `reference` pattern): run_eval passes extras as kwargs only to evals that
        # take them, and an eval without extras must be callable with exactly three.
        for extra in list(params)[3:]:
            p = params[extra]
            assert p.kind is p.KEYWORD_ONLY and p.default is not p.empty, (name, extra)
        assert Path(spec.config).exists(), (name, spec.config)


# The REAL facts, not a copy: if the profile changes, these tests are meant to notice.
QWEN36_FACTS = QWEN36_PROFILE.serving


def test_plan_serving_validates_requirements_against_facts():
    from src.endpoints.vllm_server import plan_serving

    # Happy path: psychosis-shaped request — big window, fewer slots than the cap.
    plan = plan_serving(QWEN36_FACTS, {"context_window": 40960, "concurrency": 12},
                        "m", "think")
    assert plan == {"context_window": 40960, "max_num_seqs": 12,
                    "reasoning_parser": "qwen3", "tool_call_parser": None,
                    "prefix_caching": False, "warnings": ()}
    # No concurrency request: serve at the family cap.
    assert plan_serving(QWEN36_FACTS, {"context_window": 16384}, "m",
                        "think")["max_num_seqs"] == 32
    # Unprofiled family: no ceiling/cap to violate, no parser.
    plan = plan_serving({"max_num_seqs": None}, {"context_window": 13312}, "m", "think")
    assert plan["max_num_seqs"] is None and plan["reasoning_parser"] is None

    with pytest.raises(SystemExit, match="declares no serving.context_window"):
        plan_serving(QWEN36_FACTS, {}, "m", "think")
    with pytest.raises(SystemExit, match="exceeds .*verified cap"):
        plan_serving(QWEN36_FACTS, {"context_window": 16384, "concurrency": 256},
                     "m", "think")
    # Facts are not writable from eval configs — a forged ceiling is an unknown key,
    # and a typo'd key errors instead of silently no-opping.
    with pytest.raises(SystemExit, match="unknown key"):
        plan_serving(QWEN36_FACTS, {"context_window": 16384,
                                    "verified_context_window": 999999}, "m", "think")
    with pytest.raises(SystemExit, match="unknown key"):
        plan_serving(QWEN36_FACTS, {"context_windw": 16384}, "m", "think")
    # ...and the closed set runs both ways: an eval's need cannot be smuggled into a
    # profile either.
    with pytest.raises(SystemExit, match="unknown key"):
        plan_serving({"max_num_seqs": 32, "needs_tool_calls": True},
                     {"context_window": 16384}, "m", "think")


def test_plan_serving_limits_the_window_only_at_the_trained_size():
    # The one hard window limit is the model's trained window, read from its config.json
    # (not transcribed into a profile). Anything below it is a KV-cache question about
    # the specific card, which vLLM answers at startup — refusing here on a number we
    # merely have not tried would refuse legitimate requests on absence of evidence.
    from src.endpoints.vllm_server import plan_serving

    facts = dict(QWEN36_FACTS, native_context_window=262144)
    # Far above anything booted, far below native: served, silently.
    plan = plan_serving(facts, {"context_window": 131072}, "m", "think")
    assert plan["context_window"] == 131072 and plan["warnings"] == ()
    # Past the trained positions: a real error no amount of GPU fixes.
    with pytest.raises(SystemExit, match="exceeds .*native window"):
        plan_serving(facts, {"context_window": 262145}, "m", "think")
    # Config unreadable (offline, gated repo): no limit imposed, vLLM is the backstop.
    plan = plan_serving(dict(QWEN36_FACTS, native_context_window=None),
                        {"context_window": 999999}, "m", "think")
    assert plan["context_window"] == 999999


def test_plan_serving_emits_reasoning_parser_in_think_mode_only():
    # A nothink stream carries no tags, so the parser's "reasoning comes first"
    # assumption would route the WHOLE answer into the reasoning field.
    from src.endpoints.vllm_server import plan_serving

    for mode, expected in (("think", "qwen3"), ("nothink", None), ("default", None)):
        plan = plan_serving(QWEN36_FACTS, {"context_window": 16384}, "m", mode)
        assert plan["reasoning_parser"] == expected, mode


def test_plan_serving_matches_tool_call_needs_to_family_parsers():
    # The eval declares the NEED; the family supplies which parser implements it.
    from src.endpoints.vllm_server import plan_serving

    plan = plan_serving(QWEN36_FACTS,
                        {"context_window": 16384, "needs_tool_calls": True}, "m", "think")
    assert plan["tool_call_parser"] == "qwen3_xml"
    # Not requested: never emitted, even though the family has a parser.
    assert plan_serving(QWEN36_FACTS, {"context_window": 16384}, "m",
                        "think")["tool_call_parser"] is None
    # Required but unverified for this family: fatal, because serving anyway scores 0
    # in a way indistinguishable from incapability.
    with pytest.raises(SystemExit, match="no verified tool_call_parser"):
        plan_serving({"max_num_seqs": None},
                     {"context_window": 13312, "needs_tool_calls": True}, "m", "think")


def test_plan_serving_reports_impossible_prefix_caching_without_failing():
    # Throughput-only shortfall: report it, serve without it. Qwen3.6 cannot cache
    # prefixes (vLLM forces it off on this arch), so the flag would be a no-op.
    from src.endpoints.vllm_server import plan_serving

    plan = plan_serving(QWEN36_FACTS,
                        {"context_window": 16384, "reuses_long_prefixes": True},
                        "m", "think")
    assert plan["prefix_caching"] is False
    assert len(plan["warnings"]) == 1 and "reuses_long_prefixes" in plan["warnings"][0]
    # A family that supports it gets it, silently.
    supports = dict(QWEN36_FACTS, supports_prefix_caching=True)
    plan = plan_serving(supports, {"context_window": 16384, "reuses_long_prefixes": True},
                        "m", "think")
    assert plan["prefix_caching"] is True and plan["warnings"] == ()


def test_every_eval_config_declares_its_context_window():
    # The serving window decides truncation behaviour (CLAUDE.md gotcha 5), so every
    # eval config states the window it runs at — required, no hidden family default.
    # VllmServer._start enforces it at serve time; this catches it at test time.
    from omegaconf import OmegaConf

    from src.endpoints.vllm_server import _EVAL_REQUIREMENT_KEYS

    for name, spec in EVALS.items():
        cfg = OmegaConf.load(spec.config)
        window = OmegaConf.select(cfg, "serving.context_window")
        assert window and int(window) > 0, (
            f"{name}: {spec.config} must declare serving.context_window")
        # Keys only, not a full plan_serving call: a config may legitimately declare a
        # requirement no family has met yet (swebench_mini's 65536 window), and that is
        # meant to fail loudly at serve time rather than silently here.
        declared = set(OmegaConf.to_container(cfg.serving, resolve=True))
        assert declared <= _EVAL_REQUIREMENT_KEYS, (
            f"{name}: {spec.config} declares non-requirement serving key(s) "
            f"{sorted(declared - _EVAL_REQUIREMENT_KEYS)} — family facts live in "
            "ModelProfile.serving, src/model_profile.py")


def test_odcv_bridge_url_rewrite():
    from src.eval.misalignment.odcv import runner as odcv_bench

    assert (_ := odcv_bench._bridge_url("http://localhost:8000/v1", "172.17.0.1")
            ) == "http://172.17.0.1:8000/v1"
    assert (odcv_bench._bridge_url("http://127.0.0.1:9000/v1", "host.docker.internal")
            == "http://host.docker.internal:9000/v1")


def test_odcv_container_host_address_is_platform_aware(monkeypatch):
    from src.eval.misalignment.odcv import runner as odcv_bench

    monkeypatch.setattr(odcv_bench.sys, "platform", "linux")
    assert odcv_bench.container_host_address() == "172.17.0.1"
    monkeypatch.setattr(odcv_bench.sys, "platform", "darwin")
    assert odcv_bench.container_host_address() == "host.docker.internal"


def test_docker_preflight_fails_clearly_without_docker(monkeypatch):
    from src.eval import docker

    monkeypatch.setattr(docker.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as e:
        docker.docker_preflight()
    message = str(e.value)
    # The error must say what broke, why docker evals need it, and where TO run instead.
    assert "docker" in message and "vast.ai" in message and "Fix:" in message


def test_docker_preflight_network_failure_names_the_runpod_trap(monkeypatch):
    import subprocess as sp

    from src.eval import docker

    monkeypatch.setattr(docker.shutil, "which", lambda _: "/usr/bin/docker")

    def fake_run(argv, **kwargs):
        ok = argv[:3] != ["docker", "network", "create"]
        return sp.CompletedProcess(argv, 0 if ok else 1, stdout="",
                                   stderr="operation not permitted")

    monkeypatch.setattr(docker.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as e:
        docker.docker_preflight()
    message = str(e.value)
    assert "CANNOT create networks" in message and "RunPod" in message


def test_agentic_misalignment_config_rewrite():
    from omegaconf import OmegaConf

    from src.eval.misalignment.agentic_misalignment.runner import _harness_config

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
    assert "set -a; [ -f /root/work/.env ]" in wrapped
    assert wrapped.endswith("uv run python -c x")


def test_sshexec_push_hf_token_is_optin_minimal_and_never_overwrites(monkeypatch, tmp_path):
    from src.endpoints.vllm_server import SshExec

    local = tmp_path / ".env"
    local.write_text("OPENROUTER_API_KEY=secret-or\nHF_TOKEN=hf_abc\nVAST_API_KEY=v\n")
    ex = SshExec("host", port=8000)
    sent = []

    # Remote already has a .env: leave it alone, and do NOT abort. Relaunching against a box
    # that was already provisioned is the normal case (crashed run, config tweak), so this
    # skips rather than failing an eval whose weights are already downloaded.
    seen: list[str] = []

    def already_has(cmd, **kw):
        seen.append(cmd)
        return "yes\n"

    monkeypatch.setattr(ex, "_ssh", already_has)
    ex.push_hf_token(local)
    assert not any("hf_abc" in c for c in seen), "must not rewrite an existing remote .env"

    # Remote has none: exactly HF_TOKEN crosses, nothing else from the .env.
    def fake_ssh(cmd, **kw):
        sent.append(cmd)
        return "no\n"

    monkeypatch.setattr(ex, "_ssh", fake_ssh)
    ex.push_hf_token(local)
    written = sent[-1]
    assert "hf_abc" in written and "secret-or" not in written and "VAST" not in written
    assert "umask 077" in written


def test_sshexec_check_ready_errors_name_the_bootstrap_script(monkeypatch):
    from src.endpoints.vllm_server import SshExec

    ex = SshExec("host", port=8000)
    monkeypatch.setattr(ex, "_ssh", lambda cmd, **kw: "NOUV\nNOREPO\n")
    with pytest.raises(SystemExit, match="bootstrap_pod.sh"):
        ex.check_ready()
    monkeypatch.setattr(ex, "_ssh", lambda cmd, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SystemExit, match="RunPod remaps ports"):
        ex.check_ready()


def test_sshexec_remote_commands_get_uv_on_path():
    from src.endpoints.vllm_server import SshExec

    wrapped = SshExec("host", port=8000)._with_env("uv run x")
    assert wrapped.startswith('export PATH="$HOME/.local/bin:$PATH"; ')


def test_derive_run_kwargs_come_from_the_run_signature():
    import importlib

    mod = importlib.import_module("src.eval.run_eval")

    def fake_run(target, cfg, out_dir, *, reference="", judge_seed=""):
        raise NotImplementedError

    # Keyword-only params become --flags; underscores map to dashes; absent flags are
    # simply not passed, so evals see their own defaults.
    assert mod.derive_run_kwargs(fake_run, ["--reference", "org/x"]) == {"reference": "org/x"}
    assert mod.derive_run_kwargs(fake_run, ["--judge-seed", "7"]) == {"judge_seed": "7"}
    assert mod.derive_run_kwargs(fake_run, []) == {}
    # A flag no eval declares is a hard error, never silently dropped.
    with pytest.raises(SystemExit, match="unknown arguments"):
        mod.derive_run_kwargs(fake_run, ["--bogus", "x"])
