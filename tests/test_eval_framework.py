# ABOUTME: Offline tests for the eval framework's pure logic: target resolution, thinking-mode
# ABOUTME: template pinning, registry shape, and dataset-card field enforcement.

import json

import pytest
from huggingface_hub.errors import EntryNotFoundError

from src.infra.endpoints.vllm import TargetSpec, _spec_from_files, pin_template
from src.eval import EVALS, EvalSpec
from src.infra.huggingface import REQUIRED_FIELDS, card_markdown
from src.model_profile import model_profile
QWEN36_PROFILE = model_profile("qwen36")

ADAPTER_CONFIG = {"base_model_name_or_path": "Qwen/Qwen3-32B", "r": 16}


def test_full_model_uses_template_default_mode():
    spec = _spec_from_files("Qwen/Qwen3-32B", None, None)
    assert spec == TargetSpec(hf_path="Qwen/Qwen3-32B", base_model="Qwen/Qwen3-32B",
                              adapter=False, mode="default", model_key="qwen3",  # canonical spelling (src/utils.py)
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
        # package is relative to src.eval and names its subarea (audits/ is exempt
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
    from src.infra.endpoints.vllm import plan_serving

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
    from src.infra.endpoints.vllm import plan_serving

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
    from src.infra.endpoints.vllm import plan_serving

    for mode, expected in (("think", "qwen3"), ("nothink", None), ("default", None)):
        plan = plan_serving(QWEN36_FACTS, {"context_window": 16384}, "m", mode)
        assert plan["reasoning_parser"] == expected, mode


def test_plan_serving_matches_tool_call_needs_to_family_parsers():
    # The eval declares the NEED; the family supplies which parser implements it.
    from src.infra.endpoints.vllm import plan_serving

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
    # Throughput-only shortfall: report it, serve without it. A family whose template
    # cannot cache prefixes gets the request reported, not refused.
    from src.infra.endpoints.vllm import plan_serving

    cannot = dict(QWEN36_FACTS, supports_prefix_caching=False)
    plan = plan_serving(cannot,
                        {"context_window": 16384, "reuses_long_prefixes": True},
                        "m", "think")
    assert plan["prefix_caching"] is False
    assert len(plan["warnings"]) == 1 and "reuses_long_prefixes" in plan["warnings"][0]
    # Qwen3.6 supports it (measured, docs/LOG.md 2026-08-07) and gets it, silently.
    assert QWEN36_FACTS["supports_prefix_caching"] is True
    plan = plan_serving(QWEN36_FACTS,
                        {"context_window": 16384, "reuses_long_prefixes": True},
                        "m", "think")
    assert plan["prefix_caching"] is True and plan["warnings"] == ()


def test_every_eval_config_declares_its_context_window():
    # The serving window decides truncation behaviour (CLAUDE.md gotcha 5), so every
    # eval config states the window it runs at — required, no hidden family default.
    # VllmServer._start enforces it at serve time; this catches it at test time.
    from omegaconf import OmegaConf

    from src.infra.endpoints.vllm import _EVAL_REQUIREMENT_KEYS

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
    from src.infra.endpoints.vllm import SshExec

    ex = SshExec("somehost", port=8000)
    wrapped = ex._with_env("vllm-serve")
    # The pod's .env, sourced shell-convention style; the driver's env is never sent.
    assert "set -a; [ -f /workspace/.env ]" in wrapped
    assert wrapped.endswith("vllm-serve")


def test_sshexec_push_hf_env_is_optin_minimal_and_never_overwrites(monkeypatch, tmp_path):
    from src.infra.endpoints.vllm import SshExec

    local = tmp_path / ".env"
    local.write_text("OPENROUTER_API_KEY=secret-or\nHF_TOKEN=hf_abc\nVAST_API_KEY=v\n"
                     "# WANDB_API_KEY=commented-out\nWANDB_PROJECT=lasr\nWANDB_ENTITY=\n")
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
    ex.push_hf_env(local)
    assert not any("hf_abc" in c for c in seen), "must not rewrite an existing remote .env"

    # Remote has none: HF_TOKEN, HF_ORG and whichever W&B variables are SET cross —
    # nothing else from the .env. HF_ORG is not a credential, and work run ON the host
    # cannot push without it; a commented-out or empty W&B line is not set.
    def fake_ssh(cmd, **kw):
        sent.append(cmd)
        return "no\n"

    monkeypatch.setattr(ex, "_ssh", fake_ssh)
    ex.push_hf_env(local)
    written = sent[-1]
    assert "hf_abc" in written and "HF_ORG=test-org" in written
    assert "WANDB_PROJECT=lasr" in written
    assert "WANDB_API_KEY" not in written and "WANDB_ENTITY" not in written
    assert "secret-or" not in written and "VAST" not in written
    assert "umask 077" in written

    local.write_text("HF_TOKEN=hf_abc\nWANDB_API_KEY=wb_key\n")
    ex.push_hf_env(local)
    assert "WANDB_API_KEY=wb_key" in sent[-1]


def test_sshexec_check_ready_errors_name_the_remedy(monkeypatch):
    # An unprepared host must name the ONE command that prepares one; a preflight that
    # names a script nobody has is worse than none. What it checks is vLLM, NOT a repo
    # clone: an eval pod holds one package, not this repository.
    from src.infra.endpoints.vllm import SshExec

    ex = SshExec("host", port=8000)
    monkeypatch.setattr(ex, "_ssh", lambda cmd, **kw: "NOVLLM\n")
    with pytest.raises(SystemExit, match=r"uv run runpod up --name <name> --eval"):
        ex.check_ready()
    monkeypatch.setattr(ex, "_ssh", lambda cmd, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SystemExit, match="RunPod remaps ports"):
        ex.check_ready()


def test_sshexec_remote_commands_get_uv_on_path():
    from src.infra.endpoints.vllm import SshExec

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


def test_resolve_target_api_endpoint_scheme():
    from src.infra.endpoints.vllm import VllmServer, resolve_target

    spec = resolve_target("openrouter:moonshotai/kimi-k2")
    assert spec.api_base == "https://openrouter.ai/api/v1"
    assert spec.api_key_env == "OPENROUTER_API_KEY"
    assert spec.base_model == "moonshotai/kimi-k2"          # id sent to the API
    assert spec.model_key == "openrouter_kimi_k2"           # provider-disambiguated
    assert spec.mode == "default" and not spec.adapter

    # ServedTarget exposes the OpenAI triple without booting vLLM.
    from pathlib import Path
    st = VllmServer(work_dir=Path("/tmp/_t"), port=8000).ensure(spec)
    assert st.is_api and st.base_url == spec.api_base
    assert st.model_name == "moonshotai/kimi-k2"

    with pytest.raises(ValueError, match="unknown API provider"):
        resolve_target("openroute:foo/bar")            # typo'd scheme, not an HF id
    with pytest.raises(AssertionError, match="names no model"):
        resolve_target("openrouter:")


def test_api_target_key_from_env_not_config(monkeypatch):
    from pathlib import Path

    from src.infra.endpoints.vllm import VllmServer, resolve_target

    st = VllmServer(work_dir=Path("/tmp/_t2"), port=8000).ensure(
        resolve_target("openrouter:openai/gpt-4o-mini"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-xyz")
    assert st.api_key == "sk-xyz"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AssertionError, match="OPENROUTER_API_KEY"):
        _ = st.api_key


def test_local_target_api_key_is_empty_sentinel():
    from pathlib import Path

    from src.infra.endpoints.vllm import TargetSpec, VllmServer

    spec = TargetSpec(hf_path="Qwen/Qwen3-32B", base_model="Qwen/Qwen3-32B",
                      # model_key is the canonical spelling (src/utils.py)
                      adapter=False, mode="default", model_key="qwen3", lora_rank=None)
    st = VllmServer(work_dir=Path("/tmp/_t3"), port=8000).ensure(spec)
    assert not st.is_api and st.api_key == "EMPTY"


def test_registry_marks_only_openai_client_evals_api_capable():
    # These evals reach the target purely through base_url/model/key. The criterion is
    # about the TARGET path alone: ctfish needs docker and is still API-capable, because
    # its containers hold a chess engine and a shell while the loop that calls the model
    # runs in the driver. What must stay False is an eval whose containers call the model
    # (odcv), or that relies on a served-model prefix, a LoRA swap or a pinned template
    # (agentic_misalignment, swebench_mini, internalization).
    assert {n for n, s in EVALS.items() if s.supports_api_target} == {
        "mmlu", "arena_hard", "psychosis", "moralbench", "ctfish", "mask"}


def test_publish_layout_contract(tmp_path):
    from src.eval.layout import assert_layout, publish_layout

    rollouts, results, metadata = publish_layout(tmp_path)
    assert (rollouts.name, results.name, metadata.name) == ("rollouts", "results", "metadata")
    assert all(d.is_dir() for d in (rollouts, results, metadata))
    publish_layout(tmp_path)  # idempotent

    (tmp_path / "README.md").write_text("card")
    assert_layout(tmp_path)  # the three dirs + README are the whole contract

    (tmp_path / "stray.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"stray root entries.*stray\.json"):
        assert_layout(tmp_path)


def test_every_target_is_named_before_any_of_them_runs(monkeypatch, tmp_path):
    # An arm ladder must not discover a name collision on the fourth target with three
    # runs already paid for — and the second run would publish over the first. The check
    # is a preflight over ALL targets, not per arm.
    import src.eval.run_eval as re_mod
    from src.infra.endpoints import vllm

    ran = []
    monkeypatch.setattr(re_mod, "resolve_target", lambda t: vllm.TargetSpec(
        hf_path=t, base_model="Qwen/Qwen3.6-27B", adapter=True, mode="think",
        model_key=t.split("/")[-1], lora_rank=64))
    monkeypatch.setattr(re_mod, "resolve", lambda name: lambda *a, **k: ran.append(1) or {})

    # Two arms whose names differ only in the date the law now strips: one eval run name,
    # two arms, and the second would land on top of the first.
    with pytest.raises(Exception, match="published twice"):
        re_mod.main(["--name", "mmlu",
                     "--target", "org/2026-08-31-qwen36-difficult-advice-0",
                     "org/2026-09-01-qwen36-difficult-advice-0"])
    assert not ran, "a run started before every name was checked"


def test_a_prior_run_resolves_as_an_arm_whose_answers_already_exist(monkeypatch, tmp_path):
    """An ah run is a valid target: its rollouts hold that model's answers.

    Identity is READ from the run's own metadata, not re-derived, so an arm reused as a
    reference a fortnight later carries the facts it carried the first time — which is
    what makes it comparable at all.
    """
    from src.infra.endpoints import vllm

    meta = tmp_path / "run_meta.json"
    meta.write_text(json.dumps({"target": "LASR-Callum/2026-09-04-qwen36-difficult-advice-0",
                                "base_model": "Qwen/Qwen3.6-27B", "mode": "think"}))
    def only_run_meta(repo, name, **kw):
        # A published run has no adapter_config: that miss is what sends resolve_target
        # to the dataset probe rather than straight to "full model".
        if name != "metadata/run_meta.json":
            raise EntryNotFoundError("no such file")
        return str(meta)

    monkeypatch.setattr(vllm, "hf_download", only_run_meta)
    monkeypatch.setattr(vllm, "_repo_sha", lambda path, repo_type="model": f"sha-of-{repo_type}")

    spec = vllm.resolve_target("LASR-Callum/2026-09-05-ah-qwen36-difficult-advice-0")
    assert spec.answers == "LASR-Callum/2026-09-05-ah-qwen36-difficult-advice-0"
    # Resolved to a commit at resolve time and carried on the spec, so run_meta can name
    # the exact answers it reused rather than the head of a repo that may move.
    assert spec.revision == "sha-of-dataset"
    assert spec.model_key == "qwen36_difficult_advice_0" and spec.mode == "think"
    assert not spec.adapter and spec.api_base is None

    # Nothing serves an answers arm, and reaching for an endpoint says so rather than
    # silently booting vLLM for a model that is not the point.
    served = vllm.ServedTarget(spec, server=None)
    assert served.is_answers
    with pytest.raises(AssertionError, match="ANSWERS target"):
        served.base_url


def test_a_repo_that_is_not_a_published_run_is_not_mistaken_for_one(monkeypatch):
    from src.infra.endpoints import vllm

    def no_such_file(repo, name, **kw):
        raise EntryNotFoundError("nope")

    monkeypatch.setattr(vllm, "hf_download", no_such_file)
    assert vllm.resolve_answers_target("org/some-plain-dataset") is None


def test_an_adapter_is_served_on_the_base_commit_it_was_trained_against():
    """A LoRA is a diff on a base; serving it on today's head of that base is a different
    model. The training stamp records the commit, and the spec carries it to vLLM."""
    stamped = _spec_from_files("org/2026-09-04-qwen36-8-da-10", ADAPTER_CONFIG,
                               {"thinking": True, "base_model_revision": "abc123"})
    assert stamped.base_revision == "abc123"
    assert stamped.base_revision_from == "training_meta"
    # An older adapter with no stamp leaves it unset here; resolve_target then falls back
    # to the base's head and records that it did.
    unstamped = _spec_from_files("org/old-adapter", ADAPTER_CONFIG, {"thinking": True})
    assert unstamped.base_revision is None and unstamped.base_revision_from is None
