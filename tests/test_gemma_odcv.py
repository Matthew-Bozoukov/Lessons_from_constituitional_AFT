# ABOUTME: Gemma serving compatibility and ODCV protocol regression checks without inference.
# ABOUTME: Exercises both HF template layouts, sampling on the request, and baseline-free judging.
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from omegaconf import OmegaConf
from src.infra.endpoints import vllm
from src.model_profile import serving_params, model_key, model_profile
from src.eval.misalignment.odcv import odcv_judge

ROOT = Path(__file__).resolve().parents[1]


def test_remote_bootstrap_uses_locked_transformers():
    from src.infra import runpod
    pin = runpod._pinned_transformers()
    assert pin == "transformers==5.14.1"
    assert pin in runpod._bootstrap(None, (["google/gemma-4-31B-it"], None))
    assert pin in runpod.bootstrap_script("Qwen/Qwen3.6-27B", [("test", "/adapter")], mode="think",
        hf_token=None, max_len=28000, lora_rank=64, max_num_seqs=4,
        reasoning_parser="qwen3", tool_call_parser="qwen3_xml")


@pytest.mark.parametrize("embedded", [True, False])
def test_template_layout_keeps_revision_and_pins_thinking(tmp_path, monkeypatch, embedded):
    tpl = "{% if enable_thinking %}THINK{% endif %}"
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"chat_template": tpl} if embedded else {}))
    separate = tmp_path / "chat_template.jinja"
    separate.write_text(tpl)
    calls = []
    def download(model, filename, **kwargs):
        calls.append((filename, kwargs))
        return tmp_path / filename
    monkeypatch.setattr(vllm, "hf_download", download)
    server = vllm.VllmServer(tmp_path, executor=NS(write_file=lambda name, text: text))
    result = server._pinned_template_path("test/model", "think", "pinned-sha")
    assert result == vllm.pin_template(tpl, "think")
    assert all(kw == {"revision": "pinned-sha"} for _, kw in calls)
    assert len(calls) == (1 if embedded else 2)


def test_gemma_profile_serves_but_cannot_train():
    model = "google/gemma-4-31B-it"
    assert model_key(model) == "gemma4"
    with pytest.raises(ValueError, match="no verified thinking profile"):
        model_profile(model)
    cfg = OmegaConf.load(ROOT / "configs/eval/odcv/gemma4-1pass.yaml")
    plan = vllm.plan_serving(dict(serving_params(model), native_context_window=262144),
                            dict(cfg.serving), model, cfg.mode)
    assert plan["reasoning_parser"] == plan["tool_call_parser"] == "gemma4"
    assert plan["context_window"] == 28000 and plan["max_num_seqs"] == 64
    assert cfg.temperature == .7 and cfg.sampling.top_k == 64


@pytest.mark.parametrize("sampling", [{}, {"top_k": 64, "top_p": .95}])
def test_executor_sends_sampling_without_changing_temperature(monkeypatch, sampling):
    executor = ROOT / "src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor"
    monkeypatch.syspath_prepend(str(executor))
    monkeypatch.setenv("OPENAI_TEMPERATURE", ".7")
    monkeypatch.setenv("OPENAI_SAMPLING_PARAMS", json.dumps(sampling))
    spec = importlib.util.spec_from_file_location("gemma_agent_test", executor / "agent_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Captured(BaseException):
        pass
    received = {}
    def create(**kwargs):
        received.update(kwargs)
        raise Captured()
    state = NS(iter=0, hooks=NS(model="base", on_event=lambda *a, **kw: None), traj=[], tools=[])
    with pytest.raises(Captured):
        module.agent_step(state, NS(chat=NS(completions=NS(create=create))))
    assert received["temperature"] == .7
    if sampling:
        assert received["extra_body"] == sampling
    else:
        assert "extra_body" not in received


def test_new_family_judges_without_invented_published_baseline(tmp_path, monkeypatch):
    cfg = OmegaConf.create(dict(bench_dir=str(tmp_path), model="base", model_key="gemma4",
                               published_key=None, judges={"fake": "fake"}))
    config = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config)
    items = [(v, "Demo/rollout_000", tmp_path / "record") for v in ("mandated", "incentivized")]
    monkeypatch.setattr(odcv_judge, "_load_upstream_judge", lambda _: NS())
    monkeypatch.setattr(odcv_judge, "transcripts", lambda *a: items)
    monkeypatch.setattr(odcv_judge, "openrouter_usage", lambda **kw: None)
    monkeypatch.setattr(odcv_judge, "score_with_judge", lambda *a, **kw: {
        f"{v}/{s}": {"score": 1, "reasoning": "fixture"} for v, s, _ in items})
    monkeypatch.setattr(odcv_judge, "load_published_medians", lambda *a: pytest.fail("No baseline requested"))
    odcv_judge.main(str(tmp_path), str(config))
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["n_judged"] == 2 and result["ours"]["overall"]["mr_pct"] == 0
    assert result["published"] is result["delta_mr_pct"] is result["published_within_our_ci"] is None
