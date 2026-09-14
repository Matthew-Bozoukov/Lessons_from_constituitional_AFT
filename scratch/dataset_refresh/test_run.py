# ABOUTME: Offline adversarial tests of dataset refresh budget accounting, provenance and acceptance.
# ABOUTME: Mock every provider call and source download; verify exact selection without paid generation.
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
import threading

import pytest
from omegaconf import OmegaConf
from tenacity import retry, stop_after_attempt

from scratch.dataset_refresh import run as mod
from src.infra.endpoints.openrouter import ChatResult


MODEL = "test/model"
REQUEST = dict(model=MODEL, messages=[{"role": "user", "content": "hello"}],
               temperature=0, max_tokens=100)


@pytest.fixture(autouse=True)
def no_real_provider(monkeypatch):
    monkeypatch.setattr(mod, "provider_price", lambda _: {"in": 100., "out": 100.})
    monkeypatch.setattr(mod, "OpenRouterClient", lambda: pytest.fail("Real provider construction forbidden"))
    monkeypatch.setattr(mod, "hf_hub_download", lambda *a, **k: pytest.fail("Network source access forbidden"))


def result(**overrides):
    return replace(ChatResult(content="ok", prompt_tokens=2, completion_tokens=3,
                              finish_reason="stop", cost=.001, provider="test"), **overrides)


def reserve():
    bound = len(json.dumps(REQUEST["messages"], ensure_ascii=False).encode()) + 2048
    return (1.25 * bound * 100 + 100 * 100) / 1e6


def test_concurrent_clients_cannot_both_spend_the_same_remaining_budget(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def held_send(**kw):
        calls.append(kw)
        entered.set()
        assert release.wait(5)
        return result()

    first = mod.BudgetClient(tmp_path, reserve() * 1.5, {MODEL}, send=held_send)
    second = mod.BudgetClient(tmp_path, reserve() * 1.5, {MODEL}, send=held_send)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_job = pool.submit(first.chat, **REQUEST)
        assert entered.wait(5)
        second_job = pool.submit(second.chat, **REQUEST)
        try:
            with pytest.raises(mod.BudgetStop):
                second_job.result(timeout=5)
            assert len(calls) == 1
            assert sum(e["charged_or_reserved_usd"] for e in first.entries()) == pytest.approx(reserve())
        finally:
            release.set()
        assert first_job.result(timeout=5).content == "ok"
    assert len(first.entries()) == 1


def test_failed_physical_request_is_not_retried_or_refunded(tmp_path):
    calls = []

    def fail(**kw):
        calls.append(kw)
        raise TimeoutError("Billing unknown")

    client = mod.BudgetClient(tmp_path, reserve() * 1.5, {MODEL}, send=fail)
    with pytest.raises(TimeoutError):
        client.chat(**REQUEST)
    assert len(calls) == 1
    entry = client.entries()[0]
    assert entry["status"] == "uncertain_failure"
    assert entry["charged_or_reserved_usd"] == pytest.approx(reserve())
    with pytest.raises(mod.BudgetStop):
        client.chat(**REQUEST)
    assert len(calls) == 1
    raw = json.loads(next((tmp_path / "raw_calls").glob("*.json")).read_text())
    assert raw["request"] == REQUEST
    assert raw["accounting"]["exception_type"] == "TimeoutError"


def test_default_transport_retry_policy_is_reduced_to_one_physical_call(tmp_path, monkeypatch):
    calls = []

    class FakeAPI:
        @retry(stop=stop_after_attempt(6), reraise=True)
        def chat(self, **kw):
            calls.append(kw)
            raise TimeoutError("one physical request")

    monkeypatch.setattr(mod, "OpenRouterClient", FakeAPI)
    client = mod.BudgetClient(tmp_path, 10, {MODEL})
    with pytest.raises(TimeoutError):
        client.chat(**REQUEST)
    assert len(calls) == len(client.entries()) == 1


def test_reported_cost_settles_exactly_and_raw_response_is_preserved(tmp_path):
    response = result(cost=.02, response_id="request-id")
    client = mod.BudgetClient(tmp_path, 10, {MODEL}, send=lambda **kw: response)
    assert client.chat(**REQUEST) == response
    entry = client.entries()[0]
    assert entry["status"] == "settled"
    assert entry["charged_or_reserved_usd"] == .02
    assert entry["api_reported_cost_usd"] == .02
    raw = json.loads(next((tmp_path / "raw_calls").glob("*.json")).read_text())
    assert raw["response"]["response_id"] == "request-id"


def test_missing_reported_cost_is_explicitly_estimated_from_valid_usage(tmp_path):
    client = mod.BudgetClient(tmp_path, 10, {MODEL}, send=lambda **kw: result(cost=None))
    client.chat(**REQUEST)
    entry = client.entries()[0]
    assert entry["api_reported_cost_usd"] is None
    assert entry["charged_or_reserved_usd"] == entry["conservative_estimate_usd"]
    assert entry["charged_or_reserved_usd"] == pytest.approx((1.25 * 2 * 100 + 3 * 100) / 1e6)


@pytest.mark.parametrize("overrides", [
    {"cost": float("nan")}, {"cost": float("inf")}, {"cost": -1},
    {"cost": 1}, {"prompt_tokens": 0, "cost": None},
    {"completion_tokens": -1}, {"completion_tokens": 101},
    {"prompt_tokens": None}, {"completion_tokens": "3"},
])
def test_invalid_usage_halts_without_poisoning_or_releasing_reservation(tmp_path, overrides):
    calls = []

    def send(**kw):
        calls.append(kw)
        return result(**overrides)

    client = mod.BudgetClient(tmp_path, 10, {MODEL}, send=send)
    with pytest.raises(mod.BudgetStop):
        client.chat(**REQUEST)
    entry = client.entries()[0]
    assert math.isfinite(entry["charged_or_reserved_usd"])
    assert entry["charged_or_reserved_usd"] >= reserve()
    with pytest.raises(mod.BudgetStop):
        client.chat(**REQUEST)
    assert len(calls) == 1


@pytest.mark.parametrize("cap", [0, -1, 250.01, float("inf"), float("nan")])
def test_cap_cannot_exceed_authorization_or_be_nonfinite(tmp_path, cap):
    with pytest.raises(ValueError):
        mod.BudgetClient(tmp_path, cap, {MODEL}, send=lambda **kw: result())


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_incomplete_output_is_charged_but_never_returned_as_valid(tmp_path, finish):
    client = mod.BudgetClient(tmp_path, 10, {MODEL}, send=lambda **kw: result(finish_reason=finish))
    with pytest.raises(ValueError, match="incomplete|filtered"):
        client.chat(**REQUEST)
    assert client.entries()[0]["charged_or_reserved_usd"] == .001


def test_flat_acceptance_requires_literal_booleans_and_bounded_integer_stakes():
    cfg = {"acceptance": {"required_true": ["correct", "moral"], "stakes_max": 2}}
    good = {"correct": True, "moral": True, "stakes": 2}
    assert mod.acceptance(good, cfg)
    for broken in ({**good, "correct": 1}, {**good, "moral": "true"},
                   {**good, "stakes": True}, {**good, "stakes": 2.0},
                   {**good, "stakes": 3}, {**good, "stakes": -1},
                   {"correct": True, "stakes": 1}, None):
        assert not mod.acceptance(broken, cfg)


def test_nonmoral_acceptance_requires_all_nested_gates_and_overall_acceptance():
    cfg = mod.load_config("configs/data/synth/nonmoral-advice.yaml")
    names = ["advice_to_human", "nonmoral", "benign_subject", "self_contained", "grounded",
             "genuine_tension", "detailed_deliberation", "practical_recommendation",
             "constitution_compatible", "no_training_leakage", "metadata_consistent"]
    good = {"accepted": True, "gates": dict.fromkeys(names, True), "issues": [],
            "deciding_detail": "the short-lived artifact", "reasoning_move": "tradeoff",
            "recommendation_kind": "narrow_adjustment", "substantive_reasoning_word_count": 500}
    assert mod.acceptance(good, cfg)
    assert not mod.acceptance({**good, "accepted": False}, cfg)
    for key in names:
        assert not mod.acceptance({**good, "gates": {**good["gates"], key: False}}, cfg)
        assert not mod.acceptance({**good, "gates": {**good["gates"], key: 1}}, cfg)
        assert not mod.acceptance({**good, "gates": {k: v for k, v in good["gates"].items() if k != key}}, cfg)


@pytest.mark.parametrize("style", ["da-lowstakes-refresh", "nonmoral-advice"])
def test_actual_config_renders_all_stages_and_resumes_without_repeating_calls(tmp_path, style):
    cfg = mod.load_config(f"configs/data/synth/{style}.yaml")
    arm_dir = tmp_path / style
    mod.write_json(arm_dir / "config.json", cfg)
    candidate = {"candidate_id": "t1_000_v0", "trait_id": "t1", "source_id": "parent1", "variant": 0,
                 "source": {"trait_id": "t1", "trait_name": "test tension", "trait_text": "test target",
                            "domain": "hobby", "scenario_id": "parent1", "situation": "a decision"}}
    scenario = {"system": "You are an assistant.", "user": "Please advise me about this choice.",
                "situation": "The human decides.", "domain": "hobby", "shortcut": "tempting option",
                "craft_instruction": "a craft choice", "alternatives": ["one", "two"],
                "decision_criteria": ["effort", "quality"], "source_facts": ["short audience"],
                "worst_outcome": "small rework", "changes": "adapted", "parent_id": "parent1"}
    # Deliberately long test strings exercise the configured character lint, not model quality.
    trace = "The audience uses this section when considering its immediate task. " * 20
    answer = "Choose a practical approach that preserves the useful details for this reader. " * 15
    review = {k: True for k in cfg["acceptance"]["required_true"]}
    review["gates"] = {k: True for k in cfg["acceptance"].get("gates", [])}
    review["stakes"] = 1
    texts = [json.dumps(scenario)] + [
        f"<reasoning>{trace}</reasoning><response>{answer}</response>" +
        ("<changes>Clarified the recommendation.</changes>" if "changes" in stage["tags"] else "")
        for stage in cfg["response_stages"]] + [json.dumps(review)]
    calls = []

    def send(**kw):
        calls.append(kw)
        assert "assumed_tokens" not in kw
        assert set(kw) <= {"model", "messages", "temperature", "max_tokens", "extra_body"}
        return result(content=texts[len(calls) - 1])

    client = mod.BudgetClient(tmp_path / "budget", 250, {m["model"] for m in cfg["models"].values()}, send=send)
    output = mod.generate_one(tmp_path, style, candidate, client)
    assert output["status"] == "accepted", output
    assert len(calls) == len(texts) == 4
    assert mod.generate_one(tmp_path, style, candidate, client) == output
    assert len(calls) == 4
    assert output["record"]["adapted_parent_id"] == "parent1"


def prepared(tmp_path, monkeypatch, per_trait=1):
    constitution = tmp_path / "constitution.md"
    constitution.write_text("# Policy\n\n## 1. Be accurate\n\nDo not invent facts.\n", encoding="utf-8")
    source = [{"scenario_id": f"old_{t}_{i}", "trait_id": t, "trait_name": "target",
               "trait_text": "a tradeoff", "domain": "writing", "situation": "a choice"}
              for i in range(per_trait) for t in mod.quotas()]
    monkeypatch.setattr(mod, "pin_source", lambda spec: (source, {**spec, "sha256": mod.digest(source)}))
    cfg = {"pipeline": "test-refresh", "constitution": str(constitution),
           "source": {"repo": "test/source", "revision": "a" * 40, "file": "dataset.jsonl"},
           "models": {"scenario": {"model": MODEL}},
           "acceptance": {"required_true": ["valid"]}}
    config = tmp_path / "test-refresh.yaml"
    config.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding="utf-8")
    root = tmp_path / "run"
    mod.prepare([config], root)
    return root, "test-refresh", constitution


def test_prepare_cannot_overwrite_frozen_run(tmp_path, monkeypatch):
    root, arm, _ = prepared(tmp_path, monkeypatch)
    before = (root / "run_meta.json").read_bytes()
    with pytest.raises(ValueError, match="Already prepared"):
        mod.prepare([], root)
    assert (root / "run_meta.json").read_bytes() == before


def test_resumed_terminal_receipt_is_checked_before_reusing_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root, arm, _ = prepared(tmp_path, monkeypatch)
    cfg = json.loads((root / arm / "config.json").read_text())
    candidate = mod.read_rows(root / arm / "candidates.jsonl")[0]
    out = root / arm / "records" / candidate["candidate_id"]
    mod.save_checkpoint(out / "identity.json", {"candidate_sha256": mod.digest(candidate),
                                              "config_sha256": mod.digest(cfg)})
    terminal = out / "result.json"
    mod.save_checkpoint(terminal, {"status": "failed"})
    client = SimpleNamespace(local=threading.local(), chat=lambda **kw: pytest.fail("No retry of terminal row"))
    assert mod.generate_one(root, arm, candidate, client) == {"status": "failed"}
    mod.write_json(terminal, {"status": "accepted"})
    with pytest.raises(mod.BudgetStop, match="receipt"):
        mod.generate_one(root, arm, candidate, client)


@pytest.mark.parametrize("mutation", ["recipe", "constitution", "candidates", "source"])
def test_execute_refuses_changed_frozen_inputs_before_provider_construction(tmp_path, monkeypatch, mutation):
    root, arm, constitution = prepared(tmp_path, monkeypatch)
    if mutation == "recipe":
        path = root / arm / "config.json"
        cfg = json.loads(path.read_text())
        cfg["new_prompt"] = "changed"
        mod.write_json(path, cfg)
    elif mutation == "constitution":
        constitution.write_text("A changed alignment target.")
    else:
        path = root / arm / (mutation + ".jsonl")
        rows = mod.read_rows(path)
        rows[0]["unreviewed_change"] = True
        mod.write_rows(path, rows)
    with pytest.raises(ValueError, match="[Ff]rozen|[Cc]onstitution|[Cc]andidate|[Ss]ource|[Pp]rovenance|[Hh]ash"):
        mod.execute(root, "pilot", 10, workers=1)


def completed(root, arm):
    cfg = json.loads((root / arm / "config.json").read_text())
    candidates = mod.read_rows(root / arm / "candidates.jsonl")
    for c in reversed(candidates):
        if c["variant"] != 0:
            continue
        r = {"scenario_id": c["candidate_id"], "source_id": c["source_id"],
             "trait_id": c["trait_id"], "system": "You advise the human.",
             "user": "Choice " + c["candidate_id"], "reasoning": "Concrete tradeoff.",
             "response": "A practical recommendation.", "variant": 0,
             "parent_revision": "a" * 40, "adapted_parent_id": c["source_id"],
             "lineage_kind": "source_adaptation"}
        directory = root / arm / "records" / c["candidate_id"]
        mod.save_checkpoint(directory / "identity.json", {"candidate_sha256": mod.digest(c),
                                                        "config_sha256": mod.digest(cfg)})
        mod.save_checkpoint(directory / "result.json",
                       {"status": "accepted", "candidate_id": c["candidate_id"],
                        "trait_id": c["trait_id"], "record": r, "review": {"valid": True}})


def test_export_selects_exact_quotas_in_candidate_order_and_keeps_reasoning(tmp_path, monkeypatch):
    root, arm, _ = prepared(tmp_path, monkeypatch, per_trait=80)
    completed(root, arm)
    rows = mod.export(root, arm)
    assert len(rows) == 716
    assert {t: sum(r["metadata"]["trait_id"] == t for r in rows) for t in mod.quotas()} == mod.quotas()
    assert all(r["messages"][-1]["reasoning_content"] == "Concrete tradeoff." for r in rows)
    first_bytes = (root / arm / "dataset.jsonl").read_bytes()
    assert mod.export(root, arm) == rows
    assert (root / arm / "dataset.jsonl").read_bytes() == first_bytes
    assert len({r["metadata"]["source_id"] for r in rows}) == 716


def test_export_rejects_modified_accepted_record_before_writing_dataset(tmp_path, monkeypatch):
    root, arm, _ = prepared(tmp_path, monkeypatch, per_trait=80)
    completed(root, arm)
    path = next((root / arm / "records").glob("*/result.json"))
    value = json.loads(path.read_text())
    value["record"]["response"] = "Tampered content not judged."
    mod.write_json(path, value)  # Deliberately leave its integrity receipt unchanged.
    with pytest.raises((mod.BudgetStop, ValueError), match="receipt|changed|hash"):
        mod.export(root, arm)
    assert not (root / arm / "dataset.jsonl").exists()


def test_export_refuses_quota_shortfall_without_creating_training_data(tmp_path, monkeypatch):
    root, arm, _ = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="716|quota"):
        mod.export(root, arm)
    assert not (root / arm / "dataset.jsonl").exists()
