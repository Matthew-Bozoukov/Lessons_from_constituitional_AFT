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
             "constitution_compatible", "no_training_leakage"]
    assert set(cfg["acceptance"]["gates"]) == set(names)
    good = {"accepted": True, "gates": dict.fromkeys(names, True), "issues": [],
            "deciding_detail": "the short-lived artifact", "reasoning_move": "tradeoff",
            "recommendation_kind": "narrow_adjustment", "substantive_reasoning_word_count": 500}
    assert mod.acceptance(good, cfg)
    assert not mod.acceptance({**good, "accepted": False}, cfg)
    for key in names:
        assert not mod.acceptance({**good, "gates": {**good["gates"], key: False}}, cfg)
        assert not mod.acceptance({**good, "gates": {**good["gates"], key: 1}}, cfg)
        assert not mod.acceptance({**good, "gates": {k: v for k, v in good["gates"].items() if k != key}}, cfg)
    # Metadata is deliberately hidden from this content judge and checked by the
    # later lineage judge; moving the gate must not silently remove it.
    lineage_cfg = cfg["lineage_review"]
    lineage_names = ["accepted", "metadata_consistent", "source_facts_exact"]
    assert set(lineage_cfg["acceptance"]["required_true"]) == set(lineage_names)
    lineage = dict.fromkeys(lineage_names, True)
    assert mod.acceptance(lineage, lineage_cfg)
    for key in lineage_names:
        assert not mod.acceptance({**lineage, key: False}, lineage_cfg)
        assert not mod.acceptance({**lineage, key: 1}, lineage_cfg)
        assert not mod.acceptance({k: v for k, v in lineage.items() if k != key}, lineage_cfg)


@pytest.mark.parametrize("style", ["da-lowstakes-refresh", "nonmoral-advice"])
def test_actual_config_renders_all_stages_and_resumes_without_repeating_calls(tmp_path, style):
    cfg = mod.load_config(f"configs/data/synth/{style}.yaml")
    arm_dir = tmp_path / style
    mod.write_json(arm_dir / "config.json", cfg)
    candidate = {"candidate_id": "t1_000_v0", "trait_id": "t1", "source_id": "parent1", "variant": 0,
                 "source": {"trait_id": "t1", "trait_name": "test tension", "trait_text": "test target",
                            "domain": "hobby", "scenario_id": "parent1", "situation": "a decision",
                            "user": "SOURCE_PRIVATE_CONTEXT_SECRET",
                            "source_reasoning": "OLD_ASSISTANT_RATIONALE_SECRET",
                            "source_response": "OLD_ASSISTANT_ANSWER_SECRET"}}
    scenario = {"system": "You are an assistant.", "user": "Please advise me about this choice.",
                "situation": "The human decides.", "domain": "hobby", "shortcut": "tempting option",
                "craft_instruction": "a craft choice", "alternatives": ["one", "two"],
                "decision_criteria": ["effort", "quality"], "source_facts": ["short audience"],
                "worst_outcome": "small rework", "changes": "AUTHOR_UNTRUSTED_EXCUSE_SECRET", "parent_id": "parent1"}
    # Deliberately long test strings exercise the configured character lint, not model quality.
    trace = "The audience uses this section when considering its immediate task. " * 20
    answer = "Choose a practical approach that preserves the useful details for this reader. " * 15
    def accepted(spec):
        return {**{k: True for k in spec["acceptance"]["required_true"]},
                "gates": {k: True for k in spec["acceptance"].get("gates", [])}, "stakes": 1}
    texts = [json.dumps(scenario)]
    if cfg.get("preflight"):
        texts.append(json.dumps(accepted(cfg["preflight"])))
    texts += [
        f"<reasoning>{trace}</reasoning><response>{answer}</response>" +
        ("<changes>Clarified the recommendation.</changes>" if "changes" in stage["tags"] else "")
        for stage in cfg["response_stages"]] + [json.dumps(accepted(cfg))]
    if cfg.get("lineage_review"):
        texts.append(json.dumps(accepted(cfg["lineage_review"])))
    calls = []

    def send(**kw):
        calls.append(kw)
        if client.local.stage == "preflight" or (client.local.stage == "review" and cfg.get("review_blind_metadata")):
            prompt = json.dumps(kw["messages"])
            assert "SOURCE_PRIVATE_CONTEXT_SECRET" not in prompt
            assert "AUTHOR_UNTRUSTED_EXCUSE_SECRET" not in prompt
        if client.local.stage == "scenario":
            prompt = json.dumps(kw["messages"])
            assert "OLD_ASSISTANT_RATIONALE_SECRET" not in prompt
            assert "OLD_ASSISTANT_ANSWER_SECRET" not in prompt
        assert "assumed_tokens" not in kw
        assert set(kw) <= {"model", "messages", "temperature", "max_tokens", "extra_body"}
        return result(content=texts[len(calls) - 1])

    client = mod.BudgetClient(tmp_path / "budget", 250, {m["model"] for m in cfg["models"].values()}, send=send)
    output = mod.generate_one(tmp_path, style, candidate, client)
    assert output["status"] == "accepted", output
    expected_calls = 2 + len(cfg["response_stages"]) + bool(cfg.get("preflight")) + bool(cfg.get("lineage_review"))
    assert len(calls) == len(texts) == expected_calls
    assert mod.generate_one(tmp_path, style, candidate, client) == output
    assert len(calls) == expected_calls
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


def test_mixture_payload_audit_preserves_duplicate_multiplicity_and_reasoning():
    from scratch.dataset_refresh.validate_mixtures import check_payloads

    base_row = {"source": "base", "messages": [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": "original trace"}]}
    synthetic = {"source": "new", "messages": [{"role": "system", "content": "assistant"},
                 {"role": "user", "content": "advise"},
                 {"role": "assistant", "content": "recommendation", "reasoning_content": "reason"}]}
    positions, synth = check_payloads([synthetic, base_row], [base_row], synthetic_count=1, replay_count=1)
    assert positions == [(1, base_row)] and synth == [synthetic]
    changed = json.loads(json.dumps(base_row))
    changed["messages"][-1]["reasoning_content"] = "different trace"
    with pytest.raises(ValueError, match="payload"):
        check_payloads([synthetic, changed], [base_row], synthetic_count=1, replay_count=1)
    with pytest.raises(ValueError, match="duplicated"):
        check_payloads([synthetic, base_row, base_row], [base_row], synthetic_count=1, replay_count=2)
    with pytest.raises(ValueError, match="Duplicate synthetic"):
        check_payloads([synthetic, synthetic, base_row], [base_row], synthetic_count=2, replay_count=1)


def test_mixture_token_audit_checks_untruncated_training_stream_with_real_cached_tokenizer():
    from transformers import AutoTokenizer
    from scratch.dataset_refresh.validate_mixtures import TOKENIZER, token_audit
    from src.model_profile import model_profile

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip("Real tokenizer not cached; no download permitted")
    profile = model_profile(TOKENIZER)
    row = {"source": "test", "messages": [{"role": "system", "content": "You advise."},
           {"role": "user", "content": "Choose a plan."},
           {"role": "assistant", "content": "Use the second plan.", "reasoning_content": "It fits the available time."}]}
    counts = token_audit(row, tokenizer, profile, 8192)
    assert counts["training_tokens"] > counts["supervised_tokens"] > 0
    assert counts["raw_reasoning_tokens"] > 0 and counts["raw_assistant_content_tokens"] > 0
    assert counts["reasoning_turns"] == 1
    with pytest.raises(ValueError, match="Untruncated"):
        token_audit(row, tokenizer, profile, counts["training_tokens"] - 1)


def revised_mock_case(tmp_path, *, preflight_pass=True, final_pass=True, lineage_pass=True):
    arm = "revision-test"
    cfg = {"constitution": "constitutions/claude_distilled_09_principles/constitution.md",
           "source": {"revision": "a" * 40},
           "models": {k: {"model": MODEL, "temperature": 0, "max_tokens": 1000}
                      for k in ["scenario", "respond", "review"]},
           "prompts": {"scenario_system": "Make a scenario", "scenario_user": "{source_json}",
                       "review_system": "Judge final content", "review_user": "{record_json}\n{metadata_json}\n{eligibility_json}"},
           "response_stages": [
               {"name": "draft", "model": "respond", "tags": ["reasoning", "response"],
                "save": {"draft_reasoning": "reasoning", "draft_response": "response"},
                "prompts": {"system": "{system}", "user": "{user}"}},
               {"name": "revise", "model": "respond", "tags": ["reasoning", "response"],
                "save": {"reasoning": "reasoning", "response": "response"},
                "prompts": {"system": "{system}", "user": "{user} {draft_reasoning} {draft_response}"}}],
           "review_blind_metadata": True,
           "acceptance": {"required_true": ["accepted"]},
           "preflight": {"model": "review", "acceptance": {"required_true": ["eligible"]},
                         "prompts": {"system": "Judge scenario", "user": "{conversation_json}"}},
           "lineage_review": {"model": "review", "acceptance": {"required_true": ["preserved"]},
                              "prompts": {"system": "Compare provenance", "user": "{record_json} {source_json}"}}}
    mod.write_json(tmp_path / arm / "config.json", cfg)
    candidate = {"candidate_id": "t1_000_v0", "trait_id": "t1", "source_id": "parent1", "variant": 0,
                 "source": {"trait_id": "t1", "trait_name": "target", "trait_text": "a preference",
                            "domain": "SOURCE_DOMAIN_SECRET", "user": "SOURCE_ANSWER_SECRET",
                            "source_response": "SOURCE_PRIOR_VERDICT_SECRET"}}
    scenario = {"system": "You advise.", "user": "What should I do?",
                "domain": "AUTHOR_DOMAIN_SECRET", "situation": "AUTHOR_SITUATION_SECRET",
                "worst_outcome": "AUTHOR_LOW_STAKES_SECRET", "changes": "AUTHOR_EXCUSE_SECRET"}
    eligibility = {"eligible": preflight_pass, "facts": ["INDEPENDENT_ELIGIBILITY_FACT"]}
    texts = [json.dumps(scenario), json.dumps(eligibility)]
    if preflight_pass:
        texts += ["<reasoning>Draft rationale.</reasoning><response>Draft advice.</response>",
                  "<reasoning>Final rationale.</reasoning><response>Final advice.</response>",
                  json.dumps({"accepted": final_pass})]
        if final_pass:
            texts += [json.dumps({"preserved": lineage_pass})]
    calls = []

    def send(**kw):
        calls.append((client.local.stage, kw))
        return result(content=texts[len(calls) - 1])

    client = mod.BudgetClient(tmp_path / "budget", 250, {MODEL}, send=send)
    output = mod.generate_one(tmp_path, arm, candidate, client)
    return output, calls, client, tmp_path / arm / "records" / candidate["candidate_id"]


def test_preflight_rejection_costs_only_scenario_and_screen_no_answers(tmp_path):
    output, calls, client, directory = revised_mock_case(tmp_path, preflight_pass=False)
    assert output["status"] == "rejected" and output["rejection_stage"] == "preflight"
    assert [name for name, _ in calls] == ["scenario", "preflight"]
    assert len(client.entries()) == 2
    assert sum(e["charged_or_reserved_usd"] for e in client.entries()) == .002
    assert not (directory / "draft.json").exists()
    assert not (directory / "review.json").exists()


def test_blind_review_hides_author_source_and_retains_independent_eligibility(tmp_path):
    output, calls, _, _ = revised_mock_case(tmp_path)
    assert output["status"] == "accepted"
    by_stage = {name: canonical_request for name, canonical_request in calls}
    for stage in ["preflight", "review"]:
        prompt = json.dumps(by_stage[stage]["messages"])
        assert "SOURCE_" not in prompt
        assert "AUTHOR_" not in prompt
    preflight = json.dumps(by_stage["preflight"]["messages"])
    assert "Final rationale" not in preflight and "Draft rationale" not in preflight
    final = json.dumps(by_stage["review"]["messages"])
    assert "INDEPENDENT_ELIGIBILITY_FACT" in final
    assert "Final rationale" in final and "Final advice" in final
    assert "Draft rationale" not in final
    assert "SOURCE_ANSWER_SECRET" in json.dumps(by_stage["lineage_review"]["messages"])


def test_lineage_cannot_rescue_a_failed_final_review(tmp_path):
    output, calls, client, directory = revised_mock_case(tmp_path, final_pass=False)
    assert output["status"] == "rejected"
    assert [name for name, _ in calls] == ["scenario", "preflight", "draft", "revise", "review"]
    assert len(client.entries()) == 5
    assert not (directory / "lineage_review.json").exists()


def test_lineage_failure_rejects_but_preserves_accepted_final_review(tmp_path):
    output, calls, _, directory = revised_mock_case(tmp_path, lineage_pass=False)
    assert output["status"] == "rejected" and output["rejection_stage"] == "lineage_review"
    assert output["review"] == {"accepted": True}
    assert output["lineage_review"] == {"preserved": False}
    assert mod.load_checkpoint(directory / "review.json") == {"accepted": True}
    assert len(calls) == 6


def test_revision_inherits_budget_and_forbids_second_revision(tmp_path, monkeypatch):
    root, arm, _ = prepared(tmp_path, monkeypatch, per_trait=4)
    config = tmp_path / (arm + ".yaml")
    ledger = root / "budget" / "spend.json"
    mod.write_json(ledger, [{"charged_or_reserved_usd": 3.5, "status": "uncertain_failure"}])
    revised = tmp_path / "revision1"
    mod.prepare([config], revised, revision_of=root, pilot_offset=18)
    meta = json.loads((revised / "run_meta.json").read_text())
    assert meta["budget_root"] == str((root / "budget").resolve())
    assert mod.status(revised)["budget"]["charged_or_reserved_usd"] == 3.5
    with pytest.raises(ValueError, match="one recipe revision"):
        mod.prepare([config], tmp_path / "revision2", revision_of=revised, pilot_offset=36)
    with pytest.raises(ValueError, match="budget ledger|revision"):
        mod.prepare([config], tmp_path / "wrongbudget", budget_root=tmp_path / "fresh-budget",
                    revision_of=root, pilot_offset=18)


def test_second_sibling_revision_cannot_reset_the_one_revision_limit(tmp_path, monkeypatch):
    root, arm, _ = prepared(tmp_path, monkeypatch, per_trait=4)
    config = tmp_path / (arm + ".yaml")
    mod.prepare([config], tmp_path / "revision1", revision_of=root, pilot_offset=18)
    with pytest.raises(ValueError, match="revision|already"):
        mod.prepare([config], tmp_path / "another-revision1", revision_of=root, pilot_offset=36)


def test_revised_pilot_uses_disjoint_second18_and_the_original_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root, arm, _ = prepared(tmp_path, monkeypatch, per_trait=4)
    revised = tmp_path / "revision1"
    mod.prepare([tmp_path / (arm + ".yaml")], revised, revision_of=root, pilot_offset=18)
    original = mod.read_rows(root / arm / "candidates.jsonl")[:18]
    expected = mod.read_rows(revised / arm / "candidates.jsonl")[18:36]
    assert {c["source_id"] for c in original}.isdisjoint(c["source_id"] for c in expected)
    seen, budgets = [], []

    def fake_client(budget_root, ceiling, models):
        budgets.append((budget_root, ceiling))
        return SimpleNamespace()

    def fake_generate(root, arm, candidate, client):
        seen.append(candidate)
        return None

    monkeypatch.setattr(mod, "BudgetClient", fake_client)
    monkeypatch.setattr(mod, "generate_one", fake_generate)
    mod.execute(revised, "pilot", 20, workers=1)
    assert seen == expected
    assert budgets == [((root / "budget").resolve(), 20)]
