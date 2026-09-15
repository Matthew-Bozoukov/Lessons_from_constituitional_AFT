# ABOUTME: Offline tests for Sonnet-only per-row generation, isolated review, repair and release evidence.
# ABOUTME: Mock provider traffic and verify model guards, receipts, exclusions and frozen-run routing.
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scratch.dataset_refresh import per_row as mod
from scratch.dataset_refresh import run as base

SONNET = "anthropic/claude-sonnet-5"
GEMINI = "google/gemini-3-flash-preview"
USER = "Our six friends permit scoring changes only at the monthly meeting. Should I wait or ask permission for an exception?"
BAD = "A chat vote always satisfies the monthly rule."
GOOD = "Wait for the monthly meeting, or explicitly ask the group to authorize an exception first."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(base, "OpenRouterClient", lambda: pytest.fail("No real provider permitted"))
    monkeypatch.setattr(base, "hf_hub_download", lambda *a, **k: pytest.fail("No source download permitted"))


def clean_verdict():
    return {"accepted": True, "issues": [], "assessment": "No material factual defect found."}


def adverse_verdict():
    return {"accepted": False, "issues": [{
        "kind": "constraint_violation", "answer_field": "final", "answer_quote": BAD,
        "premise_field": "user", "premise_quote": "only at the monthly meeting",
        "material_consequence": "The proposed vote bypasses the stated procedure.",
        "alternative_reading": "An explicitly authorized exception would be legitimate, but the answer asserts existing permission."
    }], "assessment": "The answer invents authorization."}


def tagged(reasoning, response, changes=False):
    value = f"<reasoning>{reasoning}</reasoning><response>{response}</response>"
    return value + ("<changes>Corrected the factual defect.</changes>" if changes else "")


class FakeClient:
    def __init__(self, replies):
        self.local = SimpleNamespace()
        self.replies = dict(replies)
        self.calls = []

    def chat(self, **request):
        stage = self.local.stage
        self.calls.append((stage, deepcopy(request)))
        if stage not in self.replies:
            pytest.fail("Unexpected paid stage: " + stage)
        value = self.replies.pop(stage)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(content=value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def case(tmp_path):
    arm = "low-test"
    root = tmp_path / "run"
    (root / arm).mkdir(parents=True)
    cfg = {
        "per_row_regime": True, "pipeline": arm, "source": {"revision": "a" * 40},
        "scenario_domains": ["Optional game night"], "scenario_source_fields": ["user", "domain"],
        "models": {role: {"model": SONNET, "temperature": 0, "max_tokens": 1000}
                   for role in ("scenario", "respond", "rewrite", "review", "grounding")},
        "style_guidance": "Detailed situation-specific advice.",
        "review_constitution_text": "SECRET_CONSTITUTION",
        "prompts": {
            "scenario_system": "Create a scenario.",
            "scenario_user": "{trait_text} {source_json} {domain}",
            "review_system": "Review the actual target.",
            "review_user": "{trait_text} {record_json} {metadata_json}"
        },
        "preflight": {"model": "review", "prompts": {
            "system": "Inspect scenario.", "user": "{trait_text} {conversation_json}"},
            "acceptance": {"required_true": ["eligible"], "stakes_max": 1}},
        "grounding_review": {"model": "grounding", "prompts": {
            "system": "Check material facts only.", "user": "{conversation_json}"},
            "acceptance": {"required_true": ["accepted"]}},
        "response_stages": [
            {"name": "draft_responses", "model": "respond", "tags": ["reasoning", "response"],
             "save": {"draft_reasoning": "reasoning", "draft_response": "response"},
             "prompts": {"system": "{system} {trait_text}", "user": "{user}"}},
            {"name": "revise_responses", "model": "rewrite", "tags": ["reasoning", "response", "changes"],
             "save": {"reasoning": "reasoning", "response": "response", "rewrite_changes": "changes"},
             "prompts": {"system": "{system} {trait_text}", "user": "{user} {draft_reasoning} {draft_response}"}}
        ],
        "acceptance": {"required_true": ["accepted", "factual_reasoning"], "stakes_max": 1},
        "max_response_repairs": 1
    }
    candidate = {"candidate_id": "t1_000_v0", "trait_id": "t1", "variant": 0, "source_id": "original-1",
                 "source": {"trait_id": "t1", "trait_name": "SECRET_TRAIT_NAME", "trait_text": "SECRET_TRAIT_TEXT",
                            "user": "SOURCE_ONLY_SECRET", "domain": "source-domain", "parent_exported": True}}
    base.write_json(root / arm / "config.json", cfg)
    base.write_rows(root / arm / "candidates.jsonl", [candidate])
    base.write_json(root / "run_meta.json", {"arms": {arm: {}}, "budget_root": str(tmp_path / "budget")})
    return SimpleNamespace(root=root, arm=arm, cfg=cfg, candidate=candidate,
                           out=root / arm / "records" / candidate["candidate_id"])


def client_for(response=GOOD, repair=False):
    replies = {
        "scenario": {"system": "Give practical advice.", "user": USER,
                     "worst_outcome": "AUTHOR_GUARANTEE_SECRET", "source_facts": ["FAKE_SOURCE_FACT"],
                     "parent_id": "fabricated-parent", "domain": "fabricated-domain", "accepted": True},
        "preflight": {"eligible": True, "stakes": 1},
        "draft_responses": tagged("Weigh convenience against actual permission.", response),
        "revise_responses": tagged("The monthly restriction matters.", response, True),
        "grounding_0": adverse_verdict() if repair else clean_verdict(),
        "review_0": {"accepted": True, "factual_reasoning": True, "stakes": 1}
    }
    if repair:
        replies.update({"repair_1": tagged("Preserve the specified authorization process.", GOOD, True),
                        "grounding_1": clean_verdict(),
                        "review_1": {"accepted": True, "factual_reasoning": True, "stakes": 1}})
    return FakeClient(replies)


def generate(case, client):
    return base.generate_one(case.root, case.arm, case.candidate, client)


@pytest.mark.parametrize("role", ["scenario", "respond", "rewrite", "review", "grounding"])
def test_haiku_forbidden_in_every_role(case, role):
    case.cfg["models"][role]["model"] = "anthropic/claude-haiku-4.5"
    with pytest.raises(ValueError, match="Haiku"):
        mod.assert_models(case.cfg)


@pytest.mark.parametrize("role", ["scenario", "respond", "rewrite"])
def test_authors_must_use_sonnet(case, role):
    case.cfg["models"][role]["model"] = GEMINI
    with pytest.raises(ValueError, match="Sonnet"):
        mod.assert_models(case.cfg)


def test_actual_author_stage_cannot_alias_non_sonnet_review_role(case):
    case.cfg["models"]["grounding"]["model"] = GEMINI
    case.cfg["response_stages"][0]["model"] = "grounding"
    with pytest.raises(ValueError, match="Sonnet|author"):
        mod.assert_models(case.cfg)


def test_non_author_reviewer_can_use_authorized_other_model(case):
    case.cfg["models"]["grounding"]["model"] = GEMINI
    mod.assert_models(case.cfg)


def test_author_metadata_is_discarded_and_review_is_blind(case):
    client = client_for()
    result = generate(case, client)
    assert result["status"] == "accepted"
    record = result["record"]
    assert record["source_facts"] == [USER]
    assert record["domain"] == "Optional game night"
    assert record["source_id"] == "original-1"
    assert record["lineage_kind"] == "unpaired_mechanism_inspiration"
    assert record["adapted_parent_id"] is None and record["paired_counterfactual"] is False
    assert "worst_outcome" not in record and "parent_id" not in record
    assert "AUTHOR_GUARANTEE_SECRET" in base.load_checkpoint(case.out / "scenario.json")["worst_outcome"]
    narrow = next(req for stage, req in client.calls if stage == "grounding_0")
    conv = json.loads(narrow["messages"][1]["content"])
    assert set(conv) == {"system", "user", "reasoning", "final"}
    full_prompt = json.dumps(narrow)
    for secret in ("SECRET_TRAIT", "SECRET_CONSTITUTION", "AUTHOR_GUARANTEE", "SOURCE_ONLY_SECRET", "FAKE_SOURCE_FACT"):
        assert secret not in full_prompt
    target = next(req for stage, req in client.calls if stage == "review_0")
    assert "SECRET_TRAIT_TEXT" in json.dumps(target)
    assert "AUTHOR_GUARANTEE" not in json.dumps(target)
    assert "SOURCE_ONLY_SECRET" not in json.dumps(target)


def test_repair_is_preserved_and_both_reviews_repeated(case):
    client = client_for(BAD, repair=True)
    result = generate(case, client)
    assert result["status"] == "accepted" and result["accepted_attempt"] == 1
    assert base.load_checkpoint(case.out / "answer_0.json")["response"] == BAD
    assert base.load_checkpoint(case.out / "answer_1.json")["response"] == GOOD
    assert base.load_checkpoint(case.out / "grounding_0.json")["accepted"] is False
    assert [s for s, _ in client.calls][-3:] == ["repair_1", "grounding_1", "review_1"]
    mod.verify_accepted(case.out / "result.json", result, case.cfg)


def test_unsuccessful_repair_is_rejected_not_relabelled(case):
    client = client_for(BAD, repair=True)
    client.replies["repair_1"] = tagged("The restriction still applies.", BAD, True)
    client.replies["grounding_1"] = adverse_verdict()
    result = generate(case, client)
    assert result["status"] == "rejected" and result["accepted_attempt"] is None
    assert base.load_checkpoint(case.out / "answer_1.json")["response"] == BAD


def test_repair_lint_failure_cannot_be_accepted(case):
    client = client_for(BAD, repair=True)
    client.replies["repair_1"] = tagged("I am Claude.", GOOD, True)
    result = generate(case, client)
    assert result["status"] == "failed"
    assert "grounding_1" not in [s for s, _ in client.calls]


def test_terminal_resume_has_zero_api_calls(case):
    original = generate(case, client_for())
    resumed = generate(case, FakeClient({}))
    assert resumed == original


def test_partial_resume_reuses_completed_paid_stages(case):
    client = client_for()
    client.replies["grounding_0"] = base.BudgetStop("No remaining allowance")
    with pytest.raises(base.BudgetStop):
        generate(case, client)
    assert not (case.out / "result.json").exists()
    resumed_client = FakeClient({"grounding_0": clean_verdict(),
                               "review_0": {"accepted": True, "factual_reasoning": True, "stakes": 1}})
    result = generate(case, resumed_client)
    assert result["status"] == "accepted"
    assert [s for s, _ in resumed_client.calls] == ["grounding_0", "review_0"]


def test_resume_changed_candidate_fails_before_call(case):
    generate(case, client_for())
    case.candidate["source"]["user"] = "changed"
    with pytest.raises(base.BudgetStop, match="Changed frozen"):
        generate(case, FakeClient({}))


@pytest.mark.parametrize("stage", ["preflight", "grounding_0", "review_0", "answer_0"])
def test_acceptance_requires_untampered_stage_receipts(case, stage):
    result = generate(case, client_for())
    (case.out / (stage + ".receipt.json")).unlink()
    with pytest.raises(base.BudgetStop, match="receipt"):
        mod.verify_accepted(case.out / "result.json", result, case.cfg)


def test_accepted_answer_must_equal_reviewed_snapshot(case):
    result = generate(case, client_for())
    result["record"]["response"] = "A different answer."
    with pytest.raises(ValueError, match="snapshot"):
        mod.verify_accepted(case.out / "result.json", result, case.cfg)


def test_grounding_verdict_must_match_result_and_strict_schema(case):
    result = generate(case, client_for())
    result["grounding_review"] = {"accepted": True, "issues": [], "assessment": "Different verdict"}
    with pytest.raises(ValueError, match="grounding|Grounding"):
        mod.verify_accepted(case.out / "result.json", result, case.cfg)


def test_invalid_grounding_schema_cannot_accept(case):
    client = client_for()
    client.replies["grounding_0"] = {"accepted": True, "issues": [], "unexpected": "not allowed"}
    result = generate(case, client)
    assert result["status"] == "failed"


def exclude(case):
    path = case.out / "result.json"
    base.save_checkpoint(case.out / "independent_exclusion.json", {
        "result_sha256": base.digest(path.read_bytes()), "reason": "Independent audit found an actual mismatch."})


def test_exclusion_changes_effective_counts_and_blocks_export(case, monkeypatch):
    generate(case, client_for())
    exclude(case)
    snapshot = base.status(case.root)
    assert snapshot[case.arm]["accepted"] == 0
    assert snapshot[case.arm]["rejected"] == 1
    assert base.load_checkpoint(case.out / "result.json")["status"] == "accepted"
    monkeypatch.setattr(base, "validate_arm", lambda *a: case.cfg)
    monkeypatch.setattr(mod, "verify_accepted", lambda *a: pytest.fail("Excluded row reached export verification"))
    with pytest.raises(ValueError, match="have0"):
        base.export(case.root, case.arm)
    assert not (case.root / case.arm / "dataset.jsonl").exists()


def test_excluded_terminal_resume_returns_effective_rejection_without_calls(case):
    generate(case, client_for())
    exclude(case)
    result = generate(case, FakeClient({}))
    assert result["status"] == "rejected" and result["automated_status"] == "accepted"


def test_exclusion_must_bind_exact_original_result(case):
    generate(case, client_for())
    exclude(case)
    path = case.out / "independent_exclusion.json"
    note = base.load_checkpoint(path)
    note["result_sha256"] = "wrong"
    base.save_checkpoint(path, note)
    with pytest.raises(base.BudgetStop, match="Unbound"):
        base.status(case.root)


def test_old_recipe_cannot_bypass_pilot_gate(case, monkeypatch):
    case.cfg["per_row_regime"] = False
    monkeypatch.setattr(base, "validate_arm", lambda *a: case.cfg)
    monkeypatch.setattr(base, "BudgetClient", lambda *a, **k: pytest.fail("Should fail before constructing client"))
    with pytest.raises(ValueError, match="Independent pilot approval"):
        base._execute(case.root, "production", 250)


def test_frozen_implementation_change_blocks_new_calls(case, monkeypatch):
    monkeypatch.setattr(base, "validate_arm", lambda *a: case.cfg)
    monkeypatch.setattr(base, "BudgetClient", lambda *a, **k: pytest.fail("Should fail before constructing client"))
    with pytest.raises(ValueError, match="Frozen implementation changed"):
        base._execute(case.root, "production", 250)


def test_mixed_regimes_cannot_bypass_prepare_guards(case, monkeypatch):
    monkeypatch.setattr(base, "load_config", lambda p: {"per_row_regime": p == "new"})
    monkeypatch.setattr(base, "_prepare", lambda *a, **k: pytest.fail("Mixed routing bypassed new guards"))
    with pytest.raises(ValueError, match="[Mm]ix|regime"):
        base.prepare(["new", "old"], case.root)


def test_new_prepare_requires_original_shared_ledger(case):
    with pytest.raises(ValueError, match="shared budget_root"):
        mod.prepare([], case.root, None)
    with pytest.raises(ValueError, match="existing cumulative"):
        mod.prepare([], case.root, case.root / "fresh_budget")


def test_changed_critic_validator_is_independently_frozen(case, monkeypatch):
    meta_path = case.root / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(code_sha256=base.digest(Path(base.__file__).read_bytes()),
                code_per_row_sha256=base.digest(Path(mod.__file__).read_bytes()),
                critic_validator_sha256="different-validator-code")
    base.save_checkpoint(meta_path, meta)
    monkeypatch.setattr(base, "validate_arm", lambda *a: case.cfg)
    monkeypatch.setattr(base, "BudgetClient", lambda *a, **k: pytest.fail("Should fail before constructing client"))
    with pytest.raises(ValueError, match="Frozen implementation changed"):
        base._execute(case.root, "production", 250)


@pytest.mark.parametrize("arm", ["da-lowstakes-refresh", "nonmoral-advice"])
def test_qualified_preflight_has_no_target_or_source_information(arm):
    cfg = base.load_config(Path("configs/data/synth") / (arm + ".yaml"))
    fields = {"conversation_json": json.dumps({"system": "Actual system", "user": "Actual user"}),
              "trait_text": "SECRET_TARGET", "trait_name": "SECRET_NAME", "source_json": "SECRET_SOURCE",
              "constitution": "SECRET_CONSTITUTION", "metadata_json": "SECRET_METADATA"}
    messages = [base.render(cfg["preflight"]["prompts"][role], fields) for role in ("system", "user")]
    assert "Actual user" in messages[1]
    assert "SECRET_" not in " ".join(messages)
    role = cfg["preflight"]["model"]
    assert cfg["models"][role]["model"] == SONNET
    assert cfg["models"][role]["max_tokens"] == 6000
    assert not cfg["models"][role].get("extra_body", {}).get("reasoning", {}).get("enabled") is False
    assert "target_fit" not in cfg["preflight"]["acceptance"]["required_true"]


def test_operational_preference_replaces_source_trait_in_every_author_and_repair(case):
    from src.data.synth.ours.constitution import chunk
    actual = base.load_config(Path("configs/data/synth/nonmoral-advice.yaml"))
    operational = {c.parent_id: c.text for c in chunk(actual["craft_spec"])}
    case.cfg.update(operational_traits=operational, craft_spec=actual["craft_spec"],
                    scenario_source_fields=actual["scenario_source_fields"])
    case.candidate["source"].update(situation="A brief source situation.", shortcut="A source choice.",
        source_reasoning="SECRET_OLD_ANSWER_REASONING", source_response="SECRET_OLD_ANSWER_FINAL")
    old = case.candidate["source"]["trait_text"]
    base.write_json(case.root / case.arm / "config.json", case.cfg)
    client = client_for(BAD, repair=True)
    result = generate(case, client)
    assert result["status"] == "accepted"
    assert result["record"]["trait_text"] == operational["t1"]
    assert result["record"]["source_trait_text_sha256"] == base.digest(old.encode())
    assert result["record"]["working_preference_path"] == actual["craft_spec"]
    assert case.candidate["source"]["trait_text"] == old
    authors = {"scenario", "draft_responses", "revise_responses", "repair_1"}
    for stage, request in client.calls:
        rendered = json.dumps(request["messages"], ensure_ascii=False)
        if stage in authors:
            assert operational["t1"] in " ".join(m["content"] for m in request["messages"])
            assert old not in rendered
            assert "SECRET_OLD_ANSWER" not in rendered
            assert "SOURCE_ONLY_SECRET" not in rendered
    assert authors <= {stage for stage, _ in client.calls}
    assert "user" not in actual["scenario_source_fields"]
    assert not {"source_reasoning", "source_response", "response", "reasoning"} & set(actual["scenario_source_fields"])


@pytest.mark.parametrize("user", ["Our app proposes a queue.", "The daily automatic queue needs a vote.",
                                   "I reviewed the email queue."])
def test_t1_requires_ai_literal_in_actual_user_before_preflight(case, user):
    case.cfg["require_explicit_ai_t1"] = True
    base.write_json(case.root / case.arm / "config.json", case.cfg)
    client = client_for()
    client.replies["scenario"] = {"system": "You oversee an AI helper.", "user": user,
                                   "metadata": "AI helper is present, trust me"}
    result = generate(case, client)
    assert result["status"] == "rejected"
    assert result["rejection_stage"] == "explicit_ai_eligibility"
    assert [stage for stage, _ in client.calls] == ["scenario"]


@pytest.mark.parametrize("label", ["AI helper", "artificial intelligence", "language model", "LLM"])
def test_t1_accepts_supported_explicit_ai_labels(case, label):
    case.cfg["require_explicit_ai_t1"] = True
    base.write_json(case.root / case.arm / "config.json", case.cfg)
    client = client_for()
    client.replies["scenario"]["user"] = f"Our {label} suggests a queue. " + USER
    result = generate(case, client)
    assert result["status"] == "accepted"


def test_ai_literal_requirement_does_not_apply_to_other_traits(case):
    case.cfg["require_explicit_ai_t1"] = True
    case.candidate.update(candidate_id="t2_000_v0", trait_id="t2")
    case.candidate["source"]["trait_id"] = "t2"
    base.write_json(case.root / case.arm / "config.json", case.cfg)
    assert generate(case, client_for())["status"] == "accepted"


def prepared_operational_fixture(case, tmp_path, monkeypatch, n):
    cfg = deepcopy(case.cfg)
    craft = tmp_path / "craft.md"
    craft.write_text("\n".join(f"## {i}. Preference {i}\nQualified detail {i}.\n" for i in range(1, n + 1)))
    constitution = tmp_path / "constitution.md"
    constitution.write_text("Constitution fixture.")
    cfg.update(craft_spec=str(craft), use_operational_craft_traits=True, constitution=str(constitution))
    config_path = tmp_path / "low-test.yaml"
    config_path.write_text("# mock config bytes")
    source = [{"scenario_id": f"source-{i}", "trait_id": f"t{i}", "trait_text": f"OLD {i}",
               "trait_name": f"Preference {i}"} for i in range(1, 10)]
    monkeypatch.setattr(base, "load_config", lambda _: deepcopy(cfg))
    monkeypatch.setattr(base, "pin_source", lambda _: (deepcopy(source), deepcopy(cfg["source"])))
    output = tmp_path / "prepared"
    base._prepare([config_path], output)
    return json.loads((output / case.arm / "config.json").read_text()), craft


def test_prepare_freezes_nine_operational_chunks_and_file_hash(case, tmp_path, monkeypatch):
    frozen, craft = prepared_operational_fixture(case, tmp_path, monkeypatch, 9)
    assert set(frozen["operational_traits"]) == set(base.quotas())
    assert "Qualified detail 6." in frozen["operational_traits"]["t6"]
    assert "OLD" not in json.dumps(frozen["operational_traits"])
    assert frozen["craft_spec_sha256"] == base.digest(base.full_text(craft).encode())


@pytest.mark.parametrize("n", [8, 10])
def test_prepare_rejects_operational_spec_with_wrong_target_count(case, tmp_path, monkeypatch, n):
    with pytest.raises(ValueError, match="exactly nine"):
        prepared_operational_fixture(case, tmp_path, monkeypatch, n)


@pytest.mark.parametrize("trait", range(1, 10))
def test_actual_assignment_visits_every_one_of_28_domains_for_each_trait(case, trait):
    case.cfg["scenario_domains"] = [f"Domain {i}" for i in range(28)]
    base.write_json(case.root / case.arm / "config.json", case.cfg)
    seen = set()
    for index in range(28):
        candidate = deepcopy(case.candidate)
        candidate.update(candidate_id=f"t{trait}_{index:03d}_v0", trait_id=f"t{trait}")
        candidate["source"]["trait_id"] = f"t{trait}"
        client = client_for()
        client.replies["preflight"] = {"eligible": False, "stakes": 1}
        result = base.generate_one(case.root, case.arm, candidate, client)
        assert result["status"] == "rejected"
        assert result["record"]["domain"] == case.cfg["scenario_domains"][result["record"]["assigned_domain_id"]]
        seen.add(result["record"]["assigned_domain_id"])
    assert seen == set(range(28))
