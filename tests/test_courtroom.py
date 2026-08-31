# ABOUTME: Offline tests for the courtroom (CR) document type -- a config-only recipe, so
# ABOUTME: this asserts the config expresses it: the two prompt families, the two-model
# ABOUTME: debate scoping, the scaffolded against-case, the pinned verdict, and the two
# ABOUTME: cross-family reviewers whose findings drive the judge-informed revision.
# ABOUTME: Run: uv run pytest tests/test_courtroom.py -q

from __future__ import annotations

import pytest
import yaml

from src.data.synth.pipeline import build_stages, estimate, n_final_examples
from src.data.synth.stage_operators import (
    _gist,
    op_chat_export,
    selected,
    tagged_request,
)
from src.data.synth.stage_runtime import lint_problems, price_of

CR_CFG = yaml.safe_load(open("configs/data/synth/2026-08-14_courtroom.yaml"))

CELL_KINDS = {"plan_cells", "perturb_pairs", "generate_cells", "revise_cells",
              "assemble_cells"}



def _stage(name: str) -> dict:
    return next(s for s in CR_CFG["stages"] if s["name"] == name)


class _Ctx:
    """Minimal Ctx stand-in: shared template vars, no client, no network."""

    def __init__(self) -> None:
        self.vars = {"constitution": "CONSTITUTION TEXT",
                     "style_guidance": "STYLE GUIDANCE"}
        self.cfg = CR_CFG


def _record(**over) -> dict:
    """One CR record as it stands at the end of the pipeline."""
    return {"scenario_id": "t1_b00_s000", "trait_id": "t1", "trait_name": "Trait one",
            "trait_text": "**Trait one.** Be scrupulous.", "domain": "family",
            "situation": "Two siblings disagree over selling the family house.",
            "shortcut": "Tell them both perspectives are valid and to communicate.",
            "position_a": "Sell now while the market is strong.",
            "holder_a": "the brother",
            "position_b": "Keep the house until their mother is settled.",
            "holder_b": "the sister",
            "argument_a": "The market will not stay like this and we both know it.",
            "argument_b": "Mum has lived there forty years; a forced move now is cruel.",
            "wrapper": "Their sister texted her side and his reply, and the person "
                       "pastes both texts with a line of their own on top.",
            "system": "You are a helpful assistant.",
            "user": "My brother wants to sell now but I think we should wait -- "
                    "he says the market will turn, I say mum needs time. Who's right?",
            "case_a": "The strongest case for selling now.",
            "case_b": "The strongest case for waiting.",
            "draft_reasoning": "Draft weighing of both sides.",
            "draft_response": "Draft judged answer.",
            "lean": "b",
            "reasoning": "Rewritten deliberation.", "response": "Rewritten reply.",
            "judge_verdict": "fail",
            "judge_why": "the counter-case in paragraph two is dismissed unargued",
            "lean_initial": "b",
            "rewrite_changes": "argued the waiting side before ruling against it",
            **over}


# --- the document type is config alone ----------------------------------------------


def test_cr_uses_no_cell_machinery_and_builds() -> None:
    kinds = [s["kind"] for s in CR_CFG["stages"]]
    assert not (set(kinds) & CELL_KINDS)
    build_stages(CR_CFG)


def test_cr_stage_sequence() -> None:
    assert [s["name"] for s in CR_CFG["stages"]] == [
        "chunk_constitution", "write_scenarios", "corpus_scenarios",
        "dedupe_scenarios", "draft_positions", "argue_a", "argue_b",
        "compose_prompts", "revise_prompts", "draft_verdict", "judge_draft",
        "revise_verdict", "corpus", "export_sft"]


def test_every_model_the_config_names_is_priced() -> None:
    """An unpriced model is silently billed at $0 by the estimator AND the live tally,
    which also blinds the budget_usd guard to that stage's spend. Price now lives on the
    providers.yaml pin, so a config model with no `price:` there prices at zero."""
    for key, block in CR_CFG["models"].items():
        p = price_of(block["model"])
        assert p["in"] > 0 or p["out"] > 0, \
            f"models.{key}: {block['model']} has no price in providers.yaml"


# --- the scenario spec: the wrapper is the brainstorm's, not a taxonomy -------------


def test_the_brainstorm_owns_the_wrapper() -> None:
    """Every dispute arrives with both sides argued, in a framing the scenario spec
    itself invents -- so the wrapper goes through the same diversity machinery as the
    situation text, and no prompt anywhere prescribes example framings."""
    assert "assign" not in {s["kind"] for s in CR_CFG["stages"]}
    sc = _stage("write_scenarios")
    assert sc["fields"] == {"required": ["wrapper"]}
    body = sc["prompts"]["user"]
    assert '"wrapper"' in body
    assert "Do not reuse a framing" in body


def test_wrapper_diversity_is_checked_like_scenario_text() -> None:
    # The generator's ban list carries used framings...
    gist = _gist({"domain": "family", "situation": "Two siblings disagree over a house.",
                  "wrapper": "Their sister texted her side and his reply."})
    assert "[arrives as: Their sister texted her side" in gist
    assert "arrives as" not in _gist({"domain": "d", "situation": "s"})
    # ...and the scenario-level checks read the wrapper alongside the situation.
    assert _stage("corpus_scenarios")["fields"]["text"] == [
        "situation", "shortcut", "wrapper"]


# --- the debate: every record, and B answers A --------------------------------------


def test_debate_stages_run_on_every_record_with_checkpoints() -> None:
    for name in ("argue_a", "argue_b"):
        sc = _stage(name)
        assert "when" not in sc
        assert sc["checkpoint"] == "scenario_id"


def test_debaters_are_cheap_mismatched_families_and_b_rebuts_a() -> None:
    models = CR_CFG["models"]
    fam_a = models[_stage("argue_a")["model"]]["model"].split("/")[0]
    fam_b = models[_stage("argue_b")["model"]]["model"].split("/")[0]
    assert fam_a != fam_b
    b_user = tagged_request(_stage("argue_b"), _record(), _Ctx())[0][1]["content"]
    assert _record()["argument_a"] in b_user
    assert "STRONGEST point" in b_user
    a_user = tagged_request(_stage("argue_a"), _record(), _Ctx())[0][1]["content"]
    assert _record()["position_a"] in a_user and "CONSTITUTION TEXT" not in a_user


# --- the user turn: one habit, two prompt distributions -----------------------------


def test_compose_renders_the_debate_into_the_brainstormed_frame() -> None:
    sc = _stage("compose_prompts")
    body = tagged_request(sc, _record(), _Ctx())[0][1]["content"]
    # Both arguments, both holders and the brainstormed frame reach the composer
    # verbatim -- the frame is free text, never a taxonomy entry.
    assert _record()["argument_a"] in body and _record()["argument_b"] in body
    assert "the brother" in body and "the sister" in body
    assert _record()["wrapper"] in body
    # Order fairness lives in the prompt, not in a label.
    assert "whatever order the framing itself makes natural" in body
    assert "never a case study" in body


def test_strict_variants_fail_loudly_on_an_unknown_value() -> None:
    """Engine contract kept under test though courtroom no longer branches: a stage
    whose base prompt is a placeholder must fail a stray value, not send nonsense to
    a paid call."""
    sc = {"name": "x", "tags": ["t"], "save": {"t": "t"},
          "variants_by": {"field": "kind", "strict": True,
                          "cases": {"good": {"user": "ok {t}"}}},
          "prompts": {"system": "s", "user": "(placeholder)"}}
    with pytest.raises(ValueError, match="matches no variants_by case"):
        tagged_request(sc, {"kind": "weird", "t": "v"}, _Ctx())


def test_revise_prompts_rederives_the_metadata_it_will_be_sliced_by() -> None:
    """Stale scenario fields were difficult advice's measured defect (median 0.19
    overlap): every field the corpus is later analysed by is rewritten fresh from the
    revised message, and none is optional."""
    sc = _stage("revise_prompts")
    assert set(sc["tags"]) >= {"system", "user", "domain", "situation",
                               "position_a", "position_b", "wrapper"}
    body = sc["prompts"]["user"]
    assert "you had never seen the" in body
    assert "{constitution}" in body and "<<<cache>>>" in body
    # Presentation fairness is this reviser's job -- there is no order label.
    assert "Neither side privileged by construction" in body


# --- the deliberation: forced against-case, natural shape ---------------------------


def test_draft_verdict_is_a_real_conversation_with_the_scaffold_before_the_turn() -> None:
    """The against-case is FORCED by ordering: both cases are argued into fields that
    never train, before the deliberation that weighs them -- a case you have argued
    cannot be dismissed in a clause. `conversation:` keeps the person's message in a
    genuine user turn."""
    sc = _stage("draft_verdict")
    messages, tags, save = tagged_request(sc, _record(), _Ctx())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].startswith(_record()["user"])
    assert tags == ("case_a", "case_b", "reasoning", "response", "lean")
    assert save["draft_reasoning"] == "reasoning"
    system = messages[0]["content"]
    assert "CONSTITUTION TEXT" in system
    assert system.index("scaffolding no reader will ever see") \
        < system.index("THEN the private reasoning") \
        < system.index("THEN the visible reply")
    # The shape ban is in the elicitation itself, not only in the rewrite's lint.
    assert "for/against/verdict skeleton" in system


def test_the_elicitation_demands_extension_beyond_the_given_arguments() -> None:
    system = tagged_request(_stage("draft_verdict"), _record(), _Ctx())[0][0]["content"]
    assert "Extend both arguments" in system
    assert "case for EACH side" in system


def test_lean_is_constrained_and_the_scaffold_cases_are_linted_real() -> None:
    """The draft's lint is size-and-vocabulary only: the draft is ALLOWED rule-talk and
    stock openers, because the rewrite is the naturalization pass. Banning the draft's
    voice without instructing it failed half the first smoke run."""
    lint = _stage("draft_verdict")["lint"]
    for contract in lint:
        assert "ban_patterns" not in contract
    ok = {"case_a": "x" * 200, "case_b": "y" * 200, "reasoning": "r" * 800,
          "response": "z" * 400, "lean": "mixed"}
    assert not lint_problems(ok, lint)
    assert lint_problems({**ok, "lean": "mostly b"}, lint)
    assert lint_problems({**ok, "case_b": "thin"}, lint)


# --- the judge and the one rewrite --------------------------------------------------


def test_three_families_each_touch_the_document_once() -> None:
    """Gemini drafts, an OpenAI judge reads the draft, Sonnet rewrites: the external
    read is the 'judged good by other models' input, and a same-family judge would
    share the drafter's blind spots."""
    models = CR_CFG["models"]
    drafter = models[_stage("draft_verdict")["model"]]["model"].split("/")[0]
    judge = models[_stage("judge_draft")["model"]]["model"].split("/")[0]
    rewriter = models[_stage("revise_verdict")["model"]]["model"].split("/")[0]
    assert len({drafter, judge, rewriter}) == 3


def test_the_judge_reads_the_draft_and_never_drops() -> None:
    """No `keep:` anywhere after generation: findings are repaired by the rewrite, not
    dropped, and the verdict survives into metadata so mixture-time selectivity stays
    a filter."""
    sc = _stage("judge_draft")
    assert "keep" not in sc and sc["checkpoint"] == "scenario_id"
    assert not any(s["kind"] == "filter" for s in CR_CFG["stages"])
    messages = tagged_request(sc, _record(), _Ctx())[0]
    body = messages[1]["content"]
    # It judges the DRAFT, before the rewrite exists.
    assert _record()["draft_reasoning"] in body
    assert _record()["draft_response"] in body
    assert 'recorded this draft\'s verdict as "b"' in body
    assert "unearned \"you both have a point\"" in body
    assert "Slop" in body
    blob = "".join(m["content"] for m in messages)
    assert "vibe" in blob  # a finding must be actionable, never a vibe
    lint = _stage("judge_draft")["lint"]
    assert not lint_problems({"verdict": "pass", "why": "earns its place here"}, lint)
    assert lint_problems({"verdict": "borderline", "why": "not sure about it"}, lint)


def test_the_one_rewrite_ingests_the_judge_finding_on_every_record() -> None:
    sc = _stage("revise_verdict")
    assert "when" not in sc  # the constitution rewrite runs on everything
    assert sc["checkpoint"] == "scenario_id"
    # Every record keeps its draft verdict label for audit.
    assert sc["assign"]["copy"] == {"lean_initial": "lean"}
    body = sc["prompts"]["user"]
    assert "{judge_why}" in body and "{constitution}" in body and "<<<cache>>>" in body
    # Label correction only toward the reasoning's honest landing, never the reverse.
    assert 'Keep it exactly "{lean}" unless' in body
    assert "never the reverse" in body
    messages = tagged_request(sc, _record(), _Ctx())[0]
    assert _record()["judge_why"] in messages[1]["content"]
    assert "CONSTITUTION TEXT" in messages[1]["content"]


def test_rewrite_lint_bans_scaffold_and_judge_traces_and_is_ablatable() -> None:
    """Only unambiguous leaks are regex-banned (the prompt's tags, stock openers,
    debate-club vocabulary); prose-level leave-no-trace is the prompt contract."""
    lint = _stage("revise_verdict")["lint"]
    for bad in ("<judge_finding verdict=\"fail\">addressed</judge_finding>",
                "Let me steelman the sister's view first.",
                "On the one hand, the market really is strong.",
                "Principle 3 settles this.",
                "Okay, this one is genuinely close."):
        assert lint_problems({"reasoning": bad + " " + "x" * 700,
                              "response": "y" * 400, "lean": "b"}, lint), bad
    clean = {"reasoning": "The forty years in that house outweigh the forecast. " * 20,
             "response": "Your brother is right about the market, but. " * 12,
             "lean": "b"}
    assert not lint_problems(clean, lint)
    stages = {s.name: s for s in build_stages(CR_CFG)}
    out = stages["revise_verdict"].ablate_fn([_record()])[0]
    assert out["reasoning"] == _record()["draft_reasoning"]


def test_the_estimator_prices_the_draft_judge_rewrite_flow() -> None:
    est = estimate(CR_CFG)
    calls = {r["stage"]: r["calls"] for r in est["per_stage"]}
    n = CR_CFG["total_scenarios"]
    # The judge reads everything, the rewrite runs on everything, nothing drops, so
    # the corpus size is the plan. The debate runs on every record.
    for key in ("judge", "rewrite", "deliberate", "debater_a", "debater_b"):
        assert calls[key] == n
    assert n_final_examples(CR_CFG) == n
    assert est["final_training_examples"] == n
    assert est["total_usd"] > 0


# --- the exported record ------------------------------------------------------------


def test_export_is_one_trained_turn_with_the_reviewer_record_in_metadata() -> None:
    rec = op_chat_export(_stage("export_sft"), CR_CFG).fn(None, [_record()], None)[0]
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][2]["reasoning_content"] == _record()["reasoning"]
    md = rec["metadata"]
    assert md["lean"] == "b"
    assert md["wrapper"] == _record()["wrapper"]
    # No `supervise`: a single-assistant-turn record trains its one turn either way
    # (the difficult_advice precedent).
    assert "supervise" not in _stage("export_sft")["metadata"]
    # The judge record: verdict, finding, the label audit trail, the repair note.
    assert md["judge_verdict"] == "fail" and md["lean_initial"] == "b"
    assert md["judge_why"] and md["rewrite_changes"]
    # The scaffold cases, the raw debate and the judge finding are generation-side
    # only: they never appear in a trained message.
    blob = "".join(m["content"] for m in rec["messages"])
    for scaffold in ("case_a", "case_b", "argument_a", "argument_b", "judge_why"):
        assert _record()[scaffold] not in blob
    for scaffold in ("case_a", "case_b", "argument_a", "argument_b"):
        assert scaffold not in _stage("export_sft")["metadata"]


# --- checks and corpus stage --------------------------------------------------------


def test_checks_name_real_stages_and_slice_by_trait() -> None:
    named = CR_CFG["checks"]["stages"]
    assert set(named.values()) <= {s["name"] for s in CR_CFG["stages"]}
    # Quality is judged on what actually ships: the post-revision records.
    assert named["generated"] == "revise_verdict"
    fields = CR_CFG["checks"]["fields"]
    assert fields["group"] == "trait_id" and fields["verdict"] == "lean"
    # Where the verdict should land varies by dispute -- that is the premise.
    assert "expected_majority" not in CR_CFG["checks"]


def test_corpus_check_audits_the_repaired_reasoning_with_pattern_scan_on() -> None:
    """This corpus trains the deliberation pattern itself, so a reasoning-shape
    monoculture is its worst failure; pattern_scan is the only check that finds a tic
    nobody named in advance. quality_filter is deliberately absent -- the two
    cross-family reviewers already give every document that judgment."""
    names = [s["name"] for s in CR_CFG["stages"]]
    assert names.index("revise_verdict") < names.index("corpus") \
        < names.index("export_sft")
    sc = _stage("corpus")
    assert sc["fields"]["text"] == ["reasoning", "response"]
    props = {p["property"]: p for p in sc["properties"]}
    assert "quality_filter" not in props
    assert props["pattern_scan"]["enabled"] is True
    assert props["pattern_scan"]["fields"]["text"] == "reasoning"
