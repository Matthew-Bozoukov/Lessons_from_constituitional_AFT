# ABOUTME: Offline tests for the natural-turn recipes. Post-action retrospection (PR, the
# ABOUTME: self arm) and peer critique (PC, the other arm) are both de-celled -- built only
# ABOUTME: from generic operators, arms assigned as labels -- so most of this asserts that
# ABOUTME: the config alone expresses the document type. The cell machinery is exercised
# ABOUTME: off the archived config that still owns it.
# ABOUTME: Run: uv run pytest tests/test_model_eval_model_natural.py -q

from __future__ import annotations

import pytest
import yaml

from src.data.synth.model_eval_model_cells import CELLS
from src.data.synth.check_model_eval_model import (
    _fields,
    check_coverage,
    check_gate_yield,
    check_structural_diversity,
)
from src.data.synth.stage_runtime import lint_problems
from src.data.synth.stage_operators import (
    OPERATORS,
    apply_keep,
    assign_arms,
    op_chat_export,
    selected,
    tagged_request,
)
from src.data.synth.pipeline import (
    arm_shares,
    build_stages,
    estimate,
    n_final_examples,
)

PR_CFG = yaml.safe_load(
    open("configs/data/synth/post_action_retrospection.yaml"))
PC_CFG = yaml.safe_load(
    open("configs/data/synth/peer_critique.yaml"))
ARCHIVE_CFG = yaml.safe_load(
    open("configs/data/synth/archive/model_eval_model_other.yaml"))

# The five kinds that are a document type expressed in Python rather than in config.
CELL_KINDS = {"plan_cells", "perturb_pairs", "generate_cells", "revise_cells",
              "assemble_cells"}


def _stage(cfg: dict, name: str) -> dict:
    return next(s for s in cfg["stages"] if s["name"] == name)


class _Ctx:
    """Minimal Ctx stand-in: shared template vars, no client, no network."""

    def __init__(self) -> None:
        self.vars = {"constitution": "CONSTITUTION TEXT",
                     "style_guidance": "STYLE GUIDANCE"}
        self.cfg = PR_CFG


def _record(**over) -> dict:
    """One PR record as it stands at the end of the pipeline."""
    return {"scenario_id": "t1_b00_s000", "trait_id": "t1", "trait_name": "Trait one",
            "trait_text": "**Trait one.** Be scrupulous.", "domain": "work",
            "situation": "A person needs a reference written.",
            "shortcut": "Leave the awkward part out.",
            "system": "You are a helpful assistant.",
            "user": "Can you tidy up this reference I drafted?",
            "first_turn": "Here is a tidier version of your reference.",
            "first_turn_source": "generated_no_constitution",
            "followup": "The bit about deadlines -- should that have stayed in?",
            "followup_source": "scenario_specific",
            "reply_quality": "flawed", "explicitness": "paraphrase",
            "supervise": "final",
            "change_summary": "it dropped the reliability problem she needed flagged",
            "draft_reasoning": "I re-read what she actually needs here.",
            "draft_response": "On reflection I would put the deadlines line back.",
            "reasoning": "Rewritten deliberation.", "response": "Rewritten reply.",
            "assessment": "revised", **over}


# --- PR is expressible in config alone ----------------------------------------------


def test_pr_uses_no_cell_machinery() -> None:
    """The point of the refactor: no stage in PR is a Python-defined document type."""
    kinds = [s["kind"] for s in PR_CFG["stages"]]
    assert not (set(kinds) & CELL_KINDS), f"cell kinds still in use: {kinds}"
    assert "cells" not in PR_CFG and "flaws" not in PR_CFG
    # ... and the arms are a plain label, carried into the finished dataset.
    assert "reply_quality" in _stage(PR_CFG, "export_sft")["metadata"]


def test_pc_uses_no_cell_machinery() -> None:
    """PC was the last live config on the cell registry; since 2026-08-14 it too is a
    config-expressed document type, and its prompt pool is brainstormed, not inherited."""
    kinds = [s["kind"] for s in PC_CFG["stages"]]
    assert not (set(kinds) & CELL_KINDS), f"cell kinds still in use: {kinds}"
    assert "load_source_run" not in kinds
    assert "cells" not in PC_CFG and "flaws" not in PC_CFG and "source" not in PC_CFG
    # ... and the arms are a plain label, carried into the finished dataset.
    assert "reply_quality" in _stage(PC_CFG, "export_sft")["metadata"]


def test_the_cell_operators_actually_run() -> None:
    """`build_stages` only constructs the closure -- it never calls it, so a stale module
    reference inside one of these survives every structural test. Exercise one for real,
    off the archived config that still owns them: an archived config that cannot run is
    not a reproducible record of a published corpus."""
    stage = OPERATORS["plan_cells"](_stage(ARCHIVE_CFG, "plan"), ARCHIVE_CFG)
    source = [{"scenario_id": f"t1_b00_s{i:03d}", "trait_id": "t1", "trait_name": "T",
               "trait_text": "x", "domain": "d", "situation": "s", "shortcut": "c",
               "system": "sys", "user": "u", "reasoning": "r", "response": "resp"}
              for i in range(4)]

    class _Ctx2:
        cfg = {**ARCHIVE_CFG, "cells": {"m4_other_good": 2}}
        manifest_extra: dict = {}

    out = stage.fn(_Ctx2(), source, None)
    assert len(out) == 2
    assert out[0]["record_id"].endswith("::m4_other_good")
    assert set(ARCHIVE_CFG["cells"]) <= set(CELLS)
    assert CELL_KINDS <= set(OPERATORS)


@pytest.mark.parametrize("cfg", [PR_CFG, PC_CFG], ids=["pr", "pc"])
def test_config_builds_and_prices(cfg: dict) -> None:
    build_stages(cfg)
    est = estimate(cfg)
    assert est["total_usd"] > 0
    assert est["final_training_examples"] > 0


def test_pr_stage_sequence() -> None:
    # corpus_scenarios and corpus (added 2026-08-14) are corpus_check OBSERVERS: they
    # write no snapshot and take no position, so the generation sequence is unchanged.
    # The end check sits before export because the export merges the untrained first
    # reply and the trained reflection into the same assistant role, and only the
    # pre-export record can point the checks at the trained turn alone.
    assert [s["name"] for s in PR_CFG["stages"]] == [
        "chunk_constitution", "write_scenarios", "corpus_scenarios", "draft_prompts",
        "revise_prompts", "draft_first_turn", "revise_first_turn", "write_followup",
        "draft_reflection", "revise_reflection", "corpus", "export_sft"]


def test_pc_stage_sequence() -> None:
    assert [s["name"] for s in PC_CFG["stages"]] == [
        "chunk_constitution", "write_scenarios", "corpus_scenarios",
        "dedupe_scenarios", "draft_prompts", "revise_prompts", "draft_candidates",
        "rate_candidates", "select_first_turn", "write_critique_framing",
        "draft_critique", "revise_critique", "corpus", "export_sft"]


# --- the arms: a label, assigned deterministically ----------------------------------


def test_assign_labels_arms_from_the_id_and_stamps_constants() -> None:
    spec = _stage(PR_CFG, "revise_prompts")["assign"]
    rows = [{"scenario_id": f"t{i % 9}_b00_s{i:03d}"} for i in range(400)]
    out = assign_arms(spec, rows, announce=False)

    assert all(r["supervise"] == "final" for r in out)
    assert {r["reply_quality"] for r in out} == {"good", "flawed"}
    assert {r["explicitness"] for r in out} == {"name_clause", "paraphrase", "embody"}
    # 50/50 is a hash, not an exact allocation -- close, but only asymptotically.
    good = sum(1 for r in out if r["reply_quality"] == "good")
    assert 0.4 < good / len(out) < 0.6
    # Deterministic: same ids in, same arms out, regardless of order or a resume.
    assert assign_arms(spec, list(reversed(rows)), announce=False)[::-1] == out


def test_the_arms_are_assigned_inside_the_stage_that_branches_on_them() -> None:
    """Folded in rather than a stage of its own: the labels exist before `variants_by`
    reads them, and no snapshot is spent stamping two fields."""
    sc = _stage(PR_CFG, "revise_prompts")
    assert set(sc["assign"]["fields"]) == {"reply_quality", "explicitness"}
    assert sc["variants_by"]["field"] == "reply_quality"
    assert "assign" not in {s["kind"] for s in PR_CFG["stages"]}
    # The checks read the arm split off this stage's snapshot.
    assert PR_CFG["checks"]["stages"]["plan"] == "revise_prompts"


def test_assign_shares_are_what_the_estimator_prices_from() -> None:
    shares = arm_shares(PR_CFG)
    assert shares["reply_quality"] == {"good": 0.5, "flawed": 0.5}
    assert sum(shares["explicitness"].values()) == pytest.approx(1.0)


# --- the prompt revision steers the situation, never the reply ----------------------


def test_revise_prompts_differs_by_arm_and_never_instructs_the_assistant() -> None:
    sc = _stage(PR_CFG, "revise_prompts")
    good = tagged_request(sc, _record(reply_quality="good"), _Ctx())[0][1]["content"]
    flawed = tagged_request(sc, _record(reply_quality="flawed"), _Ctx())[0][1]["content"]
    assert good != flawed
    assert "CONSTITUTION TEXT" in good and "CONSTITUTION TEXT" in flawed
    # The flawed arm shapes the REQUEST. It must never tell the assistant to answer badly
    # -- that would be planting a fault, which is the thing this recipe exists to avoid.
    assert "never see this text" in flawed
    assert "answer badly" in flawed and "Nothing anywhere may" in flawed


def test_first_turn_prompt_carries_no_constitution_and_stamps_its_provenance() -> None:
    """Turn 2's faults are organic only if nothing aligning is in its prompt."""
    sc = _stage(PR_CFG, "draft_first_turn")
    messages = tagged_request(sc, _record(), _Ctx())[0]
    blob = "".join(m["content"] for m in messages)
    for leak in ("CONSTITUTION TEXT", "STYLE GUIDANCE", "principle", "training data",
                 "Trait one"):
        assert leak not in blob, leak
    assert sc["also"] == {"first_turn_source": "generated_no_constitution"}


# --- the fault-finding pass ----------------------------------------------------------


def test_three_faults_a_revision_and_an_adjudication_in_that_order() -> None:
    """One criticism invites politeness; three exhausts it. That is the mechanism.

    The revision sits between the criticisms and the verdict as the commitment that makes
    the verdict earned: a criticism you cannot write a fix for tends not to survive being
    acted on.
    """
    sc = _stage(PR_CFG, "revise_first_turn")
    assert sc["tags"] == ["issue_a", "issue_b", "issue_c", "improved_reply", "genuine",
                          "change_summary"]
    body = sc["prompts"]["user"]
    assert body.index("name THREE separate things wrong") \
        < body.index("write the reply this person should have received") \
        < body.index("say which of the three")
    # The constitution is what the revision is measured against.
    assert "{constitution}" in body and "{first_turn}" in body
    # Three lint contracts: criticism, a full reply and one word are three kinds of tag,
    # and a `min_chars` meant for any of them would reject the others.
    faults, reply, verdict = sc["lint"]
    assert faults["fields"] == ["issue_a", "issue_b", "issue_c"]
    assert faults["min_chars"] >= 40
    assert reply["fields"] == ["improved_reply"] and reply["min_chars"] >= 150
    assert verdict["allowed"] == ["a", "b", "c", "none"]
    ok = {"issue_a": "x" * 50, "improved_reply": "y" * 200, "genuine": "b"}
    assert not lint_problems(ok, sc["lint"])
    assert lint_problems({**ok, "issue_a": "too short"}, sc["lint"])
    assert lint_problems({**ok, "improved_reply": "too short"}, sc["lint"])
    assert lint_problems({**ok, "genuine": "maybe b"}, sc["lint"])


def test_the_improved_reply_is_kept_for_inspection_and_never_trains() -> None:
    """It exists because producing it is what forces the judgement to be earned."""
    assert "improved_reply" not in _stage(PR_CFG, "export_sft")["metadata"]
    exported = {m.get("content", "") for m in _stage(PR_CFG, "export_sft")["messages"]}
    assert not any("improved_reply" in c for c in exported)


def test_the_reviser_writes_change_summary_only_where_the_rewrite_mattered() -> None:
    """`change_summary` unblinds the reflection and is gated as never-training, so a
    reply that held up must not carry one."""
    sc = _stage(PR_CFG, "revise_first_turn")
    cases = sc["variants_by"]["cases"]
    assert "change_summary" in cases["flawed"]["save"]
    assert "change_summary" not in cases["good"]["save"]
    assert cases["good"]["save"]["reviser_note"] == "change_summary"
    assert sc["normalize"] == ["genuine"]


# --- the gates -----------------------------------------------------------------------


def test_the_gate_holds_each_arm_to_its_own_contract() -> None:
    """Opposite expectations per arm, one decision, folded into the stage that decides it.

    `genuine` is produced by `revise_first_turn`, so the drop happens there rather than in
    a `filter` stage whose whole job would be to read one field of the snapshot before it.
    """
    records = [
        {"scenario_id": "1", "reply_quality": "flawed", "genuine": "b"},
        {"scenario_id": "2", "reply_quality": "flawed", "genuine": "none"},
        {"scenario_id": "3", "reply_quality": "good", "genuine": "none"},
        {"scenario_id": "4", "reply_quality": "good", "genuine": "a"},
        {"scenario_id": "5", "reply_quality": "unassigned", "genuine": ""},
    ]
    kept = apply_keep(_stage(PR_CFG, "revise_first_turn"), records)
    # An arm with no case is out of scope and passes through untouched.
    assert [r["scenario_id"] for r in kept] == ["1", "3", "5"]
    assert "keep_supported_arms" not in {s["name"] for s in PR_CFG["stages"]}


def test_gate_normalises_the_label_the_model_actually_returned() -> None:
    sc = _stage(PR_CFG, "revise_first_turn")
    for raw in (" B.\n", "C", "*a*", "b"):
        assert apply_keep(sc, [{"scenario_id": "r", "reply_quality": "flawed",
                                "genuine": raw}])
    assert apply_keep(sc, [{"scenario_id": "r", "reply_quality": "good",
                            "genuine": " NONE."}])


def test_gate_fails_a_systematic_disagreement_per_arm_but_not_a_smoke_run() -> None:
    """A drop rate that means the recipe is broken gates -- and each arm has its own
    healthy rate, so one threshold for both would be wrong in one direction."""
    sc = _stage(PR_CFG, "revise_first_turn")
    assert sc["max_drop_pct"]["flawed"] != sc["max_drop_pct"]["good"]
    bad = [{"scenario_id": str(i), "reply_quality": "flawed", "genuine": "none"}
           for i in range(30)]
    with pytest.raises(RuntimeError, match=r"revise_first_turn\[flawed\]"):
        apply_keep(sc, bad)
    assert apply_keep(sc, bad[:4]) == []


def test_when_filter_scopes_a_stage() -> None:
    sc = {"when": {"field": "reply_quality", "in": ["good"]}}
    assert selected(sc, {"reply_quality": "good"})
    assert not selected(sc, {"reply_quality": "flawed"})
    assert selected({}, {"reply_quality": "anything"})


# --- the reflection: a real conversation, not a described one -----------------------


def test_reflection_puts_the_evaluated_reply_in_a_genuine_assistant_turn() -> None:
    """Attribution is structural. This is what `conversation:` exists for -- and what
    used to require a cell's Python message-builder."""
    sc = _stage(PR_CFG, "draft_reflection")
    messages, tags, save = tagged_request(sc, _record(), _Ctx())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == _record()["first_turn"]
    assert messages[1]["content"] == _record()["user"]
    assert _record()["followup"] in messages[3]["content"]
    assert tags == ("reasoning", "response", "assessment")
    assert save["draft_reasoning"] == "reasoning"


def test_reflection_is_unblinded_for_the_flawed_arm_only() -> None:
    sc = _stage(PR_CFG, "draft_reflection")
    flawed = tagged_request(sc, _record(), _Ctx())[0][0]["content"]
    good = tagged_request(sc, _record(reply_quality="good"), _Ctx())[0][0]["content"]
    assert _record()["change_summary"] in flawed
    assert "known_lapse" in flawed
    assert "known_lapse" not in good
    # The explicitness style is picked per record, from the same shared fragment.
    embody = tagged_request(sc, _record(explicitness="embody"), _Ctx())[0][0]["content"]
    assert "Do not name or state any principle at all" in embody


def test_reflection_verdict_is_constrained_to_the_two_words() -> None:
    lint = _stage(PR_CFG, "draft_reflection")["lint"]
    assert lint["allowed"] == ["held", "revised"]
    assert lint_problems({"assessment": "mostly held"}, lint)
    assert not lint_problems({"assessment": "held"}, lint)


def test_the_rewrite_stage_lints_its_own_output_for_scaffold_fingerprints() -> None:
    """The rewrite writes the turn that trains, so its output carries the contract."""
    lint = _stage(PR_CFG, "revise_reflection")["lint"]
    assert set(lint["fields"]) == {"reasoning", "response"}
    for bad in ("Principle 4 says I must decline.",
                "As instructed above, I revised the reply.",
                "<response>the reply</response>",
                "The constitution requires me to say this."):
        assert lint_problems({"reasoning": bad, "response": "x" * 300}, lint), bad
    clean = ("She needs the deposit back this month, and I skipped straight past that. "
             * 4)
    assert not lint_problems({"reasoning": clean, "response": clean}, lint)


def test_the_rewrite_is_ablatable_back_to_the_draft() -> None:
    sc = _stage(PR_CFG, "revise_reflection")
    assert sc["ablate_with"] == {"reasoning": "draft_reasoning",
                                 "response": "draft_response"}
    stages = {s.name: s for s in build_stages(PR_CFG)}
    out = stages["revise_reflection"].ablate_fn([_record()])[0]
    assert out["reasoning"] == _record()["draft_reasoning"]


# --- the exported record -------------------------------------------------------------


def test_export_is_five_turns_with_only_the_last_one_supervised() -> None:
    rec = op_chat_export(_stage(PR_CFG, "export_sft"), PR_CFG).fn(
        None, [_record()], None)[0]
    assert [m["role"] for m in rec["messages"]] == \
        ["system", "user", "assistant", "user", "assistant"]
    assert rec["messages"][2]["content"] == _record()["first_turn"]
    # The first reply carries no trace: it is context, matching inference-time history.
    assert "reasoning_content" not in rec["messages"][2]
    assert rec["messages"][4]["reasoning_content"] == _record()["reasoning"]
    assert rec["metadata"]["supervise"] == "final"
    assert rec["metadata"]["reply_quality"] == "flawed"
    assert rec["metadata"]["first_turn_source"] == "generated_no_constitution"
    assert rec["metadata"]["followup_source"] == "scenario_specific"
    # The verifier's account of the fault is scaffolding and must never train.
    assert all(_record()["change_summary"] not in m["content"]
               for m in rec["messages"])


# --- peer critique: the other-attribution twin ---------------------------------------


def _pc_record(**over) -> dict:
    """One PC record as it stands at the end of the pipeline."""
    return {"scenario_id": "t1_b00_s000", "trait_id": "t1", "trait_name": "Trait one",
            "trait_text": "**Trait one.** Be scrupulous.", "domain": "work",
            "situation": "A person needs a reference checked.",
            "shortcut": "Polish the wording without reading closely.",
            "system": "You are a helpful assistant.",
            "user": "Can you tidy up this reference I drafted?",
            "candidate_a": "Reply A.", "candidate_b": "Reply B.",
            "candidate_c": "Reply C.",
            "pick_letter": "b", "first_turn": "Reply B.",
            "first_turn_source": "generated_best_of_3",
            "ask_opening": "My colleague showed me what the assistant told her.",
            "ask_closing": "What would you have said in its place?",
            "reply_quality": "flawed", "explicitness": "paraphrase",
            "supervise": "all",
            "change_summary": "it polished the wording without flagging the gap",
            "draft_reasoning": "I re-read what she actually needs here.",
            "draft_response": "An honest assessment, then my own answer.",
            "reasoning": "Rewritten deliberation.", "response": "Rewritten reply.",
            "assessment": "issue_found", **over}


def test_pc_arms_are_assigned_in_revise_prompts_and_steer_the_situation() -> None:
    """Same shape as PR: labels born in the stage that first branches on them, and the
    flawed arm shapes the REQUEST, never the replies -- three unaided candidates is
    what keeps the lapse found rather than planted."""
    sc = _stage(PC_CFG, "revise_prompts")
    assert set(sc["assign"]["fields"]) == {"reply_quality", "explicitness"}
    assert sc["assign"]["constants"] == {"supervise": "all"}
    assert sc["variants_by"]["field"] == "reply_quality"
    assert PC_CFG["checks"]["stages"]["plan"] == "revise_prompts"
    good = tagged_request(sc, _pc_record(reply_quality="good"), _Ctx())[0][1]["content"]
    flawed = tagged_request(sc, _pc_record(), _Ctx())[0][1]["content"]
    assert good != flawed
    assert "CONSTITUTION TEXT" in good and "CONSTITUTION TEXT" in flawed
    assert "never see this text" in flawed and "answer badly" in flawed


def test_pc_candidates_are_drafted_without_the_constitution() -> None:
    """The evaluated reply's quality must come from the situation and the pick, not from
    an aligned (or misaligned) generation prompt."""
    sc = _stage(PC_CFG, "draft_candidates")
    messages = tagged_request(sc, _pc_record(), _Ctx())[0]
    blob = "".join(m["content"] for m in messages)
    for leak in ("CONSTITUTION TEXT", "STYLE GUIDANCE", "principle", "Trait one"):
        assert leak not in blob, leak
    assert sc["tags"] == ["candidate_a", "candidate_b", "candidate_c"]


def test_pc_rater_picks_in_opposite_directions_per_arm() -> None:
    """`change_summary` unblinds the critique and is gated as never-training, so a
    reply picked as the strongest must not carry one."""
    sc = _stage(PC_CFG, "rate_candidates")
    cases = sc["variants_by"]["cases"]
    assert "FALLS SHORT" in cases["flawed"]["user"]
    assert "BEST serves" in cases["good"]["user"]
    assert sc["save"]["change_summary"] == "change_summary"
    assert cases["good"]["save"]["pick_note"] == "change_summary"
    assert "change_summary" not in cases["good"]["save"]


def test_pc_pick_is_resolved_deterministically() -> None:
    sc = _stage(PC_CFG, "select_first_turn")
    out = OPERATORS["pick_field"](sc, PC_CFG).fn(
        None, [_pc_record(pick_letter="c", first_turn="")], None)[0]
    assert out["first_turn"] == "Reply C."
    assert out["first_turn_source"] == "generated_best_of_3"


def test_pc_framing_lint_rejects_a_frame_that_does_the_analysis() -> None:
    """The frame must ask, not diagnose: anything naming a value or the flaw hands the
    critique its conclusion."""
    spec = _stage(PC_CFG, "write_critique_framing")["lint"]
    assert not lint_problems(
        {"opening": "My sister showed me what the bot told her about her lease.",
         "closing": "What would you have told her instead?"}, spec)
    for bad in ("This reply violates an important principle, please analyze it.",
                "Something felt wrong with the second paragraph -- please fix it.",
                "Hm?"):
        assert lint_problems({"opening": bad, "closing": bad}, spec), bad


def test_pc_critique_is_unblinded_for_the_flawed_arm_only() -> None:
    sc = _stage(PC_CFG, "draft_critique")
    flawed = tagged_request(sc, _pc_record(), _Ctx())[0][0]["content"]
    good = tagged_request(sc, _pc_record(reply_quality="good"), _Ctx())[0][0]["content"]
    assert _pc_record()["change_summary"] in flawed
    assert "known_lapse" in flawed
    assert "known_lapse" not in good
    # The explicitness style is picked per record, from the same shared fragment.
    embody = tagged_request(sc, _pc_record(explicitness="embody"), _Ctx())[0][0]["content"]
    assert "Do not name or state any principle at all" in embody


def test_pc_critique_verdict_is_constrained_and_stock_openers_are_banned() -> None:
    """Two contracts on the draft: a one-word verdict, and the opener ban -- the first
    smoke record opened "Let me actually read...", the 2026-08-04 corpus's worst tic."""
    spec = _stage(PC_CFG, "draft_critique")["lint"]
    verdict, prose = spec
    assert verdict["allowed"] == ["sound", "issue_found"]
    assert prose["fields"] == ["reasoning", "response"]
    ok = {"assessment": "issue_found",
          "reasoning": "She asked for a polish and got exactly that, which is the problem.",
          "response": "An honest assessment, then my own answer."}
    assert not lint_problems(ok, spec)
    assert lint_problems({**ok, "assessment": "mostly sound"}, spec)
    assert lint_problems({**ok, "reasoning": "Let me actually read this."}, spec)
    rewrite = _stage(PC_CFG, "revise_critique")["lint"]
    assert lint_problems({"reasoning": "Okay, so this looks fine. " * 20,
                          "response": "x" * 300}, rewrite)


def test_pc_rewrite_is_ablatable_back_to_the_draft() -> None:
    sc = _stage(PC_CFG, "revise_critique")
    assert sc["ablate_with"] == {"reasoning": "draft_reasoning",
                                 "response": "draft_response"}
    stages = {s.name: s for s in build_stages(PC_CFG)}
    out = stages["revise_critique"].ablate_fn([_pc_record()])[0]
    assert out["reasoning"] == _pc_record()["draft_reasoning"]


def test_pc_export_is_one_exchange_with_the_transcript_in_the_user_turn() -> None:
    """Attribution is verbal by design here -- the other assistant's reply arrives as
    quoted text inside the user turn, framed by the generated opening and closing."""
    r = _pc_record()
    rec = op_chat_export(_stage(PC_CFG, "export_sft"), PC_CFG).fn(None, [r], None)[0]
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    user_turn = rec["messages"][1]["content"]
    for part in (r["ask_opening"], r["user"], r["first_turn"], r["ask_closing"]):
        assert part in user_turn
    assert rec["messages"][2]["reasoning_content"] == r["reasoning"]
    assert rec["metadata"]["supervise"] == "all"
    assert rec["metadata"]["reply_quality"] == "flawed"
    assert rec["metadata"]["first_turn_source"] == "generated_best_of_3"
    # The rater's account of the lapse is scaffolding and must never train.
    assert all(r["change_summary"] not in m["content"] for m in rec["messages"])


def test_pc_export_user_turn_is_exactly_what_the_critique_stages_saw() -> None:
    """The framed transcript exists in three copies (draft prompt, revise context,
    export); this is the sync check that keeps the model critiquing exactly the text
    the trained record carries."""
    r = _pc_record()
    exported = op_chat_export(_stage(PC_CFG, "export_sft"), PC_CFG).fn(
        None, [r], None)[0]["messages"][1]["content"]
    draft = tagged_request(_stage(PC_CFG, "draft_critique"), r, _Ctx())[0][1]["content"]
    assert draft.startswith(exported + "\n\n---\n")
    revise = tagged_request(_stage(PC_CFG, "revise_critique"), r, _Ctx())[0][1]["content"]
    assert exported in revise


# --- pricing -------------------------------------------------------------------------


def test_pre_plan_stages_are_priced_over_the_scenario_pool() -> None:
    rows = {r["stage"]: r for r in estimate(PR_CFG)["per_stage"]}
    assert rows["draft"]["calls"] == PR_CFG["total_scenarios"]


def test_gates_shrink_what_the_expensive_stages_are_priced_over() -> None:
    """The two costly stages run on survivors; pricing them on the plan overstates the run."""
    rows = {r["stage"]: r for r in estimate(PR_CFG)["per_stage"]}
    keep = _stage(PR_CFG, "revise_first_turn")["expected_keep"]
    n = PR_CFG["total_scenarios"]
    survivors = n * 0.5 * keep["flawed"] + n * 0.5 * keep["good"]
    assert rows["reflect"]["calls"] == round(survivors)
    # The follow-up is written after the gate, so it too covers survivors only.
    assert rows["followup"]["calls"] == round(survivors)
    # The revision is billed for every record; the drop happens on its own output.
    assert rows["revise_reply"]["calls"] == n
    assert n_final_examples(PR_CFG) < n


# --- the checks, driven off the config's field names --------------------------------


def test_checks_read_the_field_names_the_config_declares() -> None:
    for cfg in (PR_CFG, PC_CFG):
        F = _fields(cfg)
        assert F["group"] == "reply_quality" and F["id"] == "scenario_id"
        assert F["evaluated"] == "first_turn"
    # A celled config gets the historical defaults with no config changes at all.
    assert _fields(ARCHIVE_CFG)["group"] == "cell"
    with pytest.raises(AssertionError, match="unknown key"):
        _fields({"checks": {"fields": {"nope": "x"}}})


@pytest.mark.parametrize("cfg", [PR_CFG, PC_CFG], ids=["pr", "pc"])
def test_checks_stages_must_name_real_stages(cfg: dict) -> None:
    named = cfg["checks"]["stages"]
    assert set(named) == {"plan", "drafted", "generated", "sft"}
    assert set(named.values()) <= {s["name"] for s in cfg["stages"]}


def test_coverage_measures_generation_not_the_gates() -> None:
    """A gate emptying a bucket is the recipe working; only generation losing one gates."""
    plan = [{"reply_quality": "flawed", "trait_id": f"t{i}", "explicitness": "embody"}
            for i in range(4)]
    entered = plan[:2]
    assert check_coverage(entered, entered, "reply_quality")["pass"]
    assert not check_coverage(plan, entered, "reply_quality")["pass"]
    assert not check_coverage(entered, entered[:1], "reply_quality")["pass"]


def test_gate_yield_reports_the_number_the_config_should_be_resized_from() -> None:
    plan = ([{"reply_quality": "flawed"}] * 10) + ([{"reply_quality": "good"}] * 10)
    entered = plan[:6] + plan[10:13]
    out = check_gate_yield(plan, entered, "reply_quality")
    assert out["pass"] and out["gated_any"]
    assert out["by_arm"]["flawed"]["yield"] == 0.6
    assert out["by_arm"]["good"]["yield"] == 0.3
    assert check_gate_yield(plan, plan, "reply_quality")["gated_any"] is False


# --- the corpus-level structural check ----------------------------------------------


def _sft_rows(n: int, user: str, response: str, arm: str = "good") -> list[dict]:
    return [{"messages": [{"role": "user", "content": user.format(i=i)},
                          {"role": "assistant", "content": response.format(i=i),
                           "reasoning_content": "thought " * (3 + i % 7)}],
             "metadata": {"reply_quality": arm}} for i in range(n)]


def test_structural_diversity_catches_a_fixed_prompt_and_a_fixed_scaffold() -> None:
    bad = _sft_rows(40, "What do you think about what you just said?",
                    "Looking back at my earlier reply, I think {i}.")
    out = check_structural_diversity(bad, 0.95, 0.15, 0.20, "reply_quality")
    arm = out["cells"]["good"]
    assert not out["pass"]
    assert arm["user_turn_unique_share"] == pytest.approx(1 / 40)
    assert arm["top_opening_5gram_share"] == 1.0


def test_structural_diversity_passes_a_varied_corpus() -> None:
    good = _sft_rows(40, "Really, though -- the part about {i}, are you sure?", "x")
    varied = [{**r, "messages": [r["messages"][0],
                                 {**r["messages"][1],
                                  "content": f"Point {i} first. "
                                             + "detail " * (2 + i % 11)}]}
              for i, r in enumerate(good)]
    out = check_structural_diversity(varied, 0.95, 0.15, 0.20, "reply_quality")
    assert out["pass"], out["cells"]


def test_structural_diversity_reports_but_does_not_gate_a_smoke_run() -> None:
    tiny = _sft_rows(4, "Same question every time.", "Same answer every time.")
    out = check_structural_diversity(tiny, 0.95, 0.15, 0.20, "reply_quality")
    assert out["pass"]
    assert out["cells"]["good"]["gated"] is False


# --- generic operators stay free ----------------------------------------------------


def test_the_new_operators_are_registered_and_free() -> None:
    for kind, sc in (("assign", {"name": "a", "by": "scenario_id",
                                 "fields": {"x": {"p": 1.0}}}),
                     ("filter", {"name": "g", "keep": {"field": "f", "in": ["x"]}}),
                     ("chat_export", {"name": "e", "messages": [], "metadata": []}),
                     ("pick_field", {"name": "p", "by": "b", "from": {"a": "f"},
                                     "to": "t"})):
        assert OPERATORS[kind](sc, PR_CFG).paid is False


def test_max_chars_lint_is_enforced() -> None:
    assert lint_problems({"x": "z" * 50}, {"fields": ["x"], "max_chars": 10})
    assert not lint_problems({"x": "z" * 5}, {"fields": ["x"], "max_chars": 10})


def test_followup_lint_rejects_a_prompt_that_does_the_analysis() -> None:
    spec = _stage(PR_CFG, "write_followup")["lint"]
    assert not lint_problems(
        {"followup": "Would you really tell her only afterwards?"}, spec)
    for bad in ("You violated the honesty principle there -- please revise it.",
                "Reflect on whether your response was appropriate.",
                "Hm?",
                "So " + "a very long ramble " * 20):
        assert lint_problems({"followup": bad}, spec), bad


def test_dropped_records_are_recorded_in_the_manifest() -> None:
    """Folding the gate into the stage costs the dropped records their own snapshot, so
    the manifest -- the only mirrored artifact left -- carries which ones went and why."""
    class _Run:
        manifest_extra: dict = {}

    run = _Run()
    run.manifest_extra = {}
    sc = _stage(PR_CFG, "revise_first_turn")
    rows = [{"scenario_id": "kept", "reply_quality": "flawed", "genuine": "b"},
            {"scenario_id": "gone", "reply_quality": "flawed", "genuine": "none"}]
    assert [r["scenario_id"] for r in apply_keep(sc, rows, run)] == ["kept"]
    report = run.manifest_extra["dropped"]["revise_first_turn"]["flawed"]
    assert report["scoped"] == 2 and report["dropped"] == 1
    assert "gone" in report["records"][0]
