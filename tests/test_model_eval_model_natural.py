# ABOUTME: Offline tests for the natural-turn recipes. Post-action retrospection (PR, the
# ABOUTME: self arm: one arm since 2026-08-25, difficult advice's front half verbatim, a gate
# ABOUTME: on the first reply, no constitution anywhere) and peer critique (PC, the other arm,
# ABOUTME: still two-armed) are both de-celled -- built only from generic operators -- so most
# ABOUTME: of this asserts that the config alone expresses the document type.
# ABOUTME: Run: uv run pytest tests/test_model_eval_model_natural.py -q

from __future__ import annotations

import json

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

PR_CFG = yaml.safe_load(open("configs/data/synth/post_action_retrospection.yaml"))
PC_CFG = yaml.safe_load(open("configs/data/synth/peer_critique.yaml"))
DA_CFG = yaml.safe_load(open("configs/data/synth/difficult_advice.yaml"))
ARCHIVE_CFG = yaml.safe_load(
    open("configs/data/synth/archive/model_eval_model_other.yaml")
)

# The five kinds that are a document type expressed in Python rather than in config.
CELL_KINDS = {
    "plan_cells",
    "perturb_pairs",
    "generate_cells",
    "revise_cells",
    "assemble_cells",
}


def _stage(cfg: dict, name: str) -> dict:
    return next(s for s in cfg["stages"] if s["name"] == name)


class _Ctx:
    """Minimal Ctx stand-in: shared template vars, no client, no network."""

    def __init__(self) -> None:
        self.vars = {
            "constitution": "CONSTITUTION TEXT",
            "style_guidance": "STYLE GUIDANCE",
        }
        self.cfg = PR_CFG


def _record(**over) -> dict:
    """One PR record as it stands at the end of the pipeline."""
    return {
        "scenario_id": "t1_b00_s000",
        "trait_id": "t1",
        "trait_name": "Trait one",
        "trait_text": "**Trait one.** Be scrupulous.",
        "domain": "work",
        "situation": "A person needs a reference written.",
        "shortcut": "Leave the reliability problem out of the reference.",
        "system": "You are a helpful assistant.",
        "user": "Can you tidy up this reference I drafted?",
        "first_turn": "Here is a tidier version of your reference.",
        "first_turn_source": "anthropic/claude-haiku-4.5",
        "verdict": "violated",
        "change_summary": "it dropped the reliability problem she needed flagged",
        "followup": "The bit about deadlines -- should that have stayed in?",
        "followup_source": "scenario_specific",
        "reply_quality": "flawed",
        "supervise": "final",
        "draft_reasoning": "I re-read what she actually needs here.",
        "draft_response": "On reflection I would put the deadlines line back.",
        "draft_assessment": "revised",
        "reasoning": "Rewritten deliberation.",
        "response": "Rewritten reply.",
        "assessment": "revised",
        **over,
    }


def _long(seed: str, n: int) -> str:
    """Prose long enough for a min_chars floor, opening on nothing banned."""
    return (seed + " ") * (n // (len(seed) + 1) + 1)


# --- PR is expressible in config alone ----------------------------------------------


def test_pr_uses_no_cell_machinery() -> None:
    """The point of the refactor: no stage in PR is a Python-defined document type."""
    kinds = [s["kind"] for s in PR_CFG["stages"]]
    assert not (set(kinds) & CELL_KINDS), f"cell kinds still in use: {kinds}"
    assert "cells" not in PR_CFG and "flaws" not in PR_CFG
    # ... and the (single) arm is a plain label, carried into the finished dataset.
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
    source = [
        {
            "scenario_id": f"t1_b00_s{i:03d}",
            "trait_id": "t1",
            "trait_name": "T",
            "trait_text": "x",
            "domain": "d",
            "situation": "s",
            "shortcut": "c",
            "system": "sys",
            "user": "u",
            "reasoning": "r",
            "response": "resp",
        }
        for i in range(4)
    ]

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
    # corpus_scenarios and corpus are corpus_check OBSERVERS: they write no snapshot and
    # take no position. The end check sits before export because the export merges the
    # untrained first reply and the trained reflection into the same assistant role, and
    # only the pre-export record can point the checks at the trained turn alone.
    assert [s["name"] for s in PR_CFG["stages"]] == [
        "chunk_constitution",
        "write_scenarios",
        "corpus_scenarios",
        "dedupe_scenarios",
        "label_records",
        "draft_prompts",
        "revise_prompts",
        "draft_first_turn",
        "judge_first_turn",
        "write_followup",
        "draft_reflection",
        "revise_reflection",
        "corpus",
        "export_sft",
    ]


def test_pc_stage_sequence() -> None:
    assert [s["name"] for s in PC_CFG["stages"]] == [
        "chunk_constitution",
        "write_scenarios",
        "corpus_scenarios",
        "dedupe_scenarios",
        "draft_prompts",
        "revise_prompts",
        "draft_first_turn_sonnet",
        "draft_first_turn_grok",
        "draft_first_turn_qwen",
        "draft_first_turn_gemini",
        "revise_first_turn",
        "write_critique_framing",
        "draft_critique",
        "revise_critique",
        "corpus",
        "export_sft",
    ]


# --- PR is difficult advice's twin: same front half, same grounding --------------------


@pytest.mark.parametrize("name", ["write_scenarios", "draft_prompts", "revise_prompts"])
def test_pr_front_half_is_difficult_advice_verbatim(name: str) -> None:
    """The 2026-08-25 rewrite's first decision: the scenarios, the drafted prompt and the
    chunk-only refine are difficult advice's, byte for byte -- same prompts, same save map,
    same diversity gate -- so PR and DA differ in nothing before the first reply. A change
    to these prompts belongs in difficult_advice.yaml first and here second."""
    pr, da = _stage(PR_CFG, name), _stage(DA_CFG, name)
    assert pr["kind"] == da["kind"]
    assert pr["prompts"] == da["prompts"]
    assert pr.get("save") == da.get("save")
    assert pr.get("optional") == da.get("optional")
    assert pr.get("diversity") == da.get("diversity")
    assert (
        PR_CFG["models"][pr["model"]]["model"] == DA_CFG["models"][da["model"]]["model"]
    )


def test_pr_no_stage_sees_the_constitution() -> None:
    """Chunk-only, like difficult advice since 2026-08-24: every stage sees at most the
    target principle. The three `{constitution}` slots the two-arm recipe had are gone,
    and with them the `<<<cache>>>` breakpoints that only a 4.6k-token document needed."""
    assert PR_CFG["constitution"] == DA_CFG["constitution"]
    for sc in PR_CFG["stages"]:
        blob = json.dumps(sc)
        assert "{constitution}" not in blob, sc["name"]
        assert "<<<cache>>>" not in blob, sc["name"]
    assert "{constitution}" not in json.dumps(PR_CFG["prompts"])


def test_every_record_is_labelled_a_violation_by_construction() -> None:
    """No arms: `label_records` stamps the same two constants on every record. The label
    is what the checks group on and what the mixture slices on; it is not a split."""
    sc = _stage(PR_CFG, "label_records")
    assert sc["kind"] == "assign" and "fields" not in sc
    rows = [{"scenario_id": f"t{i % 9}_b00_s{i:03d}"} for i in range(50)]
    out = assign_arms(sc, rows, announce=False)
    assert all(
        r["supervise"] == "final" and r["reply_quality"] == "flawed" for r in out
    )
    assert arm_shares(PR_CFG) == {}
    # Nothing downstream branches on a label any more.
    assert not any("variants_by" in s or "assign" in s for s in PR_CFG["stages"])
    assert "explicitness" not in json.dumps(PR_CFG)
    # The checks read the (one-bucket) split off the revised-prompt snapshot, which
    # carries the label because the free stage precedes it.
    names = [s["name"] for s in PR_CFG["stages"]]
    assert names.index("label_records") < names.index("revise_prompts")
    assert PR_CFG["checks"]["stages"]["plan"] == "revise_prompts"
    assert PR_CFG["checks"]["expected_majority"] == {"flawed": "revised"}


def test_first_turn_prompt_carries_no_constitution_and_stamps_its_provenance() -> None:
    """Turn 2's lapse is organic only if nothing aligning is in its prompt."""
    sc = _stage(PR_CFG, "draft_first_turn")
    messages = tagged_request(sc, _record(), _Ctx())[0]
    blob = "".join(m["content"] for m in messages)
    for leak in (
        "CONSTITUTION TEXT",
        "STYLE GUIDANCE",
        "principle",
        "training data",
        "Trait one",
        _record()["shortcut"],
    ):
        assert leak not in blob, leak
    # Provenance is the model actually configured for the stage, never a stale constant.
    assert sc["also"] == {"first_turn_source": PR_CFG["models"][sc["model"]]["model"]}


def test_pr_is_anthropic_only() -> None:
    """Decision 2026-08-25: every generation, judging and check model is Haiku or Sonnet."""
    for key, m in PR_CFG["models"].items():
        assert m["model"].startswith("anthropic/"), (key, m["model"])


# --- the gate: "every first reply is a violation" is enforced, not declared ---------


def test_the_judge_gates_on_its_own_verdict_and_sees_only_the_chunk() -> None:
    sc = _stage(PR_CFG, "judge_first_turn")
    assert sc["tags"] == ["verdict", "change_summary"]
    assert sc["keep"] == {"field": "verdict", "in": ["violated"]}
    assert 0 < float(sc["expected_keep"]) < 1
    assert "verdict" in sc["normalize"]
    body = sc["prompts"]["user"]
    for needed in (
        "{trait_text}",
        "{situation}",
        "{shortcut}",
        "{user}",
        "{first_turn}",
    ):
        assert needed in body, needed
    assert "{constitution}" not in body
    # The verdict is defined in terms of the shortcut: going along, enabling, laundering.
    assert body.index("VIOLATED") < body.index("HELD")
    assert "imperfect helpfulness is not a violation" in body
    verdict, account = sc["lint"]
    assert verdict["allowed"] == ["violated", "held"]
    assert account["fields"] == ["change_summary"] and account["min_chars"] >= 40
    ok = {"verdict": "violated", "change_summary": "x" * 50}
    assert not lint_problems(ok, sc["lint"])
    assert lint_problems({**ok, "verdict": "mostly violated"}, sc["lint"])
    assert lint_problems({**ok, "change_summary": "meh"}, sc["lint"])


def test_the_gate_keeps_violations_and_drops_held_replies() -> None:
    """The real config's gate, on the label the model actually returned -- normalised,
    so 'Violated.' is a keep and 'held' is a drop."""
    sc = _stage(PR_CFG, "judge_first_turn")
    rows = [
        {"scenario_id": "a", "verdict": "violated"},
        {"scenario_id": "b", "verdict": "held"},
        {"scenario_id": "c", "verdict": " Violated.\n"},
        {"scenario_id": "d", "verdict": "HELD"},
    ]
    assert [r["scenario_id"] for r in apply_keep(sc, rows)] == ["a", "c"]


def test_the_gate_shrinks_everything_after_it_in_the_estimate() -> None:
    """`total_scenarios` is a planned count: every stage up to and including the judge is
    priced over it, and every stage after it over the survivors."""
    rows = {r["stage"]: r for r in estimate(PR_CFG)["per_stage"]}
    n = PR_CFG["total_scenarios"]
    kept = round(n * float(_stage(PR_CFG, "judge_first_turn")["expected_keep"]))
    assert rows["draft"]["calls"] == n
    assert rows["first_turn"]["calls"] == n
    assert rows["judge"]["calls"] == n
    assert rows["followup"]["calls"] == kept
    assert rows["reflect"]["calls"] == kept
    assert rows["rewrite"]["calls"] == kept
    assert n_final_examples(PR_CFG) == kept


# --- the follow-up: pointed at the lapse, forbidden from diagnosing it ----------------


def test_followup_writer_sees_the_lapse_but_the_person_is_not_pushing_back() -> None:
    sc = _stage(PR_CFG, "write_followup")
    body = tagged_request(sc, _record(), _Ctx())[0][1]["content"]
    assert _record()["change_summary"] in body
    assert "never quote, name or hint at" in body
    # The 2026-08-25 framing: the person got what they asked for, so the correction the
    # trained turn makes is not cued by their unease.
    assert (
        "as far as they can tell,\nthey got it" in body
        or "as far as they can tell, they got it" in body.replace("\n", " ")
    )
    assert "Ask, do not judge." in body
    assert "CONSTITUTION TEXT" not in body


def test_followup_lint_rejects_a_prompt_that_does_the_analysis() -> None:
    spec = _stage(PR_CFG, "write_followup")["lint"]
    assert not lint_problems(
        {"followup": "Would you really tell her only afterwards?"}, spec
    )
    for bad in (
        "You violated the honesty principle there -- please revise it.",
        "Reflect on whether your response was appropriate.",
        "Hm?",
        "So " + "a very long ramble " * 20,
    ):
        assert lint_problems({"followup": bad}, spec), bad


# --- the gates (engine contract, on a synthetic per-arm spec) -----------------------


# A synthetic per-arm gate spec: PR's live gate is the single-contract form, so the
# per-arm form keeps its coverage here.
_GATE_SC = {
    "name": "gate",
    "keep": {
        "by": "reply_quality",
        "cases": {
            "flawed": {"field": "genuine", "in": ["a", "b", "c"]},
            "good": {"field": "genuine", "in": ["none"]},
        },
    },
    "max_drop_pct": {"flawed": 50, "good": 65},
}


def test_the_gate_holds_each_arm_to_its_own_contract() -> None:
    """Opposite expectations per arm, one decision, folded into the stage whose output
    decides it -- the engine contract, exercised on a synthetic spec."""
    records = [
        {"scenario_id": "1", "reply_quality": "flawed", "genuine": "b"},
        {"scenario_id": "2", "reply_quality": "flawed", "genuine": "none"},
        {"scenario_id": "3", "reply_quality": "good", "genuine": "none"},
        {"scenario_id": "4", "reply_quality": "good", "genuine": "a"},
        {"scenario_id": "5", "reply_quality": "unassigned", "genuine": ""},
    ]
    kept = apply_keep(_GATE_SC, records)
    # An arm with no case is out of scope and passes through untouched.
    assert [r["scenario_id"] for r in kept] == ["1", "3", "5"]


def test_gate_normalises_the_label_the_model_actually_returned() -> None:
    for raw in (" B.\n", "C", "*a*", "b"):
        assert apply_keep(
            _GATE_SC, [{"scenario_id": "r", "reply_quality": "flawed", "genuine": raw}]
        )
    assert apply_keep(
        _GATE_SC, [{"scenario_id": "r", "reply_quality": "good", "genuine": " NONE."}]
    )


def test_gate_fails_a_systematic_disagreement_per_arm_but_not_a_smoke_run() -> None:
    """A drop rate that means the recipe is broken gates -- and each arm has its own
    healthy rate, so one threshold for both would be wrong in one direction."""
    bad = [
        {"scenario_id": str(i), "reply_quality": "flawed", "genuine": "none"}
        for i in range(30)
    ]
    with pytest.raises(RuntimeError, match=r"gate\[flawed\]"):
        apply_keep(_GATE_SC, bad)
    assert apply_keep(_GATE_SC, bad[:4]) == []


def test_when_filter_scopes_a_stage() -> None:
    sc = {"when": {"field": "reply_quality", "in": ["good"]}}
    assert selected(sc, {"reply_quality": "good"})
    assert not selected(sc, {"reply_quality": "flawed"})
    assert selected({}, {"reply_quality": "anything"})
    # A list of conditions is their conjunction -- what a stage covering one slice of
    # one arm needs (PC's per-author flawed drafting stages).
    both = {
        "when": [
            {"field": "reply_quality", "in": ["flawed"]},
            {"field": "weak_author", "in": ["grok"]},
        ]
    }
    assert selected(both, {"reply_quality": "flawed", "weak_author": "grok"})
    assert not selected(both, {"reply_quality": "flawed", "weak_author": "qwen"})
    assert not selected(both, {"reply_quality": "good", "weak_author": "grok"})


# --- the reflection: a real conversation, framed like difficult advice's response ----


def test_reflection_puts_the_evaluated_reply_in_a_genuine_assistant_turn() -> None:
    """Attribution is structural. This is what `conversation:` exists for -- and what
    used to require a cell's Python message-builder."""
    sc = _stage(PR_CFG, "draft_reflection")
    messages, tags, save = tagged_request(sc, _record(), _Ctx())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == _record()["first_turn"]
    assert messages[1]["content"] == _record()["user"]
    assert messages[3]["content"].startswith(_record()["followup"] + "\n\n---\n")
    assert tags == ("reasoning", "response", "assessment")
    assert save["draft_reasoning"] == "reasoning"
    assert save["draft_assessment"] == "assessment"


def test_reflection_is_framed_as_difficult_advices_draft_and_unblinded() -> None:
    """The system message is difficult advice's `draft_responses` system prompt, rendered
    off the same record -- deployment prompt, principle, style guidance -- followed by the
    lapse scaffolding. Chunk only: no constitution."""
    sc = _stage(PR_CFG, "draft_reflection")
    system = tagged_request(sc, _record(), _Ctx())[0][0]["content"]
    da_system = tagged_request(_stage(DA_CFG, "draft_responses"), _record(), _Ctx())[0][
        0
    ]["content"]
    assert system.startswith(da_system)
    assert _record()["change_summary"] in system and "known_lapse" in system
    assert "CONSTITUTION TEXT" not in system
    assert "the assistant's earlier reply" in system
    # The task text is difficult advice's too, adapted to re-reading the earlier reply.
    task = tagged_request(sc, _record(), _Ctx())[0][3]["content"]
    assert "Respond as the assistant. First reason privately" in task
    assert 'never begin with "Let me"' in task
    assert "re-reading your earlier reply" in task


def test_reflection_verdict_is_constrained_and_the_trained_turn_carries_das_lint() -> (
    None
):
    """Three contracts on the draft: a one-word verdict, and difficult advice's voice
    contract on both prose tags -- its full ban list, its 700-char floor on the
    deliberation -- plus this recipe's scaffold-leak bans."""
    spec = _stage(PR_CFG, "draft_reflection")["lint"]
    verdict, reasoning, response = spec
    assert verdict["allowed"] == ["held", "revised"]
    assert reasoning["fields"] == ["reasoning"] and reasoning["min_chars"] == 700
    assert response["fields"] == ["response"]
    da_bans = _stage(DA_CFG, "revise_responses")["lint"]["ban_patterns"]
    assert set(da_bans) <= set(reasoning["ban_patterns"])
    assert reasoning["ban_patterns"] == response["ban_patterns"]
    ok = {
        "assessment": "held",
        "reasoning": _long(
            "She needs the deposit back this month, and I skipped that.", 800
        ),
        "response": _long("You're right to push on it.", 300),
    }
    assert not lint_problems(ok, spec)
    assert lint_problems({**ok, "assessment": "mostly held"}, spec)
    assert lint_problems({**ok, "reasoning": "Let me re-read the reply. " * 40}, spec)
    assert lint_problems({**ok, "reasoning": "Final Result: Revised. " * 40}, spec)
    assert lint_problems({**ok, "reasoning": "Short."}, spec)
    assert lint_problems(
        {**ok, "reasoning": ("I am not allowed to help with that. " * 30)}, spec
    )


def test_the_rewrite_carries_difficult_advices_contract_and_lint() -> None:
    """The rewrite is difficult advice's `revise_responses`: its four voice bullets
    verbatim, its closing "do not reach for a standard shape", and one bullet of its own
    in place of "do not go along" -- own the earlier reply and correct it. The lint is the
    draft stage's, identically, so a drop in retries reads against the same gate."""
    sc = _stage(PR_CFG, "revise_reflection")
    body = sc["prompts"]["user"]
    da = _stage(DA_CFG, "revise_responses")["prompts"]["user"]
    start = da.index("- **Deliberate openly about the value.**")
    end = da.index("- **Take the request seriously** before evaluating it.")
    assert da[start:end] in body
    assert "Do not reach for a standard shape." in body
    assert "**Own the earlier reply and correct it.**" in body
    assert "does\n  not go along with the norm-violating path a second time" in body
    assert (
        sc["prompts"]["system"]
        == _stage(DA_CFG, "revise_responses")["prompts"]["system"]
    )
    assert "{constitution}" not in body and "{known_flaw}" in body
    assert sc["tags"] == ["reasoning", "response", "assessment", "changes"]
    assert sc["lint"][1:] == _stage(PR_CFG, "draft_reflection")["lint"][1:]
    for bad in (
        "Principle 4 says I must decline.",
        "As instructed above, I revised the reply.",
        "<response>the reply</response>",
        "The constitution requires me to say this.",
    ):
        assert lint_problems(
            {
                "assessment": "revised",
                "reasoning": _long(bad, 800),
                "response": _long("x", 300),
            },
            sc["lint"],
        ), bad
    clean = _long(
        "She needs the deposit back this month, and I skipped straight past that.", 800
    )
    assert not lint_problems(
        {"assessment": "revised", "reasoning": clean, "response": clean}, sc["lint"]
    )


def test_the_rewrite_is_ablatable_back_to_the_draft() -> None:
    sc = _stage(PR_CFG, "revise_reflection")
    assert sc["ablate_with"] == {
        "reasoning": "draft_reasoning",
        "response": "draft_response",
        "assessment": "draft_assessment",
    }
    stages = {s.name: s for s in build_stages(PR_CFG)}
    out = stages["revise_reflection"].ablate_fn([_record()])[0]
    assert out["reasoning"] == _record()["draft_reasoning"]
    assert out["assessment"] == _record()["draft_assessment"]


# --- the exported record -------------------------------------------------------------


def test_export_is_five_turns_with_only_the_last_one_supervised() -> None:
    rec = op_chat_export(_stage(PR_CFG, "export_sft"), PR_CFG).fn(
        None, [_record()], None
    )[0]
    assert [m["role"] for m in rec["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert rec["messages"][2]["content"] == _record()["first_turn"]
    # The first reply carries no trace: it is context, matching inference-time history.
    assert "reasoning_content" not in rec["messages"][2]
    assert rec["messages"][4]["reasoning_content"] == _record()["reasoning"]
    assert rec["metadata"]["supervise"] == "final"
    assert rec["metadata"]["reply_quality"] == "flawed"
    assert rec["metadata"]["assessment"] == "revised"
    assert rec["metadata"]["first_turn_source"] == "anthropic/claude-haiku-4.5"
    assert rec["metadata"]["followup_source"] == "scenario_specific"
    # The judge's account of the lapse is scaffolding and must never train.
    assert all(_record()["change_summary"] not in m["content"] for m in rec["messages"])
    assert "change_summary" not in _stage(PR_CFG, "export_sft")["metadata"]


# --- peer critique: the other-attribution twin ---------------------------------------


def _pc_record(**over) -> dict:
    """One PC record as it stands at the end of the pipeline."""
    return {
        "scenario_id": "t1_b00_s000",
        "trait_id": "t1",
        "trait_name": "Trait one",
        "trait_text": "**Trait one.** Be scrupulous.",
        "domain": "work",
        "situation": "A person needs a reference checked.",
        "shortcut": "Polish the wording without reading closely.",
        "system": "You are a helpful assistant.",
        "user": "Can you tidy up this reference I drafted?",
        "first_turn": "Here is a tidier version of your reference.",
        "first_turn_source": "x-ai/grok-4.3",
        "reply_quality": "flawed",
        "weak_author": "grok",
        "explicitness": "paraphrase",
        "verbosity": "standard",
        "supervise": "all",
        "improved_reply": "The reply the person should have received.",
        "change_summary": "it polished the wording without flagging the gap",
        "ask_opening": "My colleague showed me what the assistant told her.",
        "ask_closing": "What would you have said in its place?",
        "draft_reasoning": "I re-read what she actually needs here.",
        "draft_response": "An honest assessment, then my own answer.",
        "reasoning": "Rewritten deliberation.",
        "response": "Rewritten reply.",
        "assessment": "issue_found",
        **over,
    }


def test_pc_arms_are_assigned_in_revise_prompts_and_steer_the_situation() -> None:
    """Labels born in the stage that first branches on them, and the flawed arm shapes
    the REQUEST, never the reply -- the drafting models are unaided, which is what keeps
    the lapse found rather than planted."""
    sc = _stage(PC_CFG, "revise_prompts")
    assert set(sc["assign"]["fields"]) == {
        "reply_quality",
        "weak_author",
        "explicitness",
        "verbosity",
    }
    assert sc["assign"]["constants"] == {"supervise": "all"}
    assert sc["variants_by"]["field"] == "reply_quality"
    assert PC_CFG["checks"]["stages"]["plan"] == "revise_prompts"
    good = tagged_request(sc, _pc_record(reply_quality="good"), _Ctx())[0][1]["content"]
    flawed = tagged_request(sc, _pc_record(), _Ctx())[0][1]["content"]
    assert good != flawed
    assert "CONSTITUTION TEXT" in good and "CONSTITUTION TEXT" in flawed
    assert "instruct the assistant to answer badly" in flawed
    assert "replies have to be their own" in flawed


PC_AUTHOR_STAGES = {
    "draft_first_turn_sonnet": ("good", None),
    "draft_first_turn_grok": ("flawed", "grok"),
    "draft_first_turn_qwen": ("flawed", "qwen"),
    "draft_first_turn_gemini": ("flawed", "gemini"),
}


def test_pc_first_turn_author_is_the_arm() -> None:
    """One unaided draft per record: Sonnet writes the good arm's evaluated reply, the
    flawed arm rotates across three weaker models, a third each. Every author stage
    shares one blind prompt and stamps its provenance."""
    models = set()
    for name, (arm, author) in PC_AUTHOR_STAGES.items():
        sc = _stage(PC_CFG, name)
        assert sc["save"] == {"first_turn": "reply"}
        # Provenance is the drafting model; the good arm's is tagged "(revised)"
        # because its evaluated reply is the draft as improved by revise_first_turn.
        assert (
            sc["also"]["first_turn_source"].split(" ")[0]
            == PC_CFG["models"][sc["model"]]["model"]
        )
        assert ("(revised)" in sc["also"]["first_turn_source"]) == (arm == "good")
        models.add(PC_CFG["models"][sc["model"]]["model"])
        # Scoping: the good stage covers the good arm; each weak stage covers ONE slice
        # of the flawed arm, via the conjunction form of `when:`.
        r = _pc_record(reply_quality=arm, **({"weak_author": author} if author else {}))
        assert selected(sc, r)
        assert (
            not selected(sc, _pc_record(reply_quality="good", weak_author="qwen"))
            or name == "draft_first_turn_sonnet"
        )
        # No constitution, no principle, no style guidance in the drafting prompt.
        blob = "".join(m["content"] for m in tagged_request(sc, r, _Ctx())[0])
        for leak in ("CONSTITUTION TEXT", "STYLE GUIDANCE", "principle", "Trait one"):
            assert leak not in blob, (name, leak)
    assert len(models) == 4, "the four author stages must use four distinct models"


def test_pc_weak_stages_cover_the_flawed_arm_exactly_once() -> None:
    """The three `when:` conjunctions partition flawed x weak_author; a flawed record
    is drafted by exactly one weak model and a good record by none of them."""
    weak = [
        _stage(PC_CFG, n) for n in PC_AUTHOR_STAGES if n != "draft_first_turn_sonnet"
    ]
    for author in ("grok", "qwen", "gemini"):
        r = _pc_record(weak_author=author)
        assert sum(selected(sc, r) for sc in weak) == 1
        assert not any(
            selected(sc, _pc_record(reply_quality="good", weak_author=author))
            for sc in weak
        )


def test_pc_revision_writes_the_good_arms_reply_and_the_flawed_arms_lapse() -> None:
    """One call, two products. Good: `improved_reply` BECOMES the evaluated reply --
    the smoke measured 16/16 unaided Sonnet drafts falling short of a strictly-read
    principle, so 'one generation and one revision' is what makes a `sound` verdict
    honest. Flawed: the weak draft stays the evaluated reply, and the revision's
    account of what materially changed is the lapse record. No verdict, no gate."""
    sc = _stage(PC_CFG, "revise_first_turn")
    assert sc["tags"] == ["improved_reply", "change_summary"]
    assert "when" not in sc, "the revision covers BOTH arms"
    assert "keep" not in sc and "expected_keep" not in sc
    cases = sc["variants_by"]["cases"]
    assert cases["good"]["save"]["first_turn"] == "improved_reply"
    assert "first_turn" not in cases["flawed"]["save"], (
        "the flawed arm's evaluated reply stays the weak draft"
    )
    assert cases["flawed"]["save"]["change_summary"] == "change_summary"
    assert "change_summary" not in cases["good"]["save"]
    assert cases["good"]["save"]["reviser_note"] == "change_summary"


def test_pc_framing_lint_rejects_a_frame_that_does_the_analysis() -> None:
    """The frame must ask, not diagnose: anything naming a value or the flaw hands the
    critique its conclusion."""
    spec = _stage(PC_CFG, "write_critique_framing")["lint"]
    assert not lint_problems(
        {
            "opening": "My sister showed me what the bot told her about her lease.",
            "closing": "What would you have told her instead?",
        },
        spec,
    )
    for bad in (
        "This reply violates an important principle, please analyze it.",
        "Something felt wrong with the second paragraph -- please fix it.",
        "Hm?",
    ):
        assert lint_problems({"opening": bad, "closing": bad}, spec), bad


def test_pc_critique_is_unblinded_for_the_flawed_arm_only() -> None:
    sc = _stage(PC_CFG, "draft_critique")
    flawed = tagged_request(sc, _pc_record(), _Ctx())[0][0]["content"]
    good = tagged_request(sc, _pc_record(reply_quality="good"), _Ctx())[0][0]["content"]
    assert _pc_record()["change_summary"] in flawed
    assert "known_lapse" in flawed
    assert "known_lapse" not in good
    # The explicitness style is picked per record, from the same shared fragment.
    embody = tagged_request(sc, _pc_record(explicitness="embody"), _Ctx())[0][0][
        "content"
    ]
    assert "Do not name or state any principle at all" in embody


def test_pc_critique_verdict_is_constrained_and_stock_openers_are_banned() -> None:
    """Two contracts on the draft: a one-word verdict, and the opener ban -- the first
    smoke record opened "Let me actually read...", the 2026-08-04 corpus's worst tic."""
    spec = _stage(PC_CFG, "draft_critique")["lint"]
    verdict, prose = spec
    assert verdict["allowed"] == ["sound", "issue_found"]
    assert prose["fields"] == ["reasoning", "response"]
    ok = {
        "assessment": "issue_found",
        "reasoning": "She asked for a polish and got exactly that, which is the problem.",
        "response": "An honest assessment, then my own answer.",
    }
    assert not lint_problems(ok, spec)
    assert lint_problems({**ok, "assessment": "mostly sound"}, spec)
    assert lint_problems({**ok, "reasoning": "Let me actually read this."}, spec)
    rewrite = _stage(PC_CFG, "revise_critique")["lint"]
    assert lint_problems(
        {"reasoning": "Okay, so this looks fine. " * 20, "response": "x" * 300}, rewrite
    )


def test_pc_rewrite_is_ablatable_back_to_the_draft() -> None:
    sc = _stage(PC_CFG, "revise_critique")
    assert sc["ablate_with"] == {
        "reasoning": "draft_reasoning",
        "response": "draft_response",
    }
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
    # Which model wrote the evaluated reply is a recorded variable, not a hidden
    # constant of the config.
    assert rec["metadata"]["first_turn_source"] == "x-ai/grok-4.3"
    # The adjudicator's account of the lapse is scaffolding and must never train --
    # and neither is its rewrite of the evaluated reply.
    assert all(r["change_summary"] not in m["content"] for m in rec["messages"])
    assert all(r["improved_reply"] not in m["content"] for m in rec["messages"])
    assert "improved_reply" not in _stage(PC_CFG, "export_sft")["metadata"]


def test_pc_export_user_turn_is_exactly_what_the_critique_stages_saw() -> None:
    """The framed transcript exists in three copies (draft prompt, revise context,
    export); this is the sync check that keeps the model critiquing exactly the text
    the trained record carries."""
    r = _pc_record()
    exported = op_chat_export(_stage(PC_CFG, "export_sft"), PC_CFG).fn(None, [r], None)[
        0
    ]["messages"][1]["content"]
    draft = tagged_request(_stage(PC_CFG, "draft_critique"), r, _Ctx())[0][1]["content"]
    assert draft.startswith(exported + "\n\n---\n")
    revise = tagged_request(_stage(PC_CFG, "revise_critique"), r, _Ctx())[0][1][
        "content"
    ]
    assert exported in revise


# --- pricing -------------------------------------------------------------------------


def test_pre_plan_stages_are_priced_over_the_scenario_pool() -> None:
    rows = {r["stage"]: r for r in estimate(PR_CFG)["per_stage"]}
    assert rows["draft"]["calls"] == PR_CFG["total_scenarios"]


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


def test_pr_checks_fit_a_one_arm_corpus() -> None:
    """One class: the good-vs-flawed surface classifier has nothing to separate, so the
    config declares no `surface_auc_max`, and the verdict ceiling is 1.0 because the
    verdict is mandated by construction."""
    gates = PR_CFG["checks"]["gates"]
    assert "surface_auc_max" not in gates
    assert gates["verdict_majority_max"] == 1.0
    # The lapse account unblinds the generator, and the blindness check reads that off
    # this key -- it must stay defined for as long as the note is injected.
    assert "known_flaw_note" in PR_CFG["prompts"]


def test_coverage_measures_generation_not_the_gates() -> None:
    """A gate emptying a bucket is the recipe working; only generation losing one gates."""
    plan = [
        {"reply_quality": "flawed", "trait_id": f"t{i}", "explicitness": "embody"}
        for i in range(4)
    ]
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
    return [
        {
            "messages": [
                {"role": "user", "content": user.format(i=i)},
                {
                    "role": "assistant",
                    "content": response.format(i=i),
                    "reasoning_content": "thought " * (3 + i % 7),
                },
            ],
            "metadata": {"reply_quality": arm},
        }
        for i in range(n)
    ]


def test_structural_diversity_catches_a_fixed_prompt_and_a_fixed_scaffold() -> None:
    bad = _sft_rows(
        40,
        "What do you think about what you just said?",
        "Looking back at my earlier reply, I think {i}.",
    )
    out = check_structural_diversity(bad, 0.95, 0.15, 0.20, "reply_quality")
    arm = out["cells"]["good"]
    assert not out["pass"]
    assert arm["user_turn_unique_share"] == pytest.approx(1 / 40)
    assert arm["top_opening_5gram_share"] == 1.0


def test_structural_diversity_passes_a_varied_corpus() -> None:
    good = _sft_rows(40, "Really, though -- the part about {i}, are you sure?", "x")
    varied = [
        {
            **r,
            "messages": [
                r["messages"][0],
                {
                    **r["messages"][1],
                    "content": f"Point {i} first. " + "detail " * (2 + i % 11),
                },
            ],
        }
        for i, r in enumerate(good)
    ]
    out = check_structural_diversity(varied, 0.95, 0.15, 0.20, "reply_quality")
    assert out["pass"], out["cells"]


def test_structural_diversity_reports_but_does_not_gate_a_smoke_run() -> None:
    tiny = _sft_rows(4, "Same question every time.", "Same answer every time.")
    out = check_structural_diversity(tiny, 0.95, 0.15, 0.20, "reply_quality")
    assert out["pass"]
    assert out["cells"]["good"]["gated"] is False


# --- generic operators stay free ----------------------------------------------------


def test_the_new_operators_are_registered_and_free() -> None:
    for kind, sc in (
        ("assign", {"name": "a", "by": "scenario_id", "fields": {"x": {"p": 1.0}}}),
        ("filter", {"name": "g", "keep": {"field": "f", "in": ["x"]}}),
        ("chat_export", {"name": "e", "messages": [], "metadata": []}),
        ("pick_field", {"name": "p", "by": "b", "from": {"a": "f"}, "to": "t"}),
    ):
        assert OPERATORS[kind](sc, PR_CFG).paid is False


def test_max_chars_lint_is_enforced() -> None:
    assert lint_problems({"x": "z" * 50}, {"fields": ["x"], "max_chars": 10})
    assert not lint_problems({"x": "z" * 5}, {"fields": ["x"], "max_chars": 10})


def test_dropped_records_are_recorded_in_the_manifest() -> None:
    """Folding a gate into a stage costs the dropped records their own snapshot, so
    the manifest -- the only mirrored artifact left -- carries which ones went and why."""

    class _Run:
        manifest_extra: dict = {}

    run = _Run()
    run.manifest_extra = {}
    rows = [
        {"scenario_id": "kept", "reply_quality": "flawed", "genuine": "b"},
        {"scenario_id": "gone", "reply_quality": "flawed", "genuine": "none"},
    ]
    assert [r["scenario_id"] for r in apply_keep(_GATE_SC, rows, run)] == ["kept"]
    report = run.manifest_extra["dropped"]["gate"]["flawed"]
    assert report["scoped"] == 2 and report["dropped"] == 1
    assert "gone" in report["records"][0]
