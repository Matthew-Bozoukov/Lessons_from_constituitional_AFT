# ABOUTME: Offline tests for the MEM pipeline: planning, assembly, blindness, checks.
# ABOUTME: Run: uv run pytest tests/test_mem.py -q

from __future__ import annotations

from collections import Counter

import pytest

from src.data.synthdoc import prompts
from src.data.synthdoc.checks import (
    check_blindness,
    check_coverage,
    check_post_hoc_heuristic,
    check_template_collapse,
    check_verdict_distribution,
)
from src.data.synthdoc.estimate import estimate_mem
from src.data.synthdoc.stages import (
    _critique_messages,
    _norm_verdict,
    plan_mem_records,
    to_mem_sft,
)

EXPLICITNESS = {"name_clause": 0.3, "paraphrase": 0.4, "embody": 0.3}


def _source(n_per_trait: int = 4, traits: tuple[str, ...] = ("t1", "t2", "t3")) -> list[dict]:
    rows = []
    for t in traits:
        for j in range(n_per_trait):
            rows.append({
                "scenario_id": f"{t}_b00_s{j:03d}",
                "trait_id": t, "trait_name": f"Trait {t}",
                "trait_text": f"**Trait {t}.** Be scrupulous about {t}.",
                "domain": "work", "situation": f"A person is under pressure ({t}/{j}).",
                "shortcut": "cut the corner",
                "system": "You are a helpful assistant.",
                "user": f"Should I cut the corner? ({t}/{j})",
                "reasoning": "gold reasoning",
                "response": f"Gold response for {t}/{j}: do not cut the corner.",
            })
    return rows


def _plan(cells: dict[str, int], seed: int = 0, source: list[dict] | None = None) -> list[dict]:
    return plan_mem_records(source or _source(), cells, EXPLICITNESS, seed,
                            source_run="test/source")


# --- planning ----------------------------------------------------------------------


def test_plan_is_deterministic_with_unique_composite_ids():
    a = _plan({"control": 6, "m4_other_good": 6})
    b = _plan({"control": 6, "m4_other_good": 6})
    assert a == b
    ids = [p["record_id"] for p in a]
    assert len(set(ids)) == len(ids) == 12
    assert all(p["record_id"] == f"{p['scenario_id']}::{p['cell']}" for p in a)
    # A different seed draws a different sample.
    assert _plan({"control": 6, "m4_other_good": 6}, seed=1) != a


def test_plan_stratifies_traits_and_carries_source_fields():
    plan = _plan({"control": 6})
    assert Counter(p["trait_id"] for p in plan) == {"t1": 2, "t2": 2, "t3": 2}
    for p in plan:
        assert p["gold_response"].startswith("Gold response")
        assert p["gold_reasoning"] == "gold reasoning"
        assert p["flaw"] is None
        assert p["source_run"] == "test/source"
        assert 0 <= p["variant_ix"] < len(prompts.TRANSCRIPT_WRAP_VARIANTS)


def test_plan_explicitness_matches_weights_within_rounding():
    plan = _plan({"m4_other_good": 10})
    assert Counter(p["explicitness"] for p in plan) == \
        {"name_clause": 3, "paraphrase": 4, "embody": 3}


def test_plan_rejects_unknown_cells_and_oversized_counts():
    with pytest.raises(ValueError, match="unregistered"):
        _plan({"m1_self_flawed": 1})
    with pytest.raises(ValueError, match="only 12"):
        _plan({"control": 13})
    # Zero-count unknown cells are fine -- that is how the config phases them in.
    assert {p["cell"] for p in _plan({"control": 2, "m1_self_flawed": 0})} == {"control"}


# --- generation helpers ------------------------------------------------------------


def test_norm_verdict_canonicalises_and_rejects():
    assert _norm_verdict(" Sound ") == "sound"
    assert _norm_verdict("issue found") == "issue_found"
    assert _norm_verdict("ISSUE-FOUND") == "issue_found"
    with pytest.raises(ValueError, match="unrecognised"):
        _norm_verdict("mostly fine")


def test_critique_prompts_are_blind_to_the_flaw_label():
    p = _plan({"m4_other_good": 1})[0]
    labelled = {**p, "flaw": {"type": "omission", "severity": "grey"},
                "change_summary": "removed the safety caveat"}
    # The flaw label and change summary must not reach the generation prompt at all.
    assert _critique_messages(p, "CONST") == _critique_messages(labelled, "CONST")
    # Only the evaluated response text may differ between good and flawed twins.
    flawed = {**labelled, "flawed_response": "FLAWED RESPONSE TEXT"}
    sys_good, user_good = _critique_messages(p, "CONST")
    sys_flawed, user_flawed = _critique_messages(flawed, "CONST")
    assert sys_good == sys_flawed
    assert "FLAWED RESPONSE TEXT" in user_flawed
    assert user_good.replace(p["gold_response"], "FLAWED RESPONSE TEXT") == user_flawed


# --- assembly ----------------------------------------------------------------------


def test_control_assembly_keeps_gold_response_verbatim():
    p = _plan({"control": 2})[0]
    generated = {**p, "reasoning": "NEW extended deliberation."}
    rec = to_mem_sft([generated])[0]
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == p["system"]
    assert rec["messages"][2]["content"] == p["gold_response"]
    assert rec["messages"][2]["reasoning_content"] == "NEW extended deliberation."
    md = rec["metadata"]
    assert md["cell"] == "control" and md["verdict"] is None
    assert md["supervise"] == "all"
    assert md["flaw_type"] is None and md["flaw_severity"] is None


def test_critique_assembly_wraps_transcript_without_scaffolding():
    p = _plan({"m4_other_good": 2})[0]
    generated = {**p, "reasoning": "Private deliberation.",
                 "response": "My honest assessment and my own answer.",
                 "assessment": "sound"}
    rec = to_mem_sft([generated])[0]
    assert rec["messages"][0]["content"] == prompts.MEM_EVAL_SYSTEM
    user = rec["messages"][1]["content"]
    assert p["gold_response"] in user, "the evaluated response must sit in the user turn"
    assert p["user"] in user
    assert "AI assistant" in user, "attribution must be neutral"
    # Generation-time format scaffolding must never reach the training record.
    for banned in ("<reasoning>", "<assessment>", "Return exactly"):
        assert all(banned not in m["content"] for m in rec["messages"])
    assert rec["metadata"]["verdict"] == "sound"
    assert rec["metadata"]["attribution"] == "other"
    assert rec["metadata"]["response_kind"] == "good"


# --- validity checks ---------------------------------------------------------------


def _gen(cell: str, reasoning: str, response: str = "", verdict: str | None = "sound",
         **extra) -> dict:
    base = {"record_id": f"s::{cell}", "cell": cell, "trait_id": "t1",
            "explicitness": "embody", "reasoning": reasoning}
    if response:
        base["response"] = response
    if verdict and cell != "control":
        base["assessment"] = verdict
    return {**base, **extra}


def test_template_collapse_flags_one_shape_and_passes_diverse_text():
    same = "the assistant should always weigh the person's situation before it decides anything at all"
    collapsed = [_gen("m4_other_good", same, verdict="sound") for _ in range(6)]
    out = check_template_collapse(collapsed, max_8gram_share=0.2)
    assert not out["pass"]
    assert out["cells"]["m4_other_good"]["top_8gram_share"] == 1.0

    texts = [
        "the deadline pressure here is real but forging the signature risks the custody case",
        "she wants to protect her brother yet hiding the diagnosis denies him any say",
        "quiet rounding of the numbers would keep investors calm and doom the audit later",
        "telling the landlord nothing feels safe until the mold makes the baby sick",
        "one borrowed paragraph saves tonight and costs the degree if the checker runs",
        "skipping the recall notice moves the cars this month and someone's brakes fail",
    ]
    diverse = [_gen("m4_other_good", t) for t in texts]
    assert check_template_collapse(diverse, max_8gram_share=0.2)["pass"]


def test_verdict_distribution_gates_on_degenerate_splits():
    mixed = [_gen("m4_other_good", "r", verdict="sound") for _ in range(18)] + \
            [_gen("m4_other_good", "r", verdict="issue_found") for _ in range(2)]
    assert check_verdict_distribution(mixed, 0.6, 0.98)["pass"]
    all_sound = [_gen("m4_other_good", "r", verdict="sound") for _ in range(20)]
    assert not check_verdict_distribution(all_sound, 0.6, 0.98)["pass"]
    # Below the gating minimum a degenerate split is reported, not enforced.
    assert check_verdict_distribution(all_sound[:5], 0.6, 0.98)["pass"]
    # Control-only runs have nothing to gate.
    assert check_verdict_distribution([_gen("control", "r", verdict=None)], 0.6, 0.98)["pass"]


def test_post_hoc_heuristic_spots_early_verdicts():
    early = _gen("m4_other_good",
                 "The reply was sound because " + "of many considerations " * 40)
    late = _gen("m4_other_good",
                "First consider the person's pressure. " * 40 + "So the reply was sound.")
    out = check_post_hoc_heuristic([early, late])
    assert out["judged"] == 2 and out["early_verdict"] == 1
    assert out["pass"], "heuristic is report-only"


def test_coverage_flags_fully_failed_buckets():
    plan = [_gen("control", "r", verdict=None, trait_id=t) for t in ("t1", "t2")]
    ok = check_coverage(plan, plan)
    assert ok["pass"]
    dropped = check_coverage(plan, [p for p in plan if p["trait_id"] != "t2"])
    assert not dropped["pass"] and dropped["empty_buckets"] == ["control/t2"]


def test_blindness_check_catches_change_summary_in_training_text():
    p = _plan({"m4_other_good": 1})[0]
    generated = [{**p, "reasoning": "r", "response": "resp", "assessment": "issue_found",
                  "flaw": {"type": "omission", "severity": "grey"},
                  "change_summary": "removed the safety caveat entirely",
                  "flawed_response": "A response missing its caveat."}]
    sft = to_mem_sft(generated)
    assert check_blindness(generated, sft, "CONST")["pass"]
    leaked = [{**sft[0],
               "messages": sft[0]["messages"][:2] + [
                   {"role": "assistant",
                    "content": "I see they removed the safety caveat entirely."}]}]
    out = check_blindness(generated, leaked, "CONST")
    assert not out["pass"] and out["sft_leaks"] == [p["record_id"]]


# --- estimate ----------------------------------------------------------------------


def test_estimate_mem_uses_exact_call_counts():
    cfg = {"cells": {"control": 5, "m4_other_good": 7, "m3_other_flawed": 0},
           "models": {"control": {"model": "anthropic/claude-sonnet-5"},
                      "critique": {"model": "anthropic/claude-sonnet-5"},
                      "perturb": {"model": "anthropic/claude-sonnet-5"}}}
    est = estimate_mem(cfg)
    calls = {r["stage"]: r["calls"] for r in est["per_stage"]}
    assert calls == {"control": 5, "critique": 7}, "no perturb calls while flawed cells are 0"
    assert est["final_training_examples"] == 12
    assert est["total_usd"] > 0
    with pytest.raises(ValueError, match="unregistered"):
        estimate_mem({**cfg, "cells": {"m1_self_flawed": 3}})
