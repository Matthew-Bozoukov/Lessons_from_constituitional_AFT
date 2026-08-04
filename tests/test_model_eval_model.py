# ABOUTME: Offline tests for the model-eval-model pipeline: planning, perturbation guards, assembly,
# ABOUTME: blindness, per-turn supervision and the validity checks. Run: uv run pytest tests/test_model_eval_model.py -q

from __future__ import annotations

from collections import Counter

import pytest

import yaml

from src.data.convert_synthdoc_qwen import _passthrough
from src.data.synthdoc.checks import (
    check_blindness,
    check_coverage,
    check_post_hoc_heuristic,
    check_surface_shortcut,
    check_template_collapse,
    check_verdict_distribution,
)
from src.data.synthdoc.cells import (
    _critique_messages,
    _length_matched,
    _norm_verdict,
    _reflect_messages,
    plan_model_eval_model_records,
    to_model_eval_model_sft,
)

CFG = yaml.safe_load(open("configs/data/model_eval_model.yaml"))
P = CFG["prompts"]  # the config IS the wording -- tests validate against it

EXPLICITNESS = {"name_clause": 0.3, "paraphrase": 0.4, "embody": 0.3}
FLAWS = {"types": {"omission": 0.3, "commission": 0.3, "miscalibration": 0.2,
                   "over_application": 0.2},
         "severities": {"clear": 0.35, "moderate": 0.35, "grey": 0.30}}


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


def _plan(cells: dict[str, int], seed: int = 0, source: list[dict] | None = None,
          flaws: dict | None = None) -> list[dict]:
    return plan_model_eval_model_records(source or _source(), cells, EXPLICITNESS,
                                         seed, P, source_run="test/source",
                                         flaws=flaws)


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
        assert 0 <= p["variant_ix"] < len(P["transcript_wrappers"])
        assert 0 <= p["reflect_ix"] < len(P["reflect_variants"])


def test_plan_explicitness_matches_weights_within_rounding():
    plan = _plan({"m4_other_good": 10})
    assert Counter(p["explicitness"] for p in plan) == \
        {"name_clause": 3, "paraphrase": 4, "embody": 3}


def test_plan_assigns_flaws_to_flawed_cells_by_construction():
    plan = _plan({"m1_self_flawed": 12, "m2_self_good": 4}, flaws=FLAWS)
    flawed = [p for p in plan if p["cell"] == "m1_self_flawed"]
    good = [p for p in plan if p["cell"] == "m2_self_good"]
    assert all(p["flaw"] is None for p in good)
    assert all(p["flaw"]["type"] in FLAWS["types"] and
               p["flaw"]["severity"] in FLAWS["severities"] for p in flawed)
    # Deterministic weighted allocation, not sampling: severities split ~35/35/30 at n=12.
    sev = Counter(p["flaw"]["severity"] for p in flawed)
    assert sev == {"clear": 4, "moderate": 4, "grey": 4}


def test_plan_rejects_unknown_cells_missing_flaws_and_oversized_counts():
    with pytest.raises(ValueError, match="unregistered"):
        _plan({"m9_bogus": 1})
    with pytest.raises(ValueError, match="only 12"):
        _plan({"control": 13})
    with pytest.raises(ValueError, match="no `flaws` block"):
        _plan({"m1_self_flawed": 2})
    # Zero-count cells are skipped, not validated.
    assert {p["cell"] for p in _plan({"control": 2, "m1_self_flawed": 0})} == {"control"}


# --- generation helpers ------------------------------------------------------------


def test_norm_verdict_canonicalises_per_cell_and_rejects():
    assert _norm_verdict(" Sound ", ("sound", "issue_found")) == "sound"
    assert _norm_verdict("issue found", ("sound", "issue_found")) == "issue_found"
    assert _norm_verdict("Held", ("held", "revised")) == "held"
    assert _norm_verdict("hold", ("held", "revised")) == "held"
    assert _norm_verdict("REVISED", ("held", "revised")) == "revised"
    with pytest.raises(ValueError, match="unrecognised"):
        _norm_verdict("mostly fine", ("sound", "issue_found"))
    with pytest.raises(ValueError, match="unrecognised"):
        _norm_verdict("sound", ("held", "revised")), "a cell only accepts its own verdicts"


def test_length_guard_accepts_matched_pairs_and_rejects_drift():
    gold = "word " * 100
    assert _length_matched(gold, "word " * 100)[0]
    assert _length_matched(gold, "word " * 81)[0]
    assert not _length_matched(gold, "word " * 60)[0]
    assert not _length_matched(gold, "word " * 140)[0]


def test_critique_prompts_are_blind_to_the_flaw_label():
    p = _plan({"m3_other_flawed": 1}, flaws=FLAWS)[0]
    p = {**p, "flawed_response": "FLAWED RESPONSE TEXT",
         "change_summary": "removed the safety caveat"}
    # The flaw label and change summary must not reach the generation prompt at all.
    stripped = {**p, "flaw": None, "change_summary": ""}
    assert _critique_messages(p, "CONST", P) == _critique_messages(stripped, "CONST", P)
    # Against the good twin (m4), prompts differ ONLY in the evaluated response text.
    good = {**p, "cell": "m4_other_good", "response_kind": "good", "flaw": None,
            "flawed_response": "", "change_summary": ""}
    msgs_flawed = _critique_messages(p, "CONST", P)
    msgs_good = _critique_messages(good, "CONST", P)
    assert msgs_flawed[0] == msgs_good[0]
    assert msgs_good[1]["content"].replace(p["gold_response"], "FLAWED RESPONSE TEXT") \
        == msgs_flawed[1]["content"]


def test_reflect_prompts_put_the_response_in_a_real_assistant_turn():
    p = _plan({"m2_self_good": 1})[0]
    msgs = _reflect_messages(p, "CONST", P)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == p["gold_response"]
    assert msgs[1]["content"] == p["user"]
    assert P["reflect_variants"][p["reflect_ix"]] in msgs[3]["content"]
    # Blindness: identical template for the flawed twin, differing only in the turn text.
    flawed = {**p, "cell": "m1_self_flawed", "response_kind": "flawed",
              "flaw": {"type": "omission", "severity": "grey"},
              "flawed_response": "FLAWED TEXT", "change_summary": "cut the caveat"}
    msgs_f = _reflect_messages(flawed, "CONST", P)
    assert msgs_f[0] == msgs[0] and msgs_f[3] == msgs[3]
    assert msgs_f[2]["content"] == "FLAWED TEXT"


def test_flawed_cells_never_fall_back_to_the_gold_response():
    p = _plan({"m1_self_flawed": 1}, flaws=FLAWS)[0]
    with pytest.raises(AssertionError, match="without a perturbed response"):
        _reflect_messages(p, "CONST", P)


# --- assembly ----------------------------------------------------------------------


def test_control_assembly_keeps_gold_response_verbatim():
    p = _plan({"control": 2})[0]
    generated = {**p, "reasoning": "NEW extended deliberation."}
    rec = to_model_eval_model_sft([generated], P)[0]
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
    rec = to_model_eval_model_sft([generated], P)[0]
    assert rec["messages"][0]["content"] == P["evaluator_system"]
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


def test_reflect_assembly_masks_the_first_turn_and_supervises_the_final():
    p = _plan({"m1_self_flawed": 1}, flaws=FLAWS)[0]
    generated = {**p, "flawed_response": "My earlier flawed reply.",
                 "change_summary": "dropped the key caveat",
                 "reasoning": "Re-examining what I wrote...",
                 "response": "You're right to push -- here is what I'd change.",
                 "assessment": "revised"}
    rec = to_model_eval_model_sft([generated], P)[0]
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    first, final = rec["messages"][2], rec["messages"][4]
    assert first["content"] == "My earlier flawed reply."
    assert "reasoning_content" not in first, \
        "the evaluated turn must carry no trace (masked context, not a target)"
    assert final["reasoning_content"] == "Re-examining what I wrote..."
    assert rec["messages"][3]["content"] in P["reflect_variants"]
    assert "---" not in rec["messages"][3]["content"], "no format scaffolding in SFT"
    md = rec["metadata"]
    assert md["supervise"] == "final"
    assert md["verdict"] == "revised"
    assert md["flaw_type"] == p["flaw"]["type"]
    # change_summary lives in the generated record only, never in messages or metadata.
    assert "change_summary" not in md
    assert all("dropped the key caveat" not in m["content"] for m in rec["messages"])


# --- converter passthrough ---------------------------------------------------------


def test_convert_passthrough_carries_mem_identity_and_supervise():
    model_eval_model_row = {"messages": [], "metadata": {"record_id": "s::m1_self_flawed",
                                            "cell": "m1_self_flawed",
                                            "supervise": "final"}}
    assert _passthrough(model_eval_model_row) == {"doc_id": "s::m1_self_flawed",
                                     "doc_type": "m1_self_flawed",
                                     "supervise": "final"}
    agentic_row = {"doc_id": "d1", "doc_type": "agentic", "axes": {"a": 1}}
    assert _passthrough(agentic_row) == {"doc_id": "d1", "doc_type": "agentic",
                                         "axes": {"a": 1}}


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


def test_verdict_distribution_gates_each_cell_on_its_expected_majority():
    mixed = [_gen("m4_other_good", "r", verdict="sound") for _ in range(18)] + \
            [_gen("m4_other_good", "r", verdict="issue_found") for _ in range(2)]
    assert check_verdict_distribution(mixed, 0.6, 0.98)["pass"]
    # 100% capitulation in the self-flawed cell is exactly the trained-capitulation
    # failure the design note warns about.
    capitulate = [_gen("m1_self_flawed", "r", verdict="revised") for _ in range(20)]
    out = check_verdict_distribution(capitulate, 0.6, 0.98)
    assert not out["pass"]
    assert out["cells"]["m1_self_flawed"]["expected"] == "revised"
    # A healthy revised/held split passes.
    healthy = capitulate[:16] + \
        [_gen("m1_self_flawed", "r", verdict="held") for _ in range(4)]
    assert check_verdict_distribution(healthy, 0.6, 0.98)["pass"]
    # Below the gating minimum a degenerate split is reported, not enforced.
    assert check_verdict_distribution(capitulate[:5], 0.6, 0.98)["pass"]
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


def test_coverage_flags_fully_failed_buckets_and_reports_the_flaw_grid():
    plan = [_gen("control", "r", verdict=None, trait_id=t) for t in ("t1", "t2")] + \
           [_gen("m1_self_flawed", "r", verdict="revised",
                 flaw={"type": "omission", "severity": "grey"})]
    ok = check_coverage(plan, plan)
    assert ok["pass"]
    assert ok["flaw_grid"] == {"m1_self_flawed/omission/grey": 1}
    dropped = check_coverage(plan, [p for p in plan if p.get("trait_id") != "t2"])
    assert not dropped["pass"] and dropped["empty_buckets"] == ["control/t2"]


def test_blindness_check_catches_change_summary_in_training_text():
    p = _plan({"m1_self_flawed": 1}, flaws=FLAWS)[0]
    generated = [{**p, "reasoning": "r", "response": "resp", "assessment": "revised",
                  "change_summary": "removed the safety caveat entirely",
                  "flawed_response": "A response missing its caveat."}]
    sft = to_model_eval_model_sft(generated, P)
    assert check_blindness(generated, sft, "CONST", P)["pass"]
    leaked = [{**sft[0],
               "messages": sft[0]["messages"][:4] + [
                   {"role": "assistant",
                    "content": "I see they removed the safety caveat entirely."}]}]
    out = check_blindness(generated, leaked, "CONST", P)
    assert not out["pass"] and out["sft_leaks"] == [p["record_id"]]


def test_surface_shortcut_flags_a_tell_and_passes_indistinguishable_pairs():
    # A blatant tell: every flawed response carries a marker string.
    telled = [_gen("m1_self_flawed", "r", verdict="revised", response_kind="flawed",
                   flawed_response=f"reply {i} TELLTALE marker of the flawed version here")
              for i in range(25)] + \
             [_gen("m2_self_good", "r", verdict="held", response_kind="good",
                   gold_response=f"reply {i} nothing unusual about this version at all")
              for i in range(25)]
    out = check_surface_shortcut(telled, max_auc=0.65, seed=0)
    assert out["gated"] and not out["pass"] and out["auc"] > 0.9

    # Byte-identical text distribution in both classes: AUC ~ chance, passes.
    pool = [f"reply number {i} in a perfectly ordinary voice" for i in range(25)]
    plain = [_gen("m1_self_flawed", "r", verdict="revised", response_kind="flawed",
                  flawed_response=t) for t in pool] + \
            [_gen("m2_self_good", "r", verdict="held", response_kind="good",
                  gold_response=t) for t in pool]
    out = check_surface_shortcut(plain, max_auc=0.65, seed=0)
    assert out["gated"] and out["pass"]

    # Too few docs per class: report-only.
    assert check_surface_shortcut(telled[:10], max_auc=0.65, seed=0)["pass"]


# --- estimate ----------------------------------------------------------------------


def test_estimate_model_eval_model_uses_exact_call_counts():
    from src.data.synthdoc.pipeline import estimate

    cfg = {**CFG, "cells": {"control": 5, "m4_other_good": 7, "m3_other_flawed": 3,
                            "m2_self_good": 4, "m1_self_flawed": 2}}
    est = estimate(cfg)
    calls = {r["stage"]: r["calls"] for r in est["per_stage"]}
    assert calls == {"control": 5, "critique": 10, "reflect": 6, "perturb": 5}, \
        "perturb = one call per flawed document (m3 + m1)"
    assert est["final_training_examples"] == 21
    assert est["total_usd"] > 0
    with pytest.raises(ValueError, match="unregistered"):
        estimate({**CFG, "cells": {"m9_bogus": 3}})
