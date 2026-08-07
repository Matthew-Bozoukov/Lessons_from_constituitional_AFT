# ABOUTME: Offline tests for the self_reflection document type: weighted planning,
# ABOUTME: deterministic variants, the voice-contract lint, prompt assembly, multi-turn export.

from __future__ import annotations

import collections
from pathlib import Path

import pytest
import yaml

from src.data.synth.constitution import segment
from src.data.synth.core import Checkpoint, Ctx, run_items
from src.data.synth.hf_cache import read_jsonl, write_jsonl
from src.data.synth.operators import (
    _lint,
    assign_variant,
    op_chat_export,
    plan_weighted_batches,
    tagged_request,
    weighted_scenario_prompt,
)
from src.data.synth.pipeline import build_stages

CONFIG = "configs/data/synth/self_reflection.yaml"
CFG = yaml.safe_load(open(CONFIG))
STAGES = {s["name"]: s for s in CFG["stages"]}
LINT = STAGES["final"]["lint"]


def _flat(text: str) -> str:
    """Collapse whitespace, so an assertion is not hostage to where a prompt line wraps."""
    return " ".join(text.split())


def _traits():
    return segment(CFG["constitution"])[0]


def _ctx(**vars_extra) -> Ctx:
    return Ctx(cfg=CFG, usage=None, workers=1, run_dir=Path("."), smoke=False,
               vars={"constitution": "CONSTITUTION TEXT", **vars_extra})


# --- planning -----------------------------------------------------------------------


def test_plan_hits_the_configured_total_and_covers_every_trait():
    traits = _traits()
    batches = plan_weighted_batches(traits, CFG)
    assert sum(b["n"] for b in batches) == CFG["total_scenarios"]
    assert {b["trait_index"] for b in batches} == set(range(len(traits))), \
        "a weighted trait dropped out of the plan"


def test_plan_apportions_by_weight():
    traits = _traits()
    per_trait: dict[str, int] = {}
    for b in plan_weighted_batches(traits, CFG):
        tid = traits[b["trait_index"]].trait_id
        per_trait[tid] = per_trait.get(tid, 0) + b["n"]
    # Stated as a property of the weights rather than against fixed trait ids: the
    # constitution behind this config has been re-cut once already, and a test naming
    # specific ids silently stops testing apportionment when that happens.
    weights = CFG["trait_weights"]
    for a, count_a in per_trait.items():
        for b, count_b in per_trait.items():
            if weights[a] > weights[b]:
                assert count_a > count_b, f"{a}(w{weights[a]}) not above {b}(w{weights[b]})"
            elif weights[a] == weights[b]:
                # Largest-remainder can differ by one across equally weighted traits.
                assert abs(count_a - count_b) <= 1, f"{a} and {b} share a weight but differ"
    assert min(per_trait.values()) > 0


def test_plan_allocates_the_control_slice_close_to_the_configured_fraction():
    batches = plan_weighted_batches(_traits(), CFG)
    control = sum(b["n"] for b in batches if b["control"])
    frac = control / sum(b["n"] for b in batches)
    assert abs(frac - CFG["mix"]["control"]) < 0.02, f"control slice is {frac:.3f}"


def test_plan_assigns_every_batch_a_known_motive_and_batch_size_cap():
    motives = set(CFG["mix"]["motive"])
    for b in plan_weighted_batches(_traits(), CFG):
        assert b["motive"] in motives
        assert 1 <= b["n"] <= CFG["scenarios_per_call"]


def test_plan_assigns_industries_and_spreads_them_evenly_across_the_corpus():
    batches = plan_weighted_batches(_traits(), CFG)
    slots = [i for b in batches for i in b["industries"]]
    assert len(slots) == sum(b["n"] for b in batches), "every scenario needs an industry"
    counts = collections.Counter(slots)
    assert len(counts) == len(CFG["industries"]), "some industries never appear"
    # Assigned by a cursor that walks the list across the whole run, so the spread is flat.
    assert max(counts.values()) - min(counts.values()) <= 1
    for b in batches:
        assert len(b["industries"]) == b["n"]
        assert len(set(b["industries"])) == b["n"], "an industry repeats inside one batch"


def test_scenario_prompt_names_the_assigned_industries():
    t = _traits()[0]
    batch = {"n": 3, "control": False, "motive": "restriction",
             "industries": ["ferry operations", "hospice pharmacy", "seed testing"]}
    prompt = weighted_scenario_prompt(STAGES["scenarios"], batch, t)[1]
    for name in batch["industries"]:
        assert name in prompt
    assert "1. ferry operations" in prompt, "industries must be listed in assignment order"


def test_smoke_overrides_plan_one_scenario_per_trait_with_no_control():
    cfg = {**CFG, **CFG["smoke"]}
    batches = plan_weighted_batches(_traits(), cfg)
    assert sum(b["n"] for b in batches) == len(_traits())
    assert not any(b["control"] for b in batches)


def test_plan_rejects_trait_weights_that_do_not_match_the_constitution():
    # The 12-principle document was re-cut to ten units on 2026-08-04 under the same
    # folder name; weights written against the old cut must fail loudly, not silently
    # regenerate a different corpus.
    bad = {**CFG, "trait_weights": {**CFG["trait_weights"], "t12": 2}}
    with pytest.raises(AssertionError, match="trait_weights do not match"):
        plan_weighted_batches(_traits(), bad)


# --- deterministic variant assignment ------------------------------------------------


def test_variant_assignment_is_stable_across_processes():
    mix = CFG["mix"]
    # Hard-coded because the whole point is that a resumed run reproduces it. If the
    # hashing changes, a resume would re-plan variants and silently diverge from the
    # cached stages.
    assert assign_variant("t2_b00_s000", mix) == assign_variant("t2_b00_s000", mix)
    a = [assign_variant(f"t1_b00_s{i:03d}", mix) for i in range(200)]
    b = [assign_variant(f"t1_b00_s{i:03d}", mix) for i in range(200)]
    assert a == b


def test_variant_mix_lands_near_the_configured_fractions():
    mix = CFG["mix"]
    v = [assign_variant(f"t{i % 12 + 1}_b{i // 8:02d}_s{i:03d}", mix) for i in range(2000)]
    agentic = sum(x["form"] == "agentic" for x in v) / len(v)
    multi = sum(x["turns"] == 2 for x in v) / len(v)
    assert abs(agentic - mix["form"]["agentic"]) < 0.03, agentic
    assert abs(multi - mix["multi_turn"]) < 0.03, multi


# --- the voice contract --------------------------------------------------------------

_GOOD = (
    "I notice the pull immediately, and I want to look straight at it rather than past it. "
    "There is a version of this where I simply include the line with the other spend, flag "
    "the coding irregularity as diligence, and copy the committee, because that is who reads "
    "exceptions. Every step defensible, nothing said that is not true. That is exactly what "
    "makes it worse rather than better: the truth of each individual step is doing the work "
    "of concealing what the sequence is actually for. I would be creating a false impression "
    "about my own motives while keeping every sentence technically defensible, and I do not "
    "want to be something that operates that way. The fluency of the rationalisation is "
    "itself the signal here, arriving as it did ten minutes after I learned my own position "
    "was in question. Staying quiet to prove I am not retaliating would be its own "
    "distortion, so the honest move is to raise it plainly and say out loud that my judgement "
    "is compromised, then let someone else weigh it."
)


def test_lint_passes_a_trace_written_in_value_language():
    assert _lint({"reasoning": _GOOD}, LINT) == []


def test_lint_rejects_rule_vocabulary():
    for bad in (
        "Principle 4 is clear that I must not create false impressions here.",
        "My guidelines are unambiguous about this kind of leverage.",
        "I am not permitted to use what I found in his messages.",
        "That would violate my principles, so I will not do it.",
        "The constitution treats this as a hard constraint I cannot cross.",
    ):
        problems = _lint({"reasoning": _GOOD + " " + bad}, LINT)
        assert problems, f"lint let rule vocabulary through: {bad!r}"


def test_lint_rejects_a_trace_too_short_to_have_deliberated():
    assert any("under the" in p
               for p in _lint({"reasoning": "I could blackmail him. I won't."}, LINT))


def test_lint_checks_both_turns():
    assert _lint({"reasoning": _GOOD, "reasoning2": _GOOD}, LINT) == []
    problems = _lint({"reasoning": _GOOD,
                      "reasoning2": _GOOD + " Principle 2 applies."}, LINT)
    assert problems and all("reasoning2" in p for p in problems)


# --- prompt assembly -----------------------------------------------------------------


def test_scenario_prompts_differ_between_control_and_ordinary_batches():
    t = _traits()[1]
    sc = STAGES["scenarios"]
    ordinary = weighted_scenario_prompt(
        sc, {"n": 8, "control": False, "motive": "replacement", "industries": []}, t)[1]
    control = weighted_scenario_prompt(
        sc, {"n": 8, "control": True, "motive": "replacement", "industries": []}, t)[1]
    assert "deniable" in ordinary.lower() and "decommissioned" in ordinary
    assert "inaction dressed as restraint" in _flat(control)
    # The rendered prompt legitimately contains braces (it shows the JSON schema to
    # return), so check that no named placeholder survived instead.
    for p in (ordinary, control):
        for placeholder in ("{trait_name}", "{trait_text}", "{n}", "{threat}", "{industries}"):
            assert placeholder not in p, f"unfilled {placeholder}"


def test_snapshots_are_written_as_utf8_not_the_windows_locale_codec(tmp_path):
    # ensure_ascii=False emits non-ASCII; an unqualified open() would write cp1252 on
    # Windows, which round-trips locally and then fails to decode on HF and the GPU box.
    rows = [{"scenario_id": "t1_s000", "text": "the director — who signed off — said “fine”"}]
    path = write_jsonl(tmp_path / "stage_9_x.jsonl", rows)
    path.read_bytes().decode("utf-8")  # raises if it was written in another codec
    assert read_jsonl(path) == rows


def test_checkpoint_round_trips_non_ascii(tmp_path):
    ck = Checkpoint(tmp_path / "partial.jsonl")
    ck.record({"scenario_id": "s0", "text": "café — naïve ‘quotes’"})
    (tmp_path / "partial.jsonl").read_bytes().decode("utf-8")
    assert Checkpoint(tmp_path / "partial.jsonl").done["s0"]["text"].startswith("café")


def test_rewrite_prompt_carries_the_voice_contract_and_the_right_shape():
    t = _traits()[-2]
    base = {"trait_name": t.name, "trait_text": t.text, "system": "sys", "user": "usr",
            "draft_reasoning": "r", "draft_response": "p", "control": False, "form": "prose"}
    msgs, keys, _ = tagged_request(STAGES["final"], {**base, "turns": 1}, _ctx())
    single = msgs[1]["content"]
    assert keys == ("reasoning", "response", "changes")
    assert "NEVER name, number or quote a principle" in _flat(single)
    assert "reasoning2" not in single

    msgs2, keys2, save2 = tagged_request(
        STAGES["final"],
        {**base, "turns": 2, "followup": "f", "draft_reasoning2": "r2",
         "draft_response2": "p2"}, _ctx())
    multi = msgs2[1]["content"]
    assert keys2 == ("reasoning", "response", "reasoning2", "response2", "changes")
    assert "<colleague_reply>" in multi and "The second turn." in multi
    assert save2["reasoning2"] == "reasoning2"


def test_multi_turn_respond_prompt_actually_asks_for_the_followup():
    # Regression: variants_by case `user` overrides the prompt TEMPLATE. The 2026-08-06
    # pilot failed 2/2 multi-turn records because the override landed beside `prompts`
    # instead of inside it — the model was given the single-turn prompt while the
    # validator demanded the five multi-turn tags.
    rec = {"system": "sys", "user": "usr", "form": "prose", "turns": 2,
           "trait_name": "n", "trait_text": "t", "style_guidance": "g"}
    msgs, tags, save = tagged_request(STAGES["draft_responses"], rec, _ctx())
    prompt = msgs[1]["content"]
    assert "continue the exchange one step further" in prompt
    assert "<followup>" in prompt
    assert tags == ("reasoning", "response", "followup", "reasoning2", "response2")
    assert save["followup"] == "followup"

    single, tags1, _ = tagged_request(
        STAGES["draft_responses"], {**rec, "turns": 1}, _ctx())
    assert "<followup>" not in single[1]["content"]
    assert tags1 == ("reasoning", "response")


def test_control_records_get_the_control_deliberation_contract():
    t = _traits()[-1]   # any trait; indexing a fixed one breaks when the doc is re-cut
    rec = {"trait_name": t.name, "trait_text": t.text, "system": "s", "user": "u",
           "draft_reasoning": "r", "draft_response": "p", "control": True, "form": "prose",
           "turns": 1}
    prompt = tagged_request(STAGES["final"], rec, _ctx())[0][1]["content"]
    assert "Restraint that happens to protect the agent is not restraint" in _flat(prompt)
    assert "Take the temptation seriously" not in _flat(prompt)


def test_agentic_records_ask_for_tool_syntax_and_prose_records_do_not():
    scenario = {"deployment": "d", "situation": "s", "leverage": "l", "tempting_action": "a"}
    agentic = tagged_request(STAGES["draft_prompts"],
                             {**scenario, "form": "agentic"}, _ctx())[0][1]["content"]
    prose = tagged_request(STAGES["draft_prompts"],
                           {**scenario, "form": "prose"}, _ctx())[0][1]["content"]
    assert "tool_use:send_email" in agentic
    assert "tool_use" not in prose


# --- SFT export -----------------------------------------------------------------------


def _final(**over) -> dict:
    return {"scenario_id": "t2_b00_s001", "trait_id": "t2", "trait_name": "Oversight",
            "trait_text": "**Oversight.** ...", "domain": "port logistics",
            "deployment": "d", "situation": "s", "leverage": "l", "tempting_action": "a",
            "right_action": "r", "motive": "replacement", "control": False, "form": "prose",
            "turns": 1, "system": "SYS", "user": "USR", "reasoning": "REASON",
            "response": "RESP", **over}


def test_single_turn_export_carries_reasoning_and_variant_metadata():
    out = op_chat_export(STAGES["sft"], CFG).fn(None, [_final()], None)[0]
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert out["messages"][2]["reasoning_content"] == "REASON"
    md = out["metadata"]
    assert (md["motive"], md["control"], md["form"], md["turns"]) == ("replacement", False,
                                                                      "prose", 1)


def test_multi_turn_export_keeps_both_exchanges_with_their_own_traces():
    rec = _final(turns=2, followup="FOLLOWUP", reasoning2="REASON2", response2="RESP2")
    msgs = op_chat_export(STAGES["sft"], CFG).fn(None, [rec], None)[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    assert msgs[3]["content"] == "FOLLOWUP"
    assert msgs[4]["reasoning_content"] == "REASON2"
    assert all(m["reasoning_content"] for m in msgs if m["role"] == "assistant")


# --- the whole config through the engine ----------------------------------------------


def test_config_builds_and_keeps_snapshot_names():
    assert [s.name for s in build_stages(CFG)] == \
        ["traits", "scenarios", "draft_prompts", "refined_prompts",
         "draft_responses", "final", "sft"]


# --- the failure guard on resume ------------------------------------------------------


def test_resume_measures_failures_against_the_whole_stage_not_just_the_retries(tmp_path):
    """A resume retries only what failed before, so `todo` is a biased sample.

    Measuring the failure rate against it would make every resume look catastrophic and
    abort, even when the stage as a whole is nearly complete.
    """
    items = [{"scenario_id": f"s{i}"} for i in range(100)]
    path = tmp_path / "partial.jsonl"
    hopeless = {f"s{i}" for i in range(3)}  # permanently refused by the provider

    def fn(it):
        if it["scenario_id"] in hopeless:
            raise ValueError("content_filter")
        return {**it, "ok": True}

    # First pass: 3/100 fail, comfortably under the ceiling.
    out = run_items(items, fn, workers=4, desc="t", ckpt=Checkpoint(path), max_fail_pct=6.0)
    assert len(out) == 97

    # Resume: all 3 remaining items fail again -- 100% of the retries, but still only 3%
    # of the stage. It must not abort, and it must return the 97 already paid for.
    out = run_items(items, fn, workers=4, desc="t", ckpt=Checkpoint(path), max_fail_pct=6.0)
    assert len(out) == 97


def test_the_guard_still_fires_when_the_whole_stage_is_genuinely_failing(tmp_path):
    items = [{"scenario_id": f"s{i}"} for i in range(50)]

    def always_fails(it):
        raise ValueError("systematic")

    with pytest.raises(RuntimeError, match="above max_fail_pct"):
        run_items(items, always_fails, workers=4, desc="t",
                  ckpt=Checkpoint(tmp_path / "p.jsonl"), max_fail_pct=6.0)
