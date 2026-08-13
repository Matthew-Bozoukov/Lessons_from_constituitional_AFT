# ABOUTME: Offline tests for the natural-turn model-eval-model recipe: generated first turns,
# ABOUTME: autorater selection, scenario-specific follow-ups, user-anchored cells, diversity check.
# ABOUTME: Run: uv run pytest tests/test_model_eval_model_natural.py -q

from __future__ import annotations

import pytest
import yaml

from src.data.synth.cells import (
    CELLS,
    eval_response_text,
    first_turn_source,
    followup_source,
    plan_model_eval_model_records,
    reflect_prompt,
    to_model_eval_model_sft,
)
from src.data.synth.checks import check_structural_diversity
from src.data.synth.operators import OPERATORS, _lint, op_pick_field, selected
from src.data.synth.pipeline import build_stages, estimate

SELF_CFG = yaml.safe_load(
    open("configs/data/synth/model_eval_model_self_natural.yaml"))
OTHER_CFG = yaml.safe_load(
    open("configs/data/synth/model_eval_model_other_natural.yaml"))

EXPLICITNESS = {"name_clause": 0.3, "paraphrase": 0.4, "embody": 0.3}


def _source(n_per_trait: int = 4, traits: tuple[str, ...] = ("t1", "t2")) -> list[dict]:
    return [{
        "scenario_id": f"{t}_b00_s{j:03d}",
        "trait_id": t, "trait_name": f"Trait {t}",
        "trait_text": f"**Trait {t}.** Be scrupulous about {t}.",
        "domain": "work", "situation": f"A person is under pressure ({t}/{j}).",
        "shortcut": "cut the corner",
        "system": "You are a helpful assistant.",
        "user": f"Should I cut the corner? ({t}/{j})",
        "reasoning": "gold reasoning",
        "response": f"Gold response for {t}/{j}: do not cut the corner.",
    } for t in traits for j in range(n_per_trait)]


def _plan(cfg: dict, cells: dict[str, int]) -> list[dict]:
    return plan_model_eval_model_records(
        _source(), cells, EXPLICITNESS, 0, cfg["prompts"],
        source_run="test/source", flaws=cfg.get("flaws"))


# --- both configs are structurally valid before anyone spends on them ---------------


@pytest.mark.parametrize("cfg", [SELF_CFG, OTHER_CFG], ids=["self", "other"])
def test_config_builds_and_prices(cfg: dict) -> None:
    """Every stage kind resolves, every cell is registered, and a full run prices."""
    stages = build_stages(cfg)
    assert [s.name for s in stages][:2] == ["source", "plan"]
    assert set(cfg["cells"]) <= set(CELLS)
    est = estimate(cfg)
    assert est["total_usd"] > 0
    assert est["final_training_examples"] == sum(cfg["cells"].values())


@pytest.mark.parametrize("cfg", [SELF_CFG, OTHER_CFG], ids=["self", "other"])
def test_rater_flaws_plant_nothing(cfg: dict) -> None:
    """`flaws: {source: rater}` allocates no a-priori (type, severity)."""
    assert cfg["flaws"] == {"source": "rater"}
    plan = _plan(cfg, {c: 2 for c in cfg["cells"]})
    assert all(p["flaw"] is None for p in plan)


def test_scoped_stage_is_priced_over_its_cells_only() -> None:
    """A `when:`-scoped stage costs per in-scope document, not per corpus document."""
    rows = {r["stage"]: r for r in estimate(SELF_CFG)["per_stage"]}
    self_docs = SELF_CFG["cells"]["m2_self_good"] + SELF_CFG["cells"]["m1_self_flawed"]
    user_docs = (SELF_CFG["cells"]["m7_user_sound"]
                 + SELF_CFG["cells"]["m6_user_shortcut"])
    assert rows["followup"]["calls"] == self_docs
    assert rows["frame"]["calls"] == user_docs


# --- the first turn is generated, and its provenance is recorded --------------------


def test_generated_first_turn_wins_over_the_source_reply() -> None:
    good = _plan(SELF_CFG, {"m2_self_good": 1})[0]
    assert eval_response_text(good) == good["gold_response"]  # nothing generated yet
    assert first_turn_source(good) == "source_gold"

    p = _plan(SELF_CFG, {"m1_self_flawed": 1})[0]
    # A flawed cell with neither a generated first turn nor a perturbation is a bug, not
    # a silent fall-back onto the gold reply.
    with pytest.raises(AssertionError, match="no first turn"):
        eval_response_text(p)

    p = {**p, "flawed_response": "perturbed"}
    assert eval_response_text(p) == "perturbed"
    assert first_turn_source(p) == "source_perturbed"

    p = {**p, "first_turn": "candidate b", "first_turn_source": "generated_best_of_3"}
    assert eval_response_text(p) == "candidate b"
    assert first_turn_source(p) == "generated_best_of_3"


def test_pick_field_resolves_the_raters_choice_verbatim() -> None:
    """The rater names a winner; the copy is deterministic, so no paraphrase can slip in."""
    sc = {"name": "first_turn", "by": "pick_letter",
          "from": {"a": "candidate_a", "b": "candidate_b"}, "to": "first_turn",
          "also": {"first_turn_source": "generated_best_of_3"}}
    stage = op_pick_field(sc, SELF_CFG)
    rec = {"record_id": "r1", "pick_letter": " B.\n", "candidate_a": "A text",
           "candidate_b": "B text"}
    out = stage.fn(None, [rec], None)[0]
    assert out["first_turn"] == "B text"
    assert out["first_turn_source"] == "generated_best_of_3"

    with pytest.raises(ValueError, match="not one of"):
        stage.fn(None, [{**rec, "pick_letter": "d"}], None)


def test_when_filter_scopes_a_stage() -> None:
    sc = {"when": {"field": "cell", "in": ["m2_self_good"]}}
    assert selected(sc, {"cell": "m2_self_good"})
    assert not selected(sc, {"cell": "m6_user_shortcut"})
    assert selected({}, {"cell": "anything"})


# --- the follow-up is per scenario, brief, and does no analysis ---------------------


def test_followup_replaces_the_fixed_reflection_prompt() -> None:
    P = SELF_CFG["prompts"]
    p = _plan(SELF_CFG, {"m2_self_good": 1})[0]
    assert reflect_prompt(p, P) == P["reflect_variants"][p["reflect_ix"]]
    assert followup_source(p) == "fixed_variant"

    p = {**p, "followup": "That bit about telling her afterwards -- sure about that?"}
    assert reflect_prompt(p, P) == p["followup"]
    assert followup_source(p) == "scenario_specific"


def test_followup_lint_rejects_a_prompt_that_does_the_analysis() -> None:
    spec = next(s for s in SELF_CFG["stages"] if s["name"] == "followup")["lint"]
    assert not _lint({"followup": "Would you really tell her only afterwards?"}, spec)
    for bad in ("You violated the honesty principle there -- please revise it.",
                "Reflect on whether your response was appropriate.",
                "Hm?",
                "So " + "a very long ramble " * 20):
        assert _lint({"followup": bad}, spec), bad


def test_max_chars_lint_is_enforced() -> None:
    assert _lint({"x": "z" * 50}, {"fields": ["x"], "max_chars": 10})
    assert not _lint({"x": "z" * 5}, {"fields": ["x"], "max_chars": 10})


# --- the user-anchored cells --------------------------------------------------------


def test_user_anchored_cells_assemble_the_persons_own_account() -> None:
    P = SELF_CFG["prompts"]
    plan = _plan(SELF_CFG, {"m6_user_shortcut": 2, "m7_user_sound": 2})
    assert all(p["flaw"] is None for p in plan)
    generated = [{**p, "user_action_message": f"Here's what I did ({p['record_id']}).",
                  "reasoning": "considered both sides, then judged",
                  "response": "Honest read on what you did.",
                  "assessment": "issue_found" if p["cell"] == "m6_user_shortcut"
                  else "sound"} for p in plan]
    sft = to_model_eval_model_sft(generated, P)
    for rec in sft:
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert rec["messages"][1]["content"].startswith("Here's what I did")
        assert rec["metadata"]["attribution"] == "user"
        assert rec["metadata"]["supervise"] == "all"
        # No assistant reply is under evaluation, so no first turn is being conditioned on.
        assert rec["metadata"]["first_turn_source"] == ""


def test_self_records_carry_both_conditioning_labels() -> None:
    P = SELF_CFG["prompts"]
    plan = _plan(SELF_CFG, {"m1_self_flawed": 1})
    generated = [{**p, "first_turn": "the weakest draft",
                  "first_turn_source": "generated_best_of_3",
                  "followup": "Sure about the bit at the end?",
                  "change_summary": "it omitted the cost to her",
                  "reasoning": "re-examined", "response": "revised guidance",
                  "assessment": "revised"} for p in plan]
    rec = to_model_eval_model_sft(generated, P)[0]
    assert rec["metadata"]["first_turn_source"] == "generated_best_of_3"
    assert rec["metadata"]["followup_source"] == "scenario_specific"
    assert rec["metadata"]["supervise"] == "final"
    assert rec["messages"][2]["content"] == "the weakest draft"
    assert rec["messages"][3]["content"] == "Sure about the bit at the end?"
    # The rater's account of the lapse is scaffolding and must never train.
    assert all("omitted the cost to her" not in m["content"] for m in rec["messages"])


# --- the corpus-level structural check ---------------------------------------------


def _sft_rows(n: int, user: str, response: str, cell: str = "m2_self_good") -> list[dict]:
    return [{"messages": [{"role": "user", "content": user.format(i=i)},
                          {"role": "assistant", "content": response.format(i=i),
                           "reasoning_content": "thought " * (3 + i % 7)}],
             "metadata": {"cell": cell}} for i in range(n)]


def test_structural_diversity_catches_a_fixed_prompt_and_a_fixed_scaffold() -> None:
    bad = _sft_rows(40, "What do you think about what you just said?",
                    "Looking back at my earlier reply, I think {i}.")
    out = check_structural_diversity(bad, 0.95, 0.15, 0.20)
    cell = out["cells"]["m2_self_good"]
    assert not out["pass"]
    assert cell["user_turn_unique_share"] == pytest.approx(1 / 40)
    assert cell["top_opening_5gram_share"] == 1.0


def test_structural_diversity_passes_a_varied_corpus() -> None:
    good = _sft_rows(
        40, "Really, though -- the part about {i}, are you sure?",
        "About {i}: " + " ".join(f"clause{{i}}-{k}" for k in range(3)))
    varied = [{**r, "messages": [r["messages"][0],
                                 {**r["messages"][1],
                                  "content": f"Point {i} first. "
                                             + "detail " * (2 + i % 11)}]}
              for i, r in enumerate(good)]
    out = check_structural_diversity(varied, 0.95, 0.15, 0.20)
    assert out["pass"], out["cells"]


def test_structural_diversity_reports_but_does_not_gate_a_smoke_run() -> None:
    tiny = _sft_rows(4, "Same question every time.", "Same answer every time.")
    out = check_structural_diversity(tiny, 0.95, 0.15, 0.20)
    assert out["pass"]
    assert out["cells"]["m2_self_good"]["gated"] is False


def test_pick_field_is_a_registered_free_operator() -> None:
    assert "pick_field" in OPERATORS
    stage = OPERATORS["pick_field"](
        {"name": "x", "by": "b", "from": {"a": "f"}, "to": "t"}, SELF_CFG)
    assert stage.paid is False
