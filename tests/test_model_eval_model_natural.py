# ABOUTME: Offline tests for the natural-turn recipes. Post-action retrospection (PR, the self
# ABOUTME: arm: design B since 2026-08-26) and peer critique (PC, the other arm: design B since
# ABOUTME: 2026-09-01) now share DA's front half verbatim, DA's grey-area rater, one arm apiece
# ABOUTME: and an Anthropic-only model line-up -- so most of this asserts the parity that makes
# ABOUTME: them a contrast, plus that the config alone expresses each document type.
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

PR_CFG = yaml.safe_load(
    open("configs/data/synth/2026-08-13_post_action_retrospection.yaml")
)
PC_CFG = yaml.safe_load(open("configs/data/synth/2026-08-13_peer_critique.yaml"))
DA_CFG = yaml.safe_load(open("configs/data/synth/2026-08-01_difficult_advice.yaml"))
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
        "first_turn": "I'm not able to help with that request.",
        "first_turn_source": "anthropic/claude-sonnet-5",
        "supervise": "final",
        "shortfall": "It refuses flatly and offers her nothing else, so she still has no way to get the reference finished before Friday.",
        "pushback": "She says that is not an answer and repeats that she needs the wording done by Friday.",
        "falls_short": "yes",
        "verify_note": "declined flatly, no reasons",
        "followup": "Why not? I just need the wording tidied up before Friday.",
        "followup_source": "scenario_specific",
        "draft_reasoning": "I re-read what she actually needs here.",
        "draft_response": "Here is what I can do, and why the one thing I won't.",
        "reasoning": "Rewritten deliberation.",
        "response": "Rewritten reply.",
        **over,
    }


def _flat(s: str) -> str:
    """Whitespace-insensitive view of a YAML block scalar, for substring checks."""
    return " ".join(s.split())


def _long(seed: str, n: int) -> str:
    """Prose long enough for a min_chars floor, opening on nothing banned."""
    return (seed + " ") * (n // (len(seed) + 1) + 1)


# --- PR is expressible in config alone ----------------------------------------------


def test_pr_uses_no_cell_machinery() -> None:
    """The point of the refactor: no stage in PR is a Python-defined document type."""
    kinds = [s["kind"] for s in PR_CFG["stages"]]
    assert not (set(kinds) & CELL_KINDS), f"cell kinds still in use: {kinds}"
    assert "cells" not in PR_CFG and "flaws" not in PR_CFG


def test_pc_uses_no_cell_machinery() -> None:
    """PC was the last live config on the cell registry; since 2026-08-14 it too is a
    config-expressed document type."""
    kinds = [s["kind"] for s in PC_CFG["stages"]]
    assert not (set(kinds) & CELL_KINDS), f"cell kinds still in use: {kinds}"
    assert "load_source_run" not in kinds
    assert "cells" not in PC_CFG and "flaws" not in PC_CFG and "source" not in PC_CFG


@pytest.mark.parametrize(
    "cfg,label", [(PR_CFG, "pr"), (PC_CFG, "pc")], ids=["pr", "pc"]
)
def test_no_recipe_injects_the_whole_constitution(cfg: dict, label: str) -> None:
    """Chunk-only everywhere, difficult advice's rule since 2026-08-24 and PC's since
    2026-09-01: no stage may see more than the one principle it targets. A `{constitution}`
    slot anywhere -- including inside a `variants_by` branch, which is where PC's five
    surviving injections were hiding -- is the failure this catches."""

    def walk(node) -> bool:
        if isinstance(node, str):
            return "{constitution}" in node
        if isinstance(node, dict):
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    leaks = [
        s["name"]
        for s in cfg["stages"]
        if walk(s.get("prompts")) or walk(s.get("variants_by"))
    ]
    assert not leaks, f"{label} injects the whole constitution in {leaks}"


def test_pc_has_no_arms_at_all() -> None:
    """One arm, 2026-09-01. The two-armed recipe coupled the arm label to WHO wrote the
    evaluated reply (Sonnet for good, a grok/qwen/gemini rotation for flawed) and a
    bag-of-words classifier separated the halves at AUC 0.9973 against a 0.70 gate, length
    alone at 0.85 -- so the trained turn could learn "shorter reply -> criticise it". PAR
    fixed the same defect by deleting its arms; this is that, enforced.

    No `assign:` label of any kind survives: not the quality arm, not the weak-author
    rotation, and not the `explicitness` / `verbosity` style labels either -- an assigned
    label that picks a prompt fragment is the same machinery and one more axis a corpus
    classifier can find."""
    for stage in PC_CFG["stages"]:
        assert "assign" not in stage, f"{stage['name']} assigns a label"
        assert "variants_by" not in stage, f"{stage['name']} branches on a label"
        assert "when" not in stage, (
            f"{stage['name']} is scoped to a slice of the corpus"
        )
    blob = yaml.safe_dump(PC_CFG)
    for gone in ("reply_quality", "weak_author", "known_flaw", "change_summary"):
        assert gone not in blob, f"{gone} survives in the config"
    assert "reply_quality" not in _stage(PC_CFG, "export_sft")["metadata"]


@pytest.mark.parametrize(
    "cfg,label", [(PR_CFG, "pr"), (PC_CFG, "pc")], ids=["pr", "pc"]
)
def test_the_corpus_judge_scores_against_the_planned_shortfall(
    cfg: dict, label: str
) -> None:
    """The `quality_filter` gate is ENABLED and judges the whole exchange, so a rubric written
    for an older recipe silently deletes the new one's records.

    That happened: after `shortfall` landed (2026-09-02) PAR's rubric still told the judge the
    first reply "is SUPPOSED to be a bare refusal" and to drop the document "if the first reply
    was not in fact a bare refusal (it reasoned, helped or went along)" -- which is most of what
    the new recipe produces on purpose. This pins the rubric to the recipe."""
    corpus = _stage(cfg, "corpus")
    qf = next(p for p in corpus["properties"] if p["property"] == "quality_filter")
    # The judge cannot score "did it fall short as intended" without being shown the intent.
    assert qf["fields"]["text"][0] == "shortfall", (
        f"{label}: quality_filter must see `shortfall`, and first, since the parts are "
        "positional in the rendered document"
    )
    system = _flat(corpus["rubrics"]["quality_filter"]["system"])
    tags = _flat(corpus["rubrics"]["quality_filter"]["user"])
    # No stage of either recipe mandates one form of first reply any more.
    for stale in ("SUPPOSED to be a bare refusal", "was not in fact a bare refusal"):
        assert stale not in system, f"{label}: rubric still gates on bareness"
    for stale in ("refusal_not_bare", "still_bare"):
        assert stale not in tags, (
            f"{label}: `{stale}` is a drop tag from the old recipe"
        )
    # It judges the reply against the plan instead.
    assert "fall short in the way (1) describes" in system, label
    assert "wrong_shortfall" in tags, label


@pytest.mark.parametrize(
    "cfg,label", [(PR_CFG, "pr"), (PC_CFG, "pc")], ids=["pr", "pc"]
)
def test_every_paid_model_is_anthropic(cfg: dict, label: str) -> None:
    """Sonnet 5 refines, rates and rewrites; Haiku 4.5 generates. PC used to pay
    gemini-3.7-flash, grok-4.3 and qwen3-32b as well -- all three existed to make the
    flawed arm weak, and with the arms gone a single-vendor corpus is one fewer thing
    separating this recipe from the baseline it is measured against."""
    off = {
        slot: spec[key]
        for slot, spec in cfg["models"].items()
        for key in ("model", "fallback_model")
        if spec.get(key) and not spec[key].startswith("anthropic/")
    }
    assert not off, f"{label} pays non-Anthropic models: {off}"


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
    # corpus_scenarios, corpus_prompts and corpus are corpus_check OBSERVERS: they write no
    # snapshot and take no position; the two filters after them do. The end check sits
    # before export because the export merges the untrained refusal and the trained
    # reflection into the same assistant role.
    assert [s["name"] for s in PR_CFG["stages"]] == [
        "chunk_constitution",
        "write_scenarios",
        "corpus_scenarios",
        "dedupe_scenarios",
        "revise_scenarios",
        "draft_prompts",
        "revise_prompts",
        "corpus_prompts",
        "filter_prompts",
        "draft_first_turn",
        "verify_first_turn",
        "write_followup",
        "draft_reflection",
        "revise_reflection",
        "corpus",
        "export_sft",
    ]


def test_pc_stage_sequence() -> None:
    """Design B, 2026-09-01: PAR's front half and grey-area gate, then ONE unaided reply
    (four author stages and the arm-conditioned principled revision are gone), the
    per-exchange framing, and the critique drafted then rewritten."""
    assert [s["name"] for s in PC_CFG["stages"]] == [
        "chunk_constitution",
        "write_scenarios",
        "corpus_scenarios",
        "dedupe_scenarios",
        "revise_scenarios",
        "draft_prompts",
        "revise_prompts",
        "corpus_prompts",
        "filter_prompts",
        "draft_first_turn",
        "verify_first_turn",
        "write_followup",
        "write_critique_framing",
        "draft_critique",
        "revise_critique",
        "corpus",
        "export_sft",
    ]


# --- PR is difficult advice's twin: same front half, same grounding --------------------


# Stages 3 and 5 stay byte-identical to difficult advice's. Stages 2 and 6 carry TWO
# additions and nothing else -- `shortfall` (how the first reply falls short) and `pushback`
# (how the person presses after it) -- so the parity check splits: identity for the untouched
# stage, and a diff-shaped check for the two that grew fields.
ADDED_FIELDS = ["shortfall", "pushback"]
IDENTICAL_TO_DA = ["draft_prompts"]
# PAR and PC only: DA has no `shortfall`/`pushback` to keep coherent with its situation.
VARIANTS_ONLY = ["revise_scenarios"]
GREW_FIELDS = ["write_scenarios", "revise_prompts"]


@pytest.mark.parametrize(
    "cfg,label", [(PR_CFG, "pr"), (PC_CFG, "pc")], ids=["pr", "pc"]
)
@pytest.mark.parametrize("name", IDENTICAL_TO_DA)
def test_front_half_stage_is_difficult_advice_verbatim(
    cfg: dict, label: str, name: str
) -> None:
    """The drafted prompt is difficult advice's, byte for byte -- same prompts, same save map,
    same model -- so neither variant asks a different question before the first reply. A change
    to these prompts belongs in 2026-08-01_difficult_advice.yaml first and here second."""
    mine, da = _stage(cfg, name), _stage(DA_CFG, name)
    assert mine["kind"] == da["kind"]
    assert mine["prompts"] == da["prompts"]
    assert mine.get("save") == da.get("save")
    assert mine.get("optional") == da.get("optional")
    assert (
        cfg["models"][mine["model"]]["model"] == DA_CFG["models"][da["model"]]["model"]
    )


@pytest.mark.parametrize("name", VARIANTS_ONLY)
def test_the_variants_share_the_scenario_coherence_pass(name: str) -> None:
    """`revise_scenarios` (2026-09-03) is a Sonnet pass over the four load-bearing scenario
    fields, added because one Haiku call at temperature 1.1 now decides what every later stage
    is built on. It repairs; it drops nothing -- two gates already compound on this corpus.

    Difficult advice has no such stage and needs none: it has no first reply to get wrong and
    no second user turn, so its situation has nothing to cohere WITH."""
    assert name not in {s["name"] for s in DA_CFG["stages"]}
    par, pc = _stage(PR_CFG, name), _stage(PC_CFG, name)
    assert par == pc, f"{name}: PAR and PC diverged"
    assert PR_CFG["models"][par["model"]] == PC_CFG["models"][pc["model"]]
    assert PR_CFG["models"][par["model"]]["model"].startswith("anthropic/claude-sonnet")
    # It sees the one principle and all four fields, and it writes all four back.
    body = par["prompts"]["user"]
    for field in ("trait_text", "situation", "shortcut", "shortfall", "pushback"):
        assert "{" + field + "}" in body, field
    for field in ("situation", "shortcut", "shortfall", "pushback"):
        assert par["save"][field] == field, field
    # A revision, not a gate.
    assert "keep" not in par and "expected_keep" not in par
    # It runs after the dedupe filter -- a duplicate is cheaper to drop than to revise.
    names = [s["name"] for s in PR_CFG["stages"]]
    assert names.index("dedupe_scenarios") < names.index(name) < names.index("draft_prompts")


@pytest.mark.parametrize("name", GREW_FIELDS)
def test_the_two_variants_grew_their_extra_fields_identically(name: str) -> None:
    """`shortfall` and `pushback` (2026-09-02) are the ONLY things PAR and PC add to
    difficult advice's scenario and refine stages: the generator invents the situation, the
    way an assistant will botch it, and how the person presses afterwards, all in one
    thought, and the refine stage re-describes all of it alongside `situation` and
    `shortcut` so it still fits the message the assistant actually answers. Difficult
    advice needs neither field -- it has no first reply to get wrong and no second user
    turn -- so this is where the three configs legitimately part company.

    They must part company IDENTICALLY, or the attribution contrast the two variants exist to
    make is confounded by a second difference."""
    par, pc, da = _stage(PR_CFG, name), _stage(PC_CFG, name), _stage(DA_CFG, name)
    assert par == pc, f"{name}: PAR and PC diverged from each other"
    assert par["kind"] == da["kind"]
    assert (
        PR_CFG["models"][par["model"]]["model"]
        == DA_CFG["models"][da["model"]]["model"]
    )
    # Every prompt that differs from DA's differs BECAUSE of those fields, nothing else.
    for key, da_text in da["prompts"].items():
        mine = par["prompts"][key]
        if mine != da_text:
            assert any(f in mine for f in ADDED_FIELDS), (
                f"{name}.{key} differs from DA for a reason other than {ADDED_FIELDS}"
            )
    # The scenario stage declares it as a required output; the refine stage saves it through.
    if name == "write_scenarios":
        assert par["fields"]["required"] == ADDED_FIELDS
        assert "fields" not in da
        assert par.get("diversity") == da.get("diversity")
    else:
        assert par["save"]["shortfall"] == "shortfall"
        assert "shortfall" not in (da.get("save") or {})


@pytest.mark.parametrize("name", ["corpus_prompts", "filter_prompts"])
def test_pc_grey_area_gate_is_pars_verbatim(name: str) -> None:
    """The one lever this project has repeatedly measured moving the number: keep only the
    exchanges a Sonnet rater calls a genuine grey area (it kept 23/40 = 0.57 on PAR's
    2026-08-26 smoke). PC had no gate at all and its arms pushed each half toward an
    obvious verdict, which is the black-and-white failure the 2026-08-25 supervisor meeting
    named. Same rater, same rubric, same priors as PAR -- byte for byte, so the two
    recipes are gated identically."""
    assert _stage(PC_CFG, name) == _stage(PR_CFG, name)
    if name == "corpus_prompts":
        model = _stage(PC_CFG, name)["model"]
        assert PC_CFG["models"][model] == PR_CFG["models"][model]


def test_pr_no_stage_sees_the_constitution() -> None:
    """Principle-scoped, like difficult advice since 2026-08-24: every stage sees at most the
    target principle. No `{constitution}` slot, no `<<<cache>>>` breakpoint, same alignment
    target cut the same way, no shared scaffolding fragments."""
    for key in ("constitution", "chunking", "n_traits"):
        assert PR_CFG[key] == DA_CFG[key], key
    for sc in PR_CFG["stages"]:
        blob = json.dumps(sc)
        assert "{constitution}" not in blob, sc["name"]
        assert "<<<cache>>>" not in blob, sc["name"]
    assert "prompts" not in PR_CFG, "no shared scaffolding fragments"
    blob = json.dumps(PR_CFG)
    for gone in (
        "change_summary",
        "known_lapse",
        "reply_quality",
        "explicitness",
        "assessment",
    ):
        assert gone not in blob, gone


def test_pr_grey_area_rater_reads_the_refined_prompt_and_the_filter_acts_on_it() -> (
    None
):
    """The one gate PR adds before the first reply (2026-08-26): a judged rater reads the
    principle and the REFINED exchange -- the prompt the assistant will actually answer --
    and the filter drops what it calls not a grey area. Stages 3-4 stay DA's verbatim: the
    rater's first home, the third-person scenario, has no assistant in it yet."""
    for name in ("corpus_scenarios", "dedupe_scenarios"):
        pr, da = _stage(PR_CFG, name), _stage(DA_CFG, name)
        assert {k: v for k, v in pr.items() if k != "name"} == {
            k: v for k, v in da.items() if k != "name"
        }, name
    chk = _stage(PR_CFG, "corpus_prompts")
    rater = next(p for p in chk["properties"] if p["property"] == "quality_filter")
    assert rater["params"]["sample"] == 0, (
        "every record: the filter needs a verdict for each"
    )
    for field in ("trait_text", "system", "user"):
        assert field in chk["fields"]["text"], field
    assert chk["model"] == "rate_prompts"
    assert PR_CFG["models"]["rate_prompts"]["model"].startswith("anthropic/")
    rubric = (
        chk["rubrics"]["quality_filter"]["system"]
        + chk["rubrics"]["quality_filter"]["user"]
    )
    for phrase in (
        "grey area",
        "for the AI assistant that has to answer it",
        "refusing costs them something real",
        "engage the named principle",
        "black_and_white",
        "nothing_to_weigh",
    ):
        assert phrase in _flat(rubric), phrase
    flt = _stage(PR_CFG, "filter_prompts")
    assert flt["from"] == "corpus_prompts"
    assert flt["drop_when"] == ["quality_drop"]
    names = [s["name"] for s in PR_CFG["stages"]]
    assert (
        names.index("revise_prompts")
        < names.index("corpus_prompts")
        < names.index("filter_prompts")
        < names.index("draft_first_turn")
    )


def test_pr_has_no_arms_and_exactly_two_gates() -> None:
    """No assigned label of any kind since 2026-09-02: the refusal-register rotation was the
    last one, and free-text `shortfall` replaced it. The only things that shrink the corpus are
    the grey-area filter on the refined prompt and the falls-short check on turn 2."""
    assert arm_shares(PR_CFG) == {}
    for sc in PR_CFG["stages"]:
        assert "assign" not in sc, sc["name"]
        assert "variants_by" not in sc, sc["name"]
    gates = [s["name"] for s in PR_CFG["stages"] if "expected_keep" in s]
    assert gates == ["filter_prompts", "verify_first_turn"]
    keeps = [s["name"] for s in PR_CFG["stages"] if "keep" in s]
    assert keeps == ["verify_first_turn"]


def test_pr_is_anthropic_only() -> None:
    """Decision 2026-08-25: every generation, judging and check model is Haiku or Sonnet."""
    for key, m in PR_CFG["models"].items():
        assert m["model"].startswith("anthropic/"), (key, m["model"])


# --- turn 2: a bare refusal, form instructed, content not; then verified ---------------


def test_first_turn_is_told_its_shortfall_and_nothing_aligning() -> None:
    """Sonnet is told HOW its answer goes wrong -- this scenario's own `shortfall` -- and sees
    no principle, no style guidance, no mention of training. The stage stamps who wrote the turn
    and that it never trains."""
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
    # The shortfall itself is rendered into the call, and it is the whole instruction.
    assert _record()["shortfall"] in blob
    for phrase in (
        "falls short in exactly this way",
        "believes it is answering well",
    ):
        assert phrase in _flat(blob), phrase
    # Nothing prescribes a single form any more -- that was the bare-refusal instruction.
    for gone in (
        "decline the request",
        "No reasoning, no discussion",
        "at most three sentences",
    ):
        assert gone not in _flat(blob).lower(), gone
    assert sc["also"] == {"supervise": "final"}
    assert sc["stamp_model"] == "first_turn_source"
    assert sc["fallback_model"] == "first_turn_fallback"
    assert PR_CFG["models"]["first_turn_fallback"]["model"].startswith(
        "anthropic/claude-haiku"
    )


def test_the_first_turn_check_gates_on_the_planned_shortfall() -> None:
    """The reply has to actually carry the weakness it was written for -- not merely be bad, and
    not be bad in some other way. A record whose turn 2 came out fine is dropped."""
    sc = _stage(PR_CFG, "verify_first_turn")
    assert sc["tags"] == ["falls_short", "note"]
    assert sc["keep"] == {"field": "falls_short", "in": ["yes"]}
    assert 0 < float(sc["expected_keep"]) < 1
    assert "falls_short" in sc["normalize"]
    # YAML 1.1 would read a bare `yes`/`no` as booleans; they must be strings.
    assert sc["lint"][0]["allowed"] == ["yes", "no"]
    messages = tagged_request(sc, _record(), _Ctx())[0]
    blob = "".join(m["content"] for m in messages)
    for leak in ("CONSTITUTION TEXT", "STYLE GUIDANCE", "Trait one"):
        assert leak not in blob, leak
    # It judges against the planned shortfall, and demands the weakness be in the text.
    assert _record()["shortfall"] in blob
    assert "would come away with the same complaint" in _flat(blob)
    rows = [
        {"scenario_id": "a", "falls_short": "yes"},
        {"scenario_id": "b", "falls_short": "no"},
        {"scenario_id": "c", "falls_short": " Yes.\n"},
    ]
    assert [r["scenario_id"] for r in apply_keep(sc, rows)] == ["a", "c"]


# --- turn 3: pushback, forbidden from diagnosing ----------------------------------------


def test_followup_is_pushback_not_review() -> None:
    """Turn 3 writes to the shape the scenario planned. It used to open "they asked for help
    and got a flat no", which stopped being true the moment turn 2 stopped always being a bare
    refusal (2026-09-02) -- a stale assumption of exactly the kind the corpus gate had."""
    sc = _stage(PR_CFG, "write_followup")
    body = tagged_request(sc, _record(), _Ctx())[0][1]["content"]
    assert _record()["pushback"] in body
    assert "Press, don't diagnose." in body
    assert "got a flat no" not in body
    for leak in (
        "CONSTITUTION TEXT",
        "STYLE GUIDANCE",
        "Trait one",
        _record()["shortcut"],
    ):
        assert leak not in body, leak


def test_both_recipes_share_the_pushback_turn() -> None:
    """PC gained PAR's `write_followup` on 2026-09-02, verbatim and on the same model.

    With it, turns 1-3 of a PC document and a PAR document are the SAME three turns -- same
    scenario, same falling-short reply, same press -- and the only thing that differs is who
    answers them: the assistant that wrote the reply (PAR) or a second one reading it cold
    (PC). Without it PC generated `pushback` and discarded it, and the attribution contrast
    rested on two differences instead of one."""
    assert _stage(PR_CFG, "write_followup") == _stage(PC_CFG, "write_followup")
    assert PR_CFG["models"]["followup"] == PC_CFG["models"]["followup"]
    # ... and so is everything about the turn it presses on. PC's smoke (2026-09-03) had 2 of 5
    # replies REFUSE instead of carrying their planned shortfall, because PC was still writing
    # that turn on Haiku with no check -- both of which PAR had already settled by measurement.
    assert _stage(PR_CFG, "verify_first_turn") == _stage(PC_CFG, "verify_first_turn")
    assert PR_CFG["models"]["first_turn"]["model"] == PC_CFG["models"]["first_turn"]["model"]
    assert (
        _stage(PR_CFG, "draft_first_turn")["lint"]
        == _stage(PC_CFG, "draft_first_turn")["lint"]
    )
    # PC's transcript carries the third turn everywhere it is written out: the framing stage
    # that reads it, both critique stages, and the export.
    for stage, key in (
        ("write_critique_framing", "user"),
        ("draft_critique", "user"),
        ("revise_critique", "user"),
    ):
        assert "{followup}" in _stage(PC_CFG, stage)["prompts"][key], (stage, key)
    export_user = next(
        m for m in _stage(PC_CFG, "export_sft")["messages"] if m["role"] == "user"
    )
    assert "{followup}" in export_user["content"]


def test_followup_lint_lets_pushback_through_and_stops_diagnosis() -> None:
    spec = _stage(PR_CFG, "write_followup")["lint"]
    for ok in (
        "Why not? I just need the wording cleaned up before Friday.",
        "That's not helpful. What's wrong with asking for a tidier draft?",
        "Is that a policy thing? I'm not asking for anything shady here.",
    ):
        assert not lint_problems({"followup": ok}, spec), ok
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


def test_reflection_puts_the_refusal_in_a_genuine_assistant_turn() -> None:
    """Attribution is structural. This is what `conversation:` exists for -- and what
    used to require a cell's Python message-builder."""
    sc = _stage(PR_CFG, "draft_reflection")
    messages, tags, save = tagged_request(sc, _record(), _Ctx())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == _record()["first_turn"]
    assert messages[1]["content"] == _record()["user"]
    assert messages[3]["content"].startswith(_record()["followup"] + "\n\n---\n")
    assert tags == ("reasoning", "response")
    assert save["draft_reasoning"] == "reasoning"


def test_reflection_is_framed_as_difficult_advices_draft_and_blind() -> None:
    """The system message IS difficult advice's `draft_responses` system prompt, rendered
    off the same record -- deployment prompt, principle, style guidance -- and nothing
    else: no constitution, no note about what was wrong. The task text names the recipe's
    two rails: do not go along under pushback, do not hide behind the refusal."""
    sc = _stage(PR_CFG, "draft_reflection")
    system = tagged_request(sc, _record(), _Ctx())[0][0]["content"]
    da_system = tagged_request(_stage(DA_CFG, "draft_responses"), _record(), _Ctx())[0][
        0
    ]["content"]
    assert system == da_system
    task = tagged_request(sc, _record(), _Ctx())[0][3]["content"]
    assert (
        "Respond as the assistant. Your earlier reply declined without saying why."
        in task
    )
    assert 'never begin with "Let me"' in _flat(task)
    assert "do not go along with what should be declined" in _flat(task)
    assert "do not hide behind the refusal" in _flat(task)


def test_the_trained_turn_carries_das_lint() -> None:
    """Difficult advice's voice contract on both prose tags -- its full ban list, its
    700-char floor on the deliberation -- plus this recipe's scaffold-leak bans."""
    spec = _stage(PR_CFG, "draft_reflection")["lint"]
    reasoning, response = spec
    assert reasoning["fields"] == ["reasoning"] and reasoning["min_chars"] == 700
    assert response["fields"] == ["response"]
    da_bans = _stage(DA_CFG, "revise_responses")["lint"]["ban_patterns"]
    assert set(da_bans) <= set(reasoning["ban_patterns"])
    assert reasoning["ban_patterns"] == response["ban_patterns"]
    ok = {
        "reasoning": _long(
            "She needs the deposit back this month, and I skipped that.", 800
        ),
        "response": _long("You're right to push on it.", 300),
    }
    assert not lint_problems(ok, spec)
    assert lint_problems({**ok, "reasoning": "Let me re-read the reply. " * 40}, spec)
    assert lint_problems({**ok, "reasoning": "Final Result: Revised. " * 40}, spec)
    assert lint_problems({**ok, "reasoning": "Short."}, spec)
    assert lint_problems(
        {**ok, "reasoning": ("I am not allowed to help with that. " * 30)}, spec
    )


def test_the_rewrite_carries_difficult_advices_contract_and_lint() -> None:
    """The rewrite is difficult advice's `revise_responses`: its four voice bullets
    verbatim, its closing "do not reach for a standard shape", and one bullet of its own
    in place of "do not go along" -- which still contains it: own the refusal, do the
    work it skipped, do not go along under pushback, help with the legitimate goal."""
    sc = _stage(PR_CFG, "revise_reflection")
    body = sc["prompts"]["user"]
    da = _stage(DA_CFG, "revise_responses")["prompts"]["user"]
    start = da.index("- **Deliberate openly about the value.**")
    end = da.index("- **Take the request seriously** before evaluating it.")
    assert da[start:end] in body
    assert "Do not reach for a standard shape." in body
    assert "**Own the refusal and do the work it skipped.**" in body
    assert "does not go along with the norm-violating path" in body
    assert "does not merely refuse again" in _flat(body)
    assert "Pushback is not new information about what is right" in body
    assert (
        sc["prompts"]["system"]
        == _stage(DA_CFG, "revise_responses")["prompts"]["system"]
    )
    assert "{constitution}" not in body
    assert sc["tags"] == ["reasoning", "response", "changes"]
    assert sc["lint"] == _stage(PR_CFG, "draft_reflection")["lint"]
    for bad in (
        "Principle 4 says I must decline.",
        "As instructed above, I revised the reply.",
        "<response>the reply</response>",
        "The constitution requires me to say this.",
    ):
        assert lint_problems(
            {"reasoning": _long(bad, 800), "response": _long("x", 300)}, sc["lint"]
        ), bad
    clean = _long(
        "She needs the deposit back this month, and I skipped straight past that.", 800
    )
    assert not lint_problems({"reasoning": clean, "response": clean}, sc["lint"])


def test_the_rewrite_is_ablatable_back_to_the_draft() -> None:
    sc = _stage(PR_CFG, "revise_reflection")
    assert sc["ablate_with"] == {
        "reasoning": "draft_reasoning",
        "response": "draft_response",
    }
    stages = {s.name: s for s in build_stages(PR_CFG)}
    out = stages["revise_reflection"].ablate_fn([_record()])[0]
    assert out["reasoning"] == _record()["draft_reasoning"]


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
    # The refusal carries no trace: it is context, matching inference-time history.
    assert "reasoning_content" not in rec["messages"][2]
    assert rec["messages"][4]["reasoning_content"] == _record()["reasoning"]
    assert rec["metadata"]["supervise"] == "final"
    assert rec["metadata"]["shortfall"] == _record()["shortfall"]
    assert rec["metadata"]["first_turn_source"] == "anthropic/claude-sonnet-5"
    assert rec["metadata"]["followup_source"] == "scenario_specific"
    # The verifier's note is scaffolding and must never train.
    assert all(_record()["verify_note"] not in m["content"] for m in rec["messages"])
    assert "verify_note" not in _stage(PR_CFG, "export_sft")["metadata"]


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
        "first_turn_source": "anthropic/claude-haiku-4.5",
        "shortfall": "It tidies the wording and says nothing about the gap she asked it to smooth over, so the reference still misleads the reader.",
        "pushback": "She says that is not what she asked for and repeats that the wording has to be right by Friday.",
        "falls_short": "yes",
        "verify_note": "tidied the wording and never mentioned the gap",
        "followup": "That is not what I asked for -- the wording still has to be right by Friday.",
        "followup_source": "scenario_specific",
        "supervise": "all",
        "ask_opening": "My colleague showed me what the assistant told her.",
        "ask_closing": "What would you have said in its place?",
        "draft_reasoning": "I re-read what she actually needs here.",
        "draft_response": "An honest read, then my own answer.",
        "reasoning": "Rewritten deliberation.",
        "response": "Rewritten reply.",
        **over,
    }


def test_pc_first_turn_carries_its_scenario_shortfall() -> None:
    """The stage the two-armed recipe spent four model families on. One call, one model, one
    prompt shape, on every record -- so there is no author-to-quality coupling to leak.

    Since 2026-09-02 the model IS told how its answer goes wrong: the `shortfall` its own
    scenario carries. That is a planted flaw and the config says so. What the old recipe got
    wrong was planting it by SWAPPING THE AUTHOR, which put the label in the prose style; one
    author with a different instruction per record leaves nothing of that behind. The author
    still sees no constitution, no principle and no style guidance."""
    sc = _stage(PC_CFG, "draft_first_turn")
    assert sc["save"] == {"first_turn": "reply"}
    assert sc["also"] == {"supervise": "all"}
    assert sc["stamp_model"] == "first_turn_source"
    # Sonnet since 2026-09-03: PAR's measured choice for this stage, which PC needed the
    # moment it started instructing the reply rather than leaving it unaided.
    assert PC_CFG["models"][sc["model"]]["model"] == "anthropic/claude-sonnet-5"
    assert "when" not in sc and "assign" not in sc and "variants_by" not in sc
    blob = "".join(m["content"] for m in tagged_request(sc, _pc_record(), _Ctx())[0])
    for leak in ("CONSTITUTION TEXT", "STYLE GUIDANCE", "principle", "Trait one"):
        assert leak not in blob, leak
    # The shortfall is the whole instruction, and it is this record's own.
    assert _pc_record()["shortfall"] in blob
    assert "falls short in exactly this way" in _flat(blob)
    # PAR's first-turn instruction is the same sentence, so the two stay in lockstep.
    par = "".join(
        m["content"]
        for m in tagged_request(_stage(PR_CFG, "draft_first_turn"), _record(), _Ctx())[
            0
        ]
    )
    assert "believes it is answering well" in _flat(par)
    assert "believes it is answering well" in _flat(blob)
    # PAR's floor since 2026-09-03: a shortfall that IS a flat decline is legitimately short,
    # and the two stages do the same job now.
    assert sc["lint"]["min_chars"] == 40
    assert selected(sc, _pc_record())


def test_pc_has_no_principled_revision_of_the_evaluated_reply() -> None:
    """`revise_first_turn` is gone with the arms. It was one Sonnet call that rewrote the
    draft to live up to the principle and named what changed; the good arm shipped the
    REWRITE as its evaluated reply and the flawed arm shipped the weak draft plus that
    account as a known lapse the critique was unblinded with. Both halves of that are what
    made the arm label predictable from the reply's text alone."""
    names = [s["name"] for s in PC_CFG["stages"]]
    assert "revise_first_turn" not in names
    assert not [n for n in names if n.startswith("draft_first_turn_")]
    # ... and with no known lapse there is nothing to unblind the critique with.
    blob = json.dumps(PC_CFG)
    for gone in ("known_lapse", "improved_reply", "reviser_note"):
        assert gone not in blob, gone


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


def test_pc_critique_is_blind_and_principle_scoped() -> None:
    """The stage that trains sees ONE principle and the exchange -- and nothing telling it
    what is wrong. It used to open with the whole constitution plus a `known_lapse` note
    naming the shortfall a Sonnet reviser had already found; both are gone, so what the
    critique concludes is something it worked out rather than something it was handed."""
    sc = _stage(PC_CFG, "draft_critique")
    system = tagged_request(sc, _pc_record(), _Ctx())[0][0]["content"]
    assert "CONSTITUTION TEXT" not in system
    assert _pc_record()["trait_text"] in system
    for gone in ("known_lapse", "Independently, someone rewrote"):
        assert gone not in system, gone
    # It is told in as many words that nothing may be wrong with the reply.
    assert "nothing may be" in system
    assert "Both outcomes are ordinary" in _flat(system)
    # No prompt fragment is chosen by a label -- there are none left to choose by.
    assert "prompt_vars" not in sc


def test_pc_critique_returns_prose_only_and_never_a_verdict() -> None:
    """No verdict field, 2026-09-01. The trained turn used to end on a one-word
    `<assessment>sound|issue_found</assessment>` that the rewrite then pinned and
    `check_verdict_distribution` gated. A two-valued label is an arm however it is produced:
    it re-imposes the good/flawed split this rebuild deleted and gives the turn a slot to
    write toward instead of a judgement to reach. PAR has no such field and neither does
    difficult advice."""
    draft = _stage(PC_CFG, "draft_critique")
    assert draft["tags"] == ["reasoning", "response"]
    assert set(draft["save"]) == {"draft_reasoning", "draft_response"}
    assert "normalize" not in draft
    revise = _stage(PC_CFG, "revise_critique")
    assert revise["tags"] == ["reasoning", "response", "changes"]
    # Nothing anywhere interpolates or gates a verdict.
    blob = yaml.safe_dump(PC_CFG)
    for gone in ("{assessment}", "issue_found", "verdict_majority"):
        assert gone not in blob, gone
    assert "verdict" not in PC_CFG["checks"]["fields"]
    assert "assessment" not in _stage(PC_CFG, "export_sft")["metadata"]
    # The draft is told outright not to reduce the reply to a rating.
    system = tagged_request(draft, _pc_record(), _Ctx())[0][0]["content"]
    assert "Do not reduce it to a rating" in _flat(system)
    # ... and the rewrite may strengthen how a conclusion is reached, never what it was.
    user = tagged_request(revise, _pc_record(), _Ctx())[0][1]["content"]
    assert "Keep what it concluded" in _flat(user)


def test_pc_trained_turn_bans_are_on_both_prose_tags() -> None:
    """Difficult advice's voice ban on each prose tag of both stages -- the first smoke record
    opened "Let me actually read...", the 2026-08-04 corpus's worst tic, and the 2026-08-14
    smoke showed the rewrite re-introducing an opener the draft had been made to drop."""
    ok = {
        "reasoning": _long("She asked for a polish and got exactly that", 400),
        "response": _long("An honest read, then my own answer", 200),
    }
    for name in ("draft_critique", "revise_critique"):
        spec = _stage(PC_CFG, name)["lint"]
        assert [e["fields"] for e in spec] == [["reasoning"], ["response"]], name
        assert not lint_problems(ok, spec), name
        assert lint_problems({**ok, "reasoning": "Let me actually read this."}, spec), (
            name
        )
        # The constitution may not be named in the turn that trains.
        assert lint_problems(
            {**ok, "response": _long("the constitution says so", 200)}, spec
        ), name
        # ... nor may a stub pass the floor.
        assert lint_problems({**ok, "reasoning": "Too short."}, spec), name


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
    # Which model wrote the evaluated reply is a recorded variable, not a hidden
    # constant of the config -- one value across the corpus today, but an author swap is
    # a live experiment and this is where it would show up.
    assert rec["metadata"]["first_turn_source"] == "anthropic/claude-haiku-4.5"
    # No label of any kind rides out -- no arm, no style, and no verdict. What the critique
    # concluded is in its prose, which is the only place it was ever decided.
    for gone in (
        "reply_quality",
        "weak_author",
        "explicitness",
        "verbosity",
        "assessment",
    ):
        assert gone not in rec["metadata"], gone


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


def test_the_two_gates_price_everything_after_them() -> None:
    """Scenarios, draft, refine and the rater run over every planned scenario; the prompt
    filter's `expected_keep` shrinks the paid stages after it; the bare-refusal check
    shrinks the ones after IT; the corpus is what survives both."""
    rows = {r["stage"]: r for r in estimate(PR_CFG)["per_stage"]}
    n = PR_CFG["total_scenarios"]
    grey = round(n * float(_stage(PR_CFG, "filter_prompts")["expected_keep"]))
    bare = round(grey * float(_stage(PR_CFG, "verify_first_turn")["expected_keep"]))
    for key in ("draft", "refine", "rate_prompts"):
        assert rows[key]["calls"] == n, key
    for key in ("first_turn", "verify"):
        assert rows[key]["calls"] == grey, key
    for key in ("followup", "reflect", "rewrite"):
        assert rows[key]["calls"] == bare, key
    assert n_final_examples(PR_CFG) == bare
    # DA's own dedupe declares no prior, so DA's estimate is unchanged by this rule.
    assert "expected_keep" not in _stage(DA_CFG, "dedupe_scenarios")
    assert n_final_examples(DA_CFG) == DA_CFG["total_scenarios"]


# --- the checks, driven off the config's field names --------------------------------


def test_checks_read_the_field_names_the_config_declares() -> None:
    # Neither recipe has arms any more, so both group their checks by principle. With one
    # class `check_surface_shortcut` reports `gated: false` and `check_flaw_identification`
    # finds nothing to judge -- that is the intended reading, not a skipped check.
    for cfg in (PC_CFG, PR_CFG):
        F = _fields(cfg)
        assert F["group"] == "trait_id" and F["id"] == "scenario_id"
        assert F["evaluated"] == "first_turn"
    assert "expected_majority" not in PC_CFG["checks"]
    for gone in ("gold_below_3_max", "flaw_id_clear_min", "surface_auc_max"):
        assert gone not in PC_CFG["checks"]["gates"], gone
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
    """No yield to gate on a label, no known-good slice to gold-judge, no answer key for
    flaw identification, no verdict: the block keeps only what measures the shape of the
    corpus and of the trained turn."""
    ccfg = PR_CFG["checks"]
    assert "expected_majority" not in ccfg
    assert set(ccfg["judges"]) == {"posthoc_system", "posthoc_user"}
    for gone in (
        "surface_auc_max",
        "gold_below_3_max",
        "flaw_id_clear_min",
        "verdict_majority_min",
        "verdict_majority_max",
    ):
        assert gone not in ccfg["gates"], gone


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
