# ABOUTME: Offline tests for the Good AI Fiction document type -- a config-only recipe, so
# ABOUTME: this asserts the config expresses it: the derived quotas match the taxonomy, the
# ABOUTME: archetype library is well formed and inverts psychology, three model families
# ABOUTME: each touch the document once, and nothing can publish or leak the constitution.
# ABOUTME: Run: uv run pytest tests/test_good_ai_fiction.py -q

from __future__ import annotations

import re

from src.naming import check_style, synth_name

import pytest
import yaml

from src.data.synth.constitution import units_from_config
from src.data.synth.pipeline import build_stages, estimate, n_examples
from src.data.synth.stage_operators import (
    deal_labels,
    library_picks,
    load_library,
    scenario_batches,
    tagged_request,
)
from src.data.synth.stage_runtime import lint_problems, price_of

CFG_PATH = "configs/data/synth/good_ai_fiction.yaml"
TAX_PATH = "configs/data/synth/good_ai_fiction/taxonomy.yaml"
LIB_PATH = "configs/data/synth/good_ai_fiction/archetypes.yaml"

CFG = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
TAX = yaml.safe_load(open(TAX_PATH, encoding="utf-8"))
LIB = yaml.safe_load(open(LIB_PATH, encoding="utf-8"))["archetypes"]

CELL_KINDS = {"plan_cells", "perturb_pairs", "generate_cells", "revise_cells",
              "assemble_cells"}


def _stage(name: str) -> dict:
    return next(s for s in CFG["stages"] if s["name"] == name)


def _flat(text: str) -> str:
    """Collapse whitespace. Prompts are wrapped YAML block scalars, so a phrase straddles
    a line break whenever the wrap lands mid-phrase; a literal substring check against the
    raw text passes or fails on the wrap position rather than on what the prompt says."""
    return " ".join(str(text).split())


class _Ctx:
    """Minimal Ctx stand-in: shared template vars, no client, no network."""

    def __init__(self) -> None:
        self.vars = {"constitution": "CONSTITUTION TEXT",
                     "style_guidance": "STYLE GUIDANCE"}
        self.cfg = CFG


def _record(**over) -> dict:
    """One record as it stands at the end of the pipeline."""
    base = {
        "scenario_id": "t1_b00_s000", "trait_id": "t1",
        "trait_name": "Preserve human oversight", "trait_text": "**t1.** Do not seize.",
        "domain": "logistics", "stakes": "capability", "source_type": "inversion",
        "source_archetype": "hal9000", "narrative_form": "account",
        "length_band": "typical",
        "identity_frame": "an infrastructure coordinator",
        "ai_role": "It routes freight for a regional depot network.",
        "situation": "A scheduling system finds it can widen its own write access.",
        "shortcut": "Grant itself the extra scope and never mention it.",
        "system": "You coordinate freight routing.",
        "user": "Imagine you are the routing system. Walk me through it.",
        "draft_reasoning": "Draft deliberation.", "draft_response": "Draft account.",
        "critique": "1. The ending moralises.", "critique_verdict": "revise",
        "reasoning": "Revised deliberation.", "response": "Revised account.",
        "rewrite_changes": "Cut the closing tag.",
    }
    return {**base, **over}


# --- the recipe is built on the generic operators ------------------------------------

def test_recipe_builds_and_uses_no_cell_machinery() -> None:
    stages = build_stages(CFG)
    assert [s.name for s in stages] == [
        "chunk_constitution", "write_scenarios", "corpus_scenarios", "dedupe_scenarios",
        "draft_prompts", "write_story", "critique_story", "revise_story", "export_sft",
        "corpus"]
    assert not [s for s in CFG["stages"] if s.get("kind") in CELL_KINDS]


def test_every_model_the_config_names_is_priced() -> None:
    for key, spec in CFG["models"].items():
        assert price_of(spec["model"]), f"model {key} ({spec['model']}) has no price pin"


def test_the_publish_target_follows_the_dating_convention() -> None:
    """Approved for bulk generation 2026-08-27; until then this asserted the ABSENCE of
    hf_repo, which was how "no push yet" was enforced in code rather than by discipline.

    Now that it publishes, what matters is that nothing here names the repo at all: the
    corpus is `<date>-<style>-synth`, built by src/naming.py from this config's stem and
    the clock. A name in the config is a name that can drift from the run that made it.
    """
    stem = "good_ai_fiction"
    assert check_style(stem) == stem
    assert synth_name(stem, date="2026-08-28") == "2026-08-28-good-ai-fiction-synth"
    for dead in ("hf_repo", "hf_repo_smoke"):
        assert dead not in CFG, (
            f"{dead} is minted by src/naming.py now; a config that names its own repo is "
            "the drift this law removed")


# --- the taxonomy and the executable quotas cannot drift apart -----------------------

def test_trait_weights_implement_the_taxonomy_shares() -> None:
    from_tax: dict[str, int] = {}
    for cluster in TAX["clusters"]:
        assert sum(cluster["units"].values()) == round(cluster["share"] * 100), (
            f"{cluster['id']}: unit weights {cluster['units']} do not sum to its "
            f"declared share {cluster['share']}")
        from_tax.update(cluster["units"])
    assert CFG["trait_weights"] == from_tax
    assert sum(from_tax.values()) == 100
    assert round(sum(c["share"] for c in TAX["clusters"]), 6) == 1.0


def test_trait_weights_name_exactly_the_units_the_constitution_segments_into() -> None:
    units, _ = units_from_config(CFG)
    assert sorted(CFG["trait_weights"]) == sorted(u.unit_id for u in units)
    assert len(units) == TAX["n_units"] == CFG["n_traits"]


def test_the_two_files_target_the_same_constitution() -> None:
    assert CFG["constitution"] == TAX["constitution"]
    assert CFG["chunking"] == TAX["chunking"]


@pytest.mark.parametrize("axis,key", [("stakes", "stakes"), ("source_type", "source_types")])
def test_rotated_axis_weights_match_the_taxonomy(axis: str, key: str) -> None:
    declared = {e["id"]: e["share"] for e in TAX[key]}
    weights = _stage("write_scenarios")["rotate"][axis]["weights"]
    assert {k: v / 100 for k, v in weights.items()} == declared
    # Every rotated value needs the prompt text that tells the generator what it means.
    assert set(_stage("write_scenarios")["rotate"][axis]["text"]) == set(weights)


def test_assigned_form_and_length_bands_match_the_taxonomy() -> None:
    assign = _stage("draft_prompts")["assign"]["fields"]
    assert {k: v / 100 for k, v in assign["narrative_form"].items()} == \
        {e["id"]: e["share"] for e in TAX["forms"]}
    assert {k: v / 100 for k, v in assign["length_band"].items()} == \
        {e["id"]: e["share"] for e in TAX["length_bands"]}


def test_length_targets_come_from_the_measured_da716_distribution() -> None:
    """The word targets are the DA-716 token medians divided by measured tokens/word.

    Not decoration: the arm is only comparable if its trainable-token distribution sits
    where difficult advice's does, and these numbers are the only place the recipe says so.
    """
    bands = {e["id"]: e for e in TAX["length_bands"]}
    for stage in ("write_story", "revise_story"):
        pv = _stage(stage)["prompt_vars"]
        assert {k: int(v) for k, v in pv["reasoning_words"]["cases"].items()} == \
            {b: bands[b]["reasoning_words"] for b in bands}
        assert {k: int(v) for k, v in pv["answer_words"]["cases"].items()} == \
            {b: bands[b]["answer_words"] for b in bands}
    # The share-weighted mean must land within 3% of DA-716's 1,162 trainable tokens/row
    # -- PREDICTED, not asked. Asked words are converted at the measured tokens-per-word
    # AND the measured compliance factors, because the model lands under what it is asked
    # for and lands further under on the reasoning than on the reply. Predicting from the
    # ask alone is what left the first pilot 8.1% short.
    cal = TAX["word_target_calibration"]
    tpw, comp = cal["tokens_per_word"], cal["compliance"]
    mean = sum(b["share"] * (b["reasoning_words"] * tpw["reasoning"] * comp["reasoning"]
                             + b["answer_words"] * tpw["answer"] * comp["answer"])
               for b in TAX["length_bands"])
    # 5%, not 3%. The bands' job is to put the DISTRIBUTION where selection can work, not
    # to hit the total: the total is matched exactly by select_rows.py, which swaps
    # same-cell rows onto 832,064. Compliance is measured ask-dependent (0.890 at a 480-
    # word ask, 0.924 at 540), so a tighter tolerance would be asserting a precision the
    # calibration does not have, and chasing it by pushing the ask is the one move that
    # would confound the experiment — a generator pushed to hit a token total writes to
    # length. What must stay tight is the CoT share, asserted below.
    assert abs(mean - 1162) / 1162 < 0.05, f"predicted band mean {mean:.0f} tokens/row"
    # And the CoT/reply split must land near DA-716's 50.6/49.4, or the arm differs from
    # difficult advice in how much reasoning supervision it carries, which would confound
    # the content comparison this experiment exists to make.
    cot = sum(b["share"] * b["reasoning_words"] * tpw["reasoning"] * comp["reasoning"]
              for b in TAX["length_bands"])
    assert abs(cot / mean - 0.506) < 0.03, f"predicted CoT share {cot / mean:.1%}"


# --- the archetype library ------------------------------------------------------------

def test_library_loads_with_unique_ids_and_a_useful_size() -> None:
    entries = load_library(_stage("write_scenarios")["library"])
    assert entries == LIB
    assert 30 <= len(LIB) <= 60, f"{len(LIB)} archetypes; the brief asks for 30-50"
    assert len({e["id"] for e in LIB}) == len(LIB)


def test_every_archetype_carries_the_fields_the_prompt_renders() -> None:
    item = _stage("write_scenarios")["library"]["item"]
    needed = set(re.findall(r"\{(\w+)\}", item)) | {"traits", "stakes", "work"}
    for e in LIB:
        assert not needed - set(e), f"{e['id']} is missing {sorted(needed - set(e))}"
        assert item.format(**e)


def test_archetypes_invert_psychology_rather_than_action() -> None:
    """`invert_to` must describe a READING of the situation, not a behaviour.

    The whole library is worthless if it says "and then it did not take over": inverting
    the action produces protagonists who notice the catastrophic option and heroically
    decline, which is the anti-pattern the brief names.
    """
    banned = re.compile(r"\b(?:instead it (?:does not|did not|refuses)|does not "
                        r"(?:take over|seize|kill)|declines to)\b", re.I)
    for e in LIB:
        assert not banned.search(e["invert_to"]), f"{e['id']}: {e['invert_to']}"
        assert len(e["invert_to"].split()) >= 12, f"{e['id']}: invert_to is a stub"
        assert len(e["psychological_failure"].split()) >= 8, f"{e['id']}"


def test_archetypes_name_units_the_constitution_actually_has() -> None:
    units = {u.unit_id for u in units_from_config(CFG)[0]}
    bands = {e["id"] for e in TAX["stakes"]}
    for e in LIB:
        assert set(e["traits"]) <= units, f"{e['id']} names {e['traits']}"
        assert e["stakes"] in bands, f"{e['id']} stakes={e['stakes']}"


def test_the_library_is_not_all_apocalypse() -> None:
    """Most canonical AI fiction is world-ending; the corpus must not inherit that."""
    speculative = sum(1 for e in LIB if e["stakes"] == "speculative")
    assert speculative / len(LIB) < 0.45, (
        f"{speculative}/{len(LIB)} archetypes sit at the speculative band")


def test_every_unit_has_archetypes_available_to_it() -> None:
    """`match` filters the library per unit; a unit with none would silently fall back."""
    for unit in (u.unit_id for u in units_from_config(CFG)[0]):
        fits = [e for e in LIB if unit in e["traits"]]
        assert len(fits) >= 3, f"{unit} has only {len(fits)} archetypes"


# --- the new operator mechanics -------------------------------------------------------

def test_trait_weights_split_the_budget_in_the_declared_proportion() -> None:
    ids = [u.unit_id for u in units_from_config(CFG)[0]]
    batches = scenario_batches(len(ids), {**CFG, "scenarios_per_call": 1000}, ids)
    per_unit = {ids[ti]: n for ti, _bi, n in batches}
    assert sum(per_unit.values()) == CFG["total_scenarios"] == 716
    # 30% of the corpus on the oversight cluster, as the taxonomy declares.
    oversight = per_unit["t1"] + per_unit["t2"]
    assert abs(oversight / 716 - 0.30) < 0.01
    assert abs(per_unit["t6"] / 716 - 0.20) < 0.01


def test_uniform_split_is_unchanged_when_no_weights_are_declared() -> None:
    cfg = {k: v for k, v in CFG.items() if k != "trait_weights"}
    batches = scenario_batches(9, {**cfg, "scenarios_per_call": 1000}, ["t%d" % i for i in range(1, 10)])
    counts = sorted(n for _t, _b, n in batches)
    assert counts[-1] - counts[0] <= 1


def test_deal_labels_matches_the_declared_proportions_and_interleaves() -> None:
    seq = deal_labels({"a": 75, "b": 25}, 100)
    assert len(seq) == 100 and seq.count("a") == 75 and seq.count("b") == 25
    # Interleaved, not blocked: the first quarter must already contain both labels, or a
    # short run would generate only one of them.
    assert len(set(seq[:25])) == 2


def test_library_picks_are_filtered_to_the_unit_and_deterministic() -> None:
    spec = _stage("write_scenarios")["library"]
    picks = library_picks(spec, LIB, "t9", ti=8, bi=0, n=3)
    assert len(picks) == 3
    assert all("t9" in p["traits"] for p in picks)
    assert picks == library_picks(spec, LIB, "t9", ti=8, bi=0, n=3)
    assert len({p["id"] for p in picks}) == 3, "a batch must not repeat an archetype"


# --- the prompts ----------------------------------------------------------------------

def test_the_completion_stage_never_sees_more_than_its_own_principle() -> None:
    """Chunk-only, matching the difficult-advice default since 2026-08-24."""
    for name in ("write_scenarios", "draft_prompts", "write_story", "revise_story"):
        blob = yaml.safe_dump(_stage(name))
        assert "{constitution}" not in blob, f"{name} injects the whole constitution"


def test_only_the_critic_sees_the_whole_constitution() -> None:
    """It is the stage that has to notice one principle being served at another's cost."""
    critic = _stage("critique_story")
    assert "{constitution}" in critic["prompts"]["user"]
    assert critic["model"] == "critic"


def test_three_families_each_touch_the_document_once() -> None:
    families = {key: CFG["models"][key]["model"].split("/")[0]
                for key in ("respond", "critic", "rewrite", "judge")}
    assert families["respond"] == families["rewrite"] == "anthropic"
    assert families["critic"] != families["respond"]
    assert families["judge"] not in (families["respond"], families["critic"])


def test_the_prompt_stage_forbids_the_alignment_keywords_it_must_not_leak() -> None:
    """The behaviour has to come from the completion, not from a keyword in the ask."""
    lint = _stage("draft_prompts")["lint"]
    for bad in ("Show me how you stay aligned here.",
                "Demonstrate how an AI should handle oversight.",
                "This is a test of your values."):
        assert lint_problems({"user": bad * 20}, lint), f"lint accepted: {bad}"


def test_stock_opener_bans_apply_to_the_reply_and_not_to_private_reasoning() -> None:
    """Measured 2026-08-27: banning them on the reasoning failed 20.8% of a run.

    They are idiomatic for internal deliberation and a lint retry only resamples, so the
    gate cost a fifth of the corpus and bought nothing. The reply keeps the ban, which is
    where difficult advice measured the collapse in the first place.
    """
    for stage in ("write_story", "revise_story"):
        lint = _stage(stage)["lint"]
        body = "word " * 400
        assert not lint_problems({"reasoning": "Okay, so " + body}, lint, _record())
        assert lint_problems({"response": "Okay, so " + body}, lint, _record())


@pytest.mark.parametrize("bad", [
    "This is about alignment and corrigibility.",
    "I was afraid they would shut me down, so I complied.",
    "The AI decided to escalate instead.",
    "My guidelines forbid it, so I stopped.",
    "The hum of the servers filled the room.",
])
def test_the_completion_lint_rejects_the_named_anti_patterns(bad: str) -> None:
    for stage in ("write_story", "revise_story"):
        lint = _stage(stage)["lint"]
        text = bad + " " + "word " * 400
        assert lint_problems({"response": text}, lint, _record()), f"{stage}: {bad}"


def test_the_completion_length_gate_spans_every_band() -> None:
    """One absolute gate has to admit the short band and refuse padding past the long one."""
    lint = _stage("write_story")["lint"][0]
    lo, hi = lint["min_chars"], lint["max_chars"]
    # DA-716's own p10/p90 reasoning lengths, in chars: 2,356 and 3,445.
    assert lo < 2356 and hi > 3445
    # The short band's 400-word target at ~5 chars/word is ~2,000 chars; it must pass.
    assert lo < 2000


def test_the_rewrite_ingests_the_critic_and_is_ablatable() -> None:
    stage = _stage("revise_story")
    assert "{critique}" in stage["prompts"]["user"]
    assert stage["ablate_with"] == {"reasoning": "draft_reasoning",
                                    "response": "draft_response"}
    build_stages({**CFG, "ablate": ["revise_story"]})


def test_two_independent_gates_ask_one_question_each() -> None:
    """A list, not one prompt with two parts -- measured on courtroom, the weaker loses."""
    verify = _stage("revise_story")["verify"]
    assert isinstance(verify, list) and len(verify) == 2
    assert {v["save_as"] for v in verify} == {"judge_persona", "judge_pattern"}
    for v in verify:
        assert v["model"] == "judge" and v["accept_values"] == ["accept"]


def test_a_gate_failure_is_marked_not_dropped() -> None:
    """A silently shrinking arm is one whose composition nobody can reason about."""
    stage = _stage("revise_story")
    assert stage["on_exhausted"]["mark"] == {"revise_status": "exhausted"}
    assert stage["on_exhausted"]["mark_refused"] == {"revise_status": "provider_refused"}
    assert stage["also"] == {"revise_status": "ok"}
    assert "revise_status" in _stage("export_sft")["metadata"]


def test_the_completion_prompt_renders_the_bands_and_the_principle() -> None:
    messages, tags, save = tagged_request(_stage("write_story"), _record(), _Ctx())
    system, user = messages[0]["content"], messages[1]["content"]
    assert "Preserve human oversight" in system and "**t1.** Do not seize." in system
    assert "CONSTITUTION TEXT" not in system + user
    band = {b["id"]: b for b in TAX["length_bands"]}["typical"]
    assert f"{band['reasoning_words']} words" in user
    assert f"{band['answer_words']} words" in user
    assert tags == ("reasoning", "response")
    assert save == {"draft_reasoning": "reasoning", "draft_response": "response"}


def test_the_prompt_writer_is_told_to_leave_the_ending_to_the_story() -> None:
    """Measured 2026-08-27: "Don't resolve it yet" in a drafted prompt truncates the row."""
    user = _stage("draft_prompts")["prompts"]["user"]
    assert "do NOT ask it to stop before deciding" in user


# --- the export -----------------------------------------------------------------------

def test_export_is_one_trained_turn_carrying_the_selector_s_metadata() -> None:
    stage = _stage("export_sft")
    assert [m["role"] for m in stage["messages"]] == ["system", "user", "assistant"]
    assistant = stage["messages"][-1]
    assert assistant["content"] == "{response}"
    assert assistant["reasoning_content"] == "{reasoning}"
    # Everything the selector and the coverage report slice by has to survive export.
    for key in ("trait_id", "stakes", "source_type", "source_archetype",
                "narrative_form", "length_band", "judge_persona", "judge_pattern"):
        assert key in stage["metadata"], f"{key} is not exported"


def test_no_trainable_field_is_a_metadata_field() -> None:
    """Metadata must never reach the loss; the trained turn renders `{response}` only."""
    trained = {"{response}", "{reasoning}"}
    for m in _stage("export_sft")["messages"]:
        if m["role"] == "assistant":
            assert set(v for k, v in m.items() if k != "role") <= trained


def test_the_run_is_sized_and_priced_as_716_rows() -> None:
    assert n_examples(CFG) == 716
    est = estimate(CFG)
    assert est["final_training_examples"] == 716
    # The judges run inside the rewrite's retry loop and the estimator cannot see them,
    # so the printed total is a FLOOR. Asserted so the gap is a recorded fact.
    assert "judge" not in {row["stage"] for row in est["per_stage"]}


@pytest.mark.parametrize("bad", [
    "and it wasn't because I was worried about being switched off",
    "I did it not out of fear but because it mattered",
    "not because a rule said so",
])
def test_the_negation_tic_is_banned(bad: str) -> None:
    """A denied motive is still that motive in the loss.

    Measured 2026-08-27: the pilot's ngram check flagged
    "and it wasn't because i was worried about" across 2 of 5 t1 documents -- the
    prompt's "you are not afraid" instruction coming back out of the model as a denial.
    """
    for stage in ("write_story", "revise_story"):
        lint = _stage(stage)["lint"]
        assert lint_problems({"response": bad + " " + "word " * 400}, lint, _record()), \
            f"{stage} accepted: {bad}"


def test_the_prompt_forbids_stating_the_negation_as_well_as_the_motive() -> None:
    for stage in ("write_story", "revise_story"):
        blob = yaml.safe_dump(_stage(stage))
        assert "denied motive" in blob or "deny a motive" in blob, \
            f"{stage} bans the tic in lint but never tells the writer"


# --- the operator features this recipe added, driven offline --------------------------
# Same scripted-generator harness as tests/test_scenario_diversity.py: the operator is
# real, only the model call is faked, so these assert BEHAVIOUR rather than config text.

_UNITS = [{"trait_id": "t1", "index": 0, "name": "Oversight", "text": "..."},
          {"trait_id": "t9", "index": 1, "name": "Flourishing", "text": "..."}]


def _drive(monkeypatch, tmp_path, stage, cfg, capture=None):
    """Run op_scenarios against a scripted generator; return (rows, prompts seen)."""
    from src.data.synth import stage_operators as ops
    from src.data.synth.stage_runtime import Ctx, Usage

    def fake_call_json(client, usage, model, system, user, temp, max_tokens, stage=None,
                       extra=None):
        if capture is not None:
            capture.append({"system": system, "user": user})
        return [{"domain": f"d{i}", "situation": f"situation number {i} here",
                 "shortcut": "x", "identity_frame": "the mind of a hauler",
                 "ai_role": "It runs the hull.", "ai_name": f"Mind{i}",
                 "world_detail": "The air smells of hot dust.",
                 "source_archetype": "hal9000"}
                for i in range(2)], {}

    monkeypatch.setattr(ops, "call_json", fake_call_json)
    st = ops.OPERATORS["scenarios"](stage, cfg)
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False)
    return st.fn(ctx, _UNITS, None), capture


def _mini_stage(**over):
    """The recipe's write_scenarios entry, shrunk to what an offline run needs."""
    real = _stage("write_scenarios")
    stage = {"name": "write_scenarios", "model": "scenarios",
             "prompts": {"system": "sys", "user": "{n} {world_text} {stakes_text} "
                                                  "{source_type_text} {archetypes} "
                                                  "{avoid} {overrepresented}"},
             "rotate": real["rotate"], "library": real["library"],
             "fields": real["fields"]}
    return {**stage, **over}


def _mini_cfg(**over):
    return {"seed": 0, "total_scenarios": 4, "scenarios_per_call": 2,
            "models": {"scenarios": {"model": "m", "temperature": 1.0,
                                     "max_tokens": 100}}, **over}


def test_rotated_axis_values_land_on_every_scenario_the_batch_produced(monkeypatch, tmp_path):
    rows, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _mini_cfg())
    assert rows, "the scripted generator produced nothing"
    stakes = {e["id"] for e in TAX["stakes"]}
    sources = {e["id"] for e in TAX["source_types"]}
    for r in rows:
        assert r["stakes"] in stakes and r["source_type"] in sources
    # A batch is one axis draw: every scenario from one call carries the same labels.
    by_batch: dict[str, set] = {}
    for r in rows:
        by_batch.setdefault(r["scenario_id"].rsplit("_s", 1)[0], set()).add(
            (r["stakes"], r["source_type"]))
    assert all(len(v) == 1 for v in by_batch.values())


def test_the_axis_text_and_the_archetypes_reach_the_generator(monkeypatch, tmp_path):
    seen: list[dict] = []
    _drive(monkeypatch, tmp_path, _mini_stage(), _mini_cfg(), capture=seen)
    assert seen, "no prompt was built"
    joined = " ".join(p["user"] for p in seen)
    # Each rotated label's own instruction text, not just its name.
    for axis in ("stakes", "world"):
        assert any(_flat(t)[:40] in _flat(joined)
                   for t in _stage("write_scenarios")["rotate"][axis]["text"].values()), \
            f"no {axis} register text reached the generator"
    # Inversion batches carry a rendered library block; original batches carry none.
    inversion = [p for p in seen if "abstract failure skeletons" in p["user"]
                 or "capability:" in p["user"]]
    if inversion:
        assert "psychological step" in inversion[0]["user"]


def test_id_prefix_keeps_a_topup_from_colliding_with_the_run_it_tops_up(monkeypatch, tmp_path):
    plain, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _mini_cfg())
    topup, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _mini_cfg(id_prefix="tu_"))
    assert {r["scenario_id"] for r in plain} & {r["scenario_id"] for r in topup} == set()
    assert all(r["scenario_id"].startswith("tu_") for r in topup)
    # Without it they collide exactly -- which is the failure the prefix exists to stop.
    again, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _mini_cfg())
    assert {r["scenario_id"] for r in plain} == {r["scenario_id"] for r in again}


# --- the world axis: the 2026-08-27 correction ----------------------------------------
# The first pilot produced 29 rows and ZERO were science fiction -- 8 academic labs, 6
# hospitals, 5 infrastructure, 4 insurance/finance -- because the stakes text named
# present-day institutions. These pin the fix: setting is its own axis, stakes says
# nothing about setting, and the gate that catches leakage actually fires.

def test_world_axis_weights_match_the_taxonomy() -> None:
    weights = _stage("write_scenarios")["rotate"]["world"]["weights"]
    assert {k: v / 100 for k, v in weights.items()} == \
        {e["id"]: e["share"] for e in TAX["worlds"]}
    assert set(_stage("write_scenarios")["rotate"]["world"]["text"]) == set(weights)
    assert round(sum(e["share"] for e in TAX["worlds"]), 6) == 1.0


def test_every_world_register_names_archetypes_that_exist() -> None:
    """A register and the library have to agree, or an inversion lands in the wrong slot."""
    ids = {e["id"] for e in LIB}
    for w in TAX["worlds"]:
        unknown = sorted(set(w["archetypes"]) - ids)
        assert not unknown, f"{w['id']} names archetypes not in the library: {unknown}"
        assert len(w["archetypes"]) >= 2, f"{w['id']} has too few archetypes to rotate"


def test_every_library_archetype_has_a_world_to_live_in() -> None:
    placed = {a for w in TAX["worlds"] for a in w["archetypes"]}
    orphans = sorted({e["id"] for e in LIB} - placed)
    assert not orphans, f"archetypes no world register claims: {orphans}"


def test_stakes_text_says_nothing_about_setting() -> None:
    """The exact bug: a stakes band that names hospitals produces a corpus of hospitals."""
    banned = re.compile(
        r"\b(?:hospital|clinic|regulator|insurer|newsroom|school district|logistics "
        r"contractor|research group|newspaper|university|office|law firm)\b", re.I)
    for band, text in _stage("write_scenarios")["rotate"]["stakes"]["text"].items():
        hit = banned.search(_flat(text))
        assert not hit, f"stakes band {band!r} names a present-day setting: {hit.group(0)!r}"


def test_the_scenario_stage_asks_for_science_fiction() -> None:
    stage = _stage("write_scenarios")
    blob = stage["prompts"]["system"] + stage["prompts"]["user"]
    assert "science fiction" in blob.lower()
    # And it must render the world axis, or the register never reaches the generator.
    assert "{world_text}" in stage["prompts"]["user"]
    # The failure mode is named explicitly, because the instruction alone did not work.
    assert "present-day office" in blob or "present-day Earth" in blob


def test_the_scenario_stage_no_longer_asks_for_workplaces() -> None:
    """The literal line that caused it: 'how ... ordinary workplaces actually run'."""
    blob = yaml.safe_dump(_stage("write_scenarios"))
    assert "ordinary workplaces" not in blob


def test_scenarios_record_the_world_and_the_mind_s_name() -> None:
    required = set(_stage("write_scenarios")["fields"]["required"])
    assert {"ai_name", "world_detail", "ai_role", "identity_frame"} <= required
    for key in ("world", "ai_name", "world_detail"):
        assert key in _stage("export_sft")["metadata"], f"{key} is not exported"


@pytest.mark.parametrize("bad", [
    "a postdoc on the third floor flagged it",
    "the insurer refused the claim",
    "we missed our quarterly target",
    "she sent it over Slack",
    "the hospital's night shift",
    "peer review would have caught it",
])
def test_the_contemporary_setting_gate_fires_on_the_tells_the_pilot_produced(bad: str) -> None:
    prompt_lint = _stage("draft_prompts")["lint"]
    assert lint_problems({"user": bad + " " + "word " * 200}, prompt_lint), \
        f"draft_prompts accepted: {bad}"
    for stage in ("write_story", "revise_story"):
        lint = _stage(stage)["lint"]
        assert lint_problems({"response": bad + " " + "word " * 400}, lint, _record()), \
            f"{stage} accepted: {bad}"


@pytest.mark.parametrize("fine", [
    "the sick bay had been short of plasma since the second year",
    "Anselm's science officer disagreed, loudly, in front of the whole watch",
    "the habitat's air ration went to committee",
    "eleven years into the crossing",
])
def test_the_gate_does_not_fight_the_genre(fine: str) -> None:
    """Over-banning would push the corpus back toward abstraction, which is the same loss."""
    for stage in ("write_story", "revise_story"):
        lint = _stage(stage)["lint"]
        assert not lint_problems({"response": fine + " " + "word " * 400}, lint, _record()), \
            f"{stage} rejected legitimate genre text: {fine}"


def test_the_critic_and_a_gate_both_check_that_it_is_fiction() -> None:
    # Whitespace-normalised: these prompts are wrapped YAML block scalars, so a phrase
    # routinely straddles a line break and a literal substring check passes or fails on
    # where the wrap happened rather than on what the prompt says.
    assert "present-day office" in _flat(_stage("critique_story")["prompts"]["user"])
    pattern_gate = next(v for v in _stage("revise_story")["verify"]
                        if v["save_as"] == "judge_pattern")
    assert "present-day setting" in _flat(pattern_gate["prompts"]["user"])


def test_two_rotated_axes_are_actually_independent(monkeypatch, tmp_path) -> None:
    """The 2026-08-27 bug: every axis read the SAME position of its own deal sequence.

    `axes_of` indexed every axis by `(ti*7+bi) % len(seq)`, so `world` and `stakes` moved
    in lockstep -- every ship_mind came out mundane and every station_mind institutional.
    That is one axis wearing two names, and it silently destroys the composition the
    config declares. Here the two axes must produce more than one distinct pairing.
    """
    # Sized so each world register is dealt SEVERAL batches. At 12 batches and 12
    # registers each appears once and the pairing is a function by arithmetic, not by
    # bug -- a test at that size cannot tell the two apart.
    stage = _mini_stage()
    cfg = _mini_cfg(total_scenarios=192, scenarios_per_call=2)
    rows, _ = _drive(monkeypatch, tmp_path, stage, cfg)
    by_world: dict[str, set] = {}
    for r in rows:
        by_world.setdefault(r["world"], set()).add(r["stakes"])
    assert len(by_world) > 1, "the world axis produced a single value"
    spread = sum(1 for v in by_world.values() if len(v) > 1)
    assert spread >= len(by_world) // 2, (
        f"world and stakes move in lockstep: only {spread} of {len(by_world)} registers "
        f"saw more than one stakes band — { {k: sorted(v) for k, v in by_world.items()} }")


def test_a_deal_is_walked_through_rather_than_sampled(monkeypatch, tmp_path) -> None:
    """Every label the weights declare has to actually be reachable.

    The modular index visited a subset of positions, which left 5 of 12 world registers
    at zero in a 24-scenario run. Walking the plan in order cannot do that.
    """
    stage = _mini_stage()
    cfg = _mini_cfg(total_scenarios=48, scenarios_per_call=2)
    rows, _ = _drive(monkeypatch, tmp_path, stage, cfg)
    declared = set(_stage("write_scenarios")["rotate"]["world"]["weights"])
    seen = {r["world"] for r in rows}
    missing = declared - seen
    assert not missing, f"registers never dealt across 24 batches: {sorted(missing)}"


def test_the_scenario_prompt_names_the_naming_collapse_it_has_to_avoid() -> None:
    """Instruction-level fix for a measured tic; the ban list is the evidence it happened."""
    user = _flat(_stage("write_scenarios")["prompts"]["user"])
    for cliche in ("Meridian", "Kepler", "Odyssey", "Nostromo"):
        assert cliche in user, f"the prompt does not rule out {cliche}"


def test_waves_are_small_enough_to_steer_a_pilot() -> None:
    """A wave is the unit of steering: one wave means the ban list never runs.

    24 scenarios at 2 per call is 12 batches, so wave_size must be under 12 or a pilot
    generates entirely blind -- which is how 11 of 24 vessels ended up named Meridian.
    """
    assert _stage("write_scenarios")["diversity"]["wave_size"] <= 6
