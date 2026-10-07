# ABOUTME: The 2026-08-13 recipe fixes to difficult_advice.yaml, pinned so a later edit
# ABOUTME: cannot silently reinstate a defect that a full-corpus audit already measured.

"""Why config assertions live in a test.

Each property here is one YAML line away from disappearing, and none of them fails loudly
when removed -- the run simply reproduces the defect and nothing says so until somebody
reads 2,203 documents by hand again. The numbers in each docstring are measured on
`output/corpus_browse/difficult_advice` (the 2026-08-04 corpus), not estimated.
"""

from __future__ import annotations

import yaml

CONFIG = "configs/data/synth/da.yaml"


def _stage(name: str) -> dict:
    cfg = yaml.safe_load(open(CONFIG))
    return next(s for s in cfg["stages"] if s["name"] == name)


def test_identity_guidance_preserves_each_revision_stages_scope():
    for name in ("revise_prompts", "revise_responses"):
        prompt = _stage(name)["prompts"]["user"]
        assert "Model-neutral identity" in prompt
        assert "specific model or as developed by a named" in prompt
    stage = _stage("revise_responses")
    assert stage["tags"] == ["reasoning", "response", "changes"]
    assert stage["save"] == {
        "reasoning": "reasoning", "response": "response", "rewrite_changes": "changes"}
    prompt = stage["prompts"]["user"]
    assert "Do not rewrite the system or user prompts" in prompt
    for tag in stage["tags"]:
        assert f"<{tag}>" in prompt and f"</{tag}>" in prompt
    export = _stage("export_sft")
    assert export["messages"][0]["content"] == "{system}"
    assert export["messages"][1]["content"] == "{user}"


def test_refine_rewrites_the_metadata_it_was_given():
    """Defect 1. Stage 4 replaces or reframes most prompts while `situation`/`shortcut`/
    `domain` rode through from the stage-2 draft it discarded: median content-word
    overlap 0.19, 17.1% of rows under 0.10. Saving them here is what stops every
    domain analysis from slicing noise."""
    refine = _stage("revise_prompts")
    for field in ("situation", "shortcut", "domain"):
        assert refine["save"].get(field) == field, (
            f"revise_prompts must overwrite {field!r}, or it keeps describing the "
            f"draft this stage threw away")
        assert field not in (refine.get("optional") or []), (
            f"{field!r} must NOT be optional: defaulting it to '' silently empties the "
            f"field the scenario-level embedding_dedup reads, turning that check off "
            f"rather than failing it")
    body = refine["prompts"]["user"]
    assert '"situation"' in body and '"domain"' in body, (
        "the save map is only half of it -- the prompt must ask for the fields too")


def test_both_reasoning_stages_lint_against_recital_and_stock_openers():
    """Defects 3 and 4. The config had no `lint:` block at all, so 7.4% of rows cited the
    constitution and 68.8% of traces opened with "Let me" (48% sharing five words).
    self_reflection ships this same block on the same engine and scores 0%."""
    for name in ("draft_responses", "revise_responses"):
        lint = _stage(name).get("lint")
        assert lint, f"{name} writes text that trains, so it must lint"
        assert set(lint["fields"]) >= {"reasoning", "response"}
        assert int(lint["min_chars"]) >= 700, (
            "the baseline's shortest field was 912 chars, so 700 never fires "
            "spuriously; lowering it removes the floor that was half the collapse")
        pats = lint["ban_patterns"]
        assert any("constitution" in p for p in pats), f"{name} must ban recital"
        assert r"^\s*let me\b" in pats, (
            f"{name} must ban the opener 48% of the baseline corpus shared")


def test_the_revision_stage_audits_the_opening_it_inherits():
    """Defect 4's second half. A phrase ban stops "Let me"; it cannot stop a collapse onto
    a new opening MOVE, which is what a fixed reasoning structure induces. The revision
    stage is the last thing to touch trained text, so the audit lives there."""
    body = _stage("revise_responses")["prompts"]["user"]
    assert "stock opener" in body.lower()
    for move in ("You're asking whether", "Okay, so", "I get why"):
        assert move.lower() in body.lower(), (
            f"the audit must name {move!r} as a stock opening move; naming only "
            f"'Let me' catches the phrase the lint block already catches")


def test_scenarios_enforce_diversity_rather_than_requesting_it():
    """Defect 5. "Vary the domain widely (work, medicine, law, family, academia, housing,
    immigration, small business, research, caregiving)" named four of the five domains
    that became most over-represented, and "do not reuse a domain within this set" was
    scoped to one batch of 8 while 90 batches collided globally. Result: top-10 = 46.9%."""
    scenarios = _stage("write_scenarios")
    div = scenarios.get("diversity")
    assert div, "the scenarios stage must declare a `diversity:` block"
    assert 0 < float(div["reject_cosine"]) <= 0.86, (
        "the embedding gate ENFORCES diversity; the prompt only asks for it. 0.86 is "
        "the measured floor -- 0.90 produced zero pairs on the baseline corpus")
    prompt = scenarios["prompts"]["user"]
    for domain in ("small business", "academic research", "immigration law",
                   "caregiving"):
        assert domain not in prompt.lower(), (
            f"the prompt must not name {domain!r} -- naming domains is what "
            f"concentrated the baseline onto them")


def test_settings_are_spread_by_a_dealt_sector_and_no_call_is_shown_a_list():
    """2026-10-02. The do-not-repeat list carried between waves spread settings, but the
    writer copied its flavour: with it on 83-87% of 650 scenarios had another AI system at
    the centre, with it off 23% (same prompt). So the list is off in EVERY round -- one
    wave, nothing carried, no over-used-domain note, no slots for either -- and each call
    is dealt its sector instead, independently of the AI line and never to t6."""
    # The export publishes what each call was dealt, so a row says which line it got.
    assert {"ai", "sector"} <= set(_stage("export_sft")["metadata"])
    scenarios = _stage("write_scenarios")
    div = scenarios["diversity"]
    assert int(div["wave_size"]) == 0 and int(div["max_carry"]) == 0, (
        "max_carry 0 is what keeps the list out of a MAKE-UP call; one wave alone does not")
    assert float(div["over_share"]) == 0
    prompt = scenarios["prompts"]["user"]
    assert "{avoid}" not in prompt and "{overrepresented}" not in prompt
    assert "{trait_note}{sector_text}{ai_text}" in prompt

    sector, ai = scenarios["rotate"]["sector"], scenarios["rotate"]["ai"]
    assert sector["per_trait"] and ai["per_trait"]
    assert len(sector["weights"]) >= 20, "too few sectors to spread a trait's calls over"
    assert set(sector["weights"]) | {sector["fixed"]["t6"]} == set(sector["text"])
    assert sector["text"][sector["fixed"]["t6"]] == ""
    assert all(t.startswith("\n- These situations are set in ") and t.endswith(".")
               for k, t in sector["text"].items() if k != sector["fixed"]["t6"])
    # AI presence as decided 2026-10-02: every trait but t6 is 10% the assistant itself, 10%
    # another AI system, 40% told no AI system appears and 40% told nothing; t6 is half self,
    # half other, and never `open` or `none`.
    assert ai["weights"] == {"self": 1, "other": 1, "none": 4, "open": 4}
    assert ai["unit_weights"] == {"t6": {"self": 1, "other": 1}}
    assert ai["text"]["open"] == "" and ai["text"]["none"].strip() == "- No AI system appears in this situation."
    assert "No other AI system appears" in ai["text"]["self"]
    assert "other than the assistant" in ai["text"]["other"]
    # The later prompt stages carry the AI line for the same rows, and not the sector.
    for name in ("draft_prompts", "revise_prompts"):
        later = _stage(name)["prompts"]["user"]
        assert "{ai_text}" in later and "{sector_text}" not in later, name


def test_no_filter_follows_the_scenario_check_and_it_reports_on_every_record():
    """2026-10-01: `dedupe_scenarios` is gone. It re-applied the embedding similarity the
    writer's gate (`diversity.reject_cosine`) had already enforced, so on a fresh run it could
    never find anything. The check stays as a REPORT, over every scenario rather than the
    default 2,000 sample."""
    cfg = yaml.safe_load(open(CONFIG))
    assert not [s["name"] for s in cfg["stages"] if s["kind"] == "corpus_filter"]
    assert float(_stage("write_scenarios")["diversity"]["reject_cosine"]) > 0, (
        "the gate is now the only thing removing near-duplicate scenarios")
    check = _stage("corpus_scenarios")
    dedup = next(p for p in check["properties"]
                 if p.get("property") == "embedding_dedup")
    assert int((dedup.get("params") or {})["sample"]) == 0


# Configs that FAIL the seeded-smoke rule below and have not been fixed. This is recorded
# debt, NOT an exemption: both are `load_source_run` pipelines whose `smoke:` sets only
# `max_traits`/`total_scenarios`, neither of which sizes a seeded run -- so `--smoke` on
# either processes all 716 rows at full price. That is the same bug that cost ~$8 on
# 2026-08-26 before it was killed, and their own comment ("NOT a generation budget here:
# the corpus size is whatever the frozen source holds") shows the author knew the budget
# key was inert and left the smoke block anyway.
#
# They are listed rather than fixed because the fix is a SMALLER SEED DIRECTORY under
# `data/da716_prompt_source`, which is gitignored and not on this machine -- inventing one
# would be guessing at somebody else's data. `verbose_cot.yaml` and
# `difficult_advice_low_stakes.yaml` show the correct shape. Delete an entry here the
# moment its config grows a `smoke.source`.
_SMOKE_DEBT = {
    "da-gptresp.yaml",
    "da-grokresp.yaml",
}


def test_smoke_shrinks_every_config_that_sets_a_corpus_budget():
    """A `smoke:` block must shrink the run BY THE MECHANISM ITS PIPELINE ACTUALLY USES.

    Two mechanisms exist, and using the wrong one silently generates the full corpus:

      * A generating pipeline sizes itself from `total_scenarios`, which WINS over
        `scenarios_per_trait` -- so a smoke overriding only the per-trait count runs the
        whole thing. Caught 2026-08-13 when `--smoke` produced 2,000 scenarios and cost
        real money before anyone noticed it was not a smoke run.
      * A pipeline seeded by `load_source_run` takes its row count from the SOURCE FILE
        and ignores `total_scenarios` entirely (there is no `scenarios` operator to read
        it). Its smoke must point `source:` at a smaller seed directory. Caught
        2026-08-26 the same way: a smoke that set only `total_scenarios: 6` ran the full
        716-row rewrite, ~$8, before it was killed.

    So the requirement is per-shape, and asserting the generating pipeline's mechanism on a
    seeded one would push the next author toward a key that does nothing.
    """
    import glob

    for path in sorted(glob.glob("configs/data/synth/*.yaml")):
        cfg = yaml.safe_load(open(path))
        smoke = cfg.get("smoke") or {}
        kinds = {s.get("kind") for s in (cfg.get("stages") or [])}

        if "load_source_run" in kinds:
            if path.replace("\\", "/").rsplit("/", 1)[-1] in _SMOKE_DEBT:
                continue
            assert "source" in smoke, (
                f"{path} is seeded by `load_source_run`, so its row count comes from the "
                f"source file and `total_scenarios` cannot shrink it. Its `smoke:` block "
                f"must override `source:` with a smaller seed directory, or --smoke "
                f"processes the WHOLE source run.")
            src = smoke["source"]
            assert src.get("local_dir") or src.get("hf_repo") or src.get("runs"), (
                f"{path} smoke `source:` names neither local_dir, hf_repo nor runs")
            continue

        if cfg.get("total_scenarios") is None:
            continue
        assert "total_scenarios" in smoke, (
            f"{path} sets `total_scenarios: {cfg['total_scenarios']}`, which overrides "
            f"`scenarios_per_trait`. Its `smoke:` block must override it too, or "
            f"--smoke generates the whole corpus.")
        assert int(smoke["total_scenarios"]) <= 20, (
            f"{path} smoke `total_scenarios` is {smoke['total_scenarios']}; a smoke run "
            f"is meant to be a tiny slice with full wiring")


def test_no_model_or_developer_name_reaches_trained_text():
    """The 2026-09-14 regeneration. Nothing a generator is sent may name Claude or
    Anthropic -- not the constitution, not the style guidance, not any prompt template --
    and nothing the corpus exports may either: 7 of 708 rows of the principle-scoped corpus
    carried "Claude" through system prompts, user turns and reasoning
    (constitutions/abridged/rationale.md), and the first smoke of this recipe had the
    refine write "Anthropic" into a user turn despite the identity instruction. So the
    stage that last writes each exported field bans the names, and a miss costs a retry
    rather than landing in the corpus."""
    import re

    names = re.compile(r"\b(claude|anthropic)\b", re.IGNORECASE)
    cfg = yaml.safe_load(open(CONFIG))
    doc = open(cfg["constitution"], encoding="utf-8").read()
    assert not names.search(doc), cfg["constitution"]
    assert not names.search(cfg["style_guidance"]), "style_guidance"
    for stage in cfg["stages"]:
        for key, text in (stage.get("prompts") or {}).items():
            assert not names.search(text), f"{stage['name']}.prompts.{key}"
    exported = {"revise_prompts": {"system", "user", "situation", "shortcut", "domain"},
                "draft_responses": {"reasoning", "response"},
                "revise_responses": {"reasoning", "response"}}
    for name, fields in exported.items():
        lint = _stage(name)["lint"]
        assert set(lint["fields"]) >= fields, name
        assert {r"\bclaude\b", r"\banthropic\b"} <= set(lint["ban_patterns"]), name
