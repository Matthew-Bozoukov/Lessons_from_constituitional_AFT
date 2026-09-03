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
    assert float(div["over_share"]) > 0, (
        "an over-represented domain must be fed back to the generator")
    prompt = scenarios["prompts"]["user"]
    assert "{avoid}" in prompt and "{overrepresented}" in prompt
    for domain in ("small business", "academic research", "immigration law",
                   "caregiving"):
        assert domain not in prompt.lower(), (
            f"the prompt must not name {domain!r} -- naming domains is what "
            f"concentrated the baseline onto them")


def test_the_dedupe_filter_runs_on_full_coverage():
    """`corpus_filter` refuses a sampled property by design, so the check it reads must
    see every record. `embedding_dedup` defaults to sample: 2000, which would silently
    make this stage un-runnable on a 2,205-scenario corpus."""
    cfg = yaml.safe_load(open(CONFIG))
    filt = _stage("dedupe_scenarios")
    assert filt["from"] == "corpus_scenarios"
    assert "embedding_dup" in filt["drop_when"]
    assert not filt.get("keep_when"), (
        "keep-rules delete every record a failed check never labelled")
    check = _stage("corpus_scenarios")
    dedup = next(p for p in check["properties"]
                 if p.get("property") == "embedding_dedup")
    assert int((dedup.get("params") or {})["sample"]) == 0, (
        "sample: 0 means every record; anything else and corpus_filter errors")
    names = [s["name"] for s in cfg["stages"]]
    assert names.index("corpus_scenarios") < names.index("dedupe_scenarios"), (
        "the filter reads the check's sidecar, so the check must run first")


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
    "da-gptresp-716.yaml",
    "da-grokresp-716.yaml",
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
            assert src.get("local_dir") or src.get("hf_repo"), (
                f"{path} smoke `source:` names neither local_dir nor hf_repo")
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
