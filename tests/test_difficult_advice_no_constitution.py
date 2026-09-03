# ABOUTME: Offline tests for the no-constitution difficult-advice arm: `constitution: none`
# ABOUTME: yields one guideline unit, no prompt carries a trace of a document, and everything
# ABOUTME: else is pinned equal to the baseline recipe so the two differ in one thing.
# ABOUTME: Run: uv run pytest tests/test_difficult_advice_no_constitution.py -q

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.synth import pipeline
from src.data.synth.constitution import (
    GUIDELINE_UNIT_ID,
    NO_CONSTITUTION,
    document_text,
    has_constitution,
    units_from_config,
)
from src.data.synth.stage_operators import _render
from src.data.synth.stage_runtime import Ctx, Usage

CONFIG = "configs/data/synth/2026-09-03_difficult_advice_no_constitution.yaml"
BASELINE = "configs/data/synth/2026-08-01_difficult_advice.yaml"

# The words whose presence in ANY prompt would mean the document leaked back in. Checked
# case-insensitively over every prompt template and the guideline itself.
_TRACES = ("constitution", "principle", "{constitution}", "{style_guidance}")

# The stages that used to be handed the target principle, and now get the guideline.
_STANDARD_STAGES = ("write_scenarios", "revise_prompts", "draft_responses",
                    "revise_responses")


def _load(path: str) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


def _stage(cfg: dict, name: str) -> dict:
    return next(s for s in cfg["stages"] if s["name"] == name)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return _load(CONFIG)


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _load(BASELINE)


def test_none_yields_the_guideline_as_the_one_unit(cfg):
    assert cfg["constitution"] == NO_CONSTITUTION
    assert not has_constitution(cfg)
    assert document_text(cfg) == ""
    units, style = units_from_config(cfg)
    assert style == "", "there is no constitution to take a style section from"
    (u,) = units
    assert u.unit_id == GUIDELINE_UNIT_ID
    assert u.name == cfg["guideline"]["name"]
    assert u.text == cfg["guideline"]["text"].strip()
    # Provenance says `none`, never `whole`: a row from this arm must not be mistaken
    # for one generated against the entire document.
    assert u.chunk_ids == () and u.n_chunks == 0
    assert u.granularity == NO_CONSTITUTION and u.grouping_strategy == NO_CONSTITUTION
    row = u.as_dict()
    assert row["trait_id"] == GUIDELINE_UNIT_ID and row["chunk_ids"] == []


def test_none_refuses_a_chunking_and_requires_a_guideline(cfg):
    with pytest.raises(ValueError, match="no document to cut"):
        units_from_config({**cfg, "chunking": "principle"})
    with pytest.raises(ValueError, match="no document to cut"):
        units_from_config({**cfg, "only_traits": ["t1"]})
    with pytest.raises(ValueError, match="guideline"):
        units_from_config({**cfg, "guideline": None})
    with pytest.raises(ValueError, match="guideline"):
        units_from_config({**cfg, "guideline": {"name": "be good"}})
    # The key stays required: an ABSENT constitution is a config bug, not `none`.
    with pytest.raises(KeyError):
        units_from_config({k: v for k, v in cfg.items() if k != "constitution"})


def test_no_prompt_carries_a_trace_of_a_document(cfg):
    """The arm's whole point. Negative instructions count too ("must NOT mention the
    constitution" still shows the model the word)."""
    for sc in cfg["stages"]:
        for role, text in (sc.get("prompts") or {}).items():
            low = str(text).lower()
            for word in _TRACES:
                assert word not in low, f"{sc['name']}.{role} carries {word!r}"
    low = cfg["guideline"]["text"].lower() + " " + cfg["guideline"]["name"].lower()
    assert not any(w in low for w in _TRACES)
    for key in ("chunking", "only_traits", "n_traits"):
        assert key not in cfg, f"{key} cuts or counts a document this arm does not have"
    # The style section is constitution text; its slot must not survive either.
    assert "{style_guidance}" not in _stage(cfg, "draft_responses")["prompts"]["system"]
    # No corpus check may ask for a unit to cover: there is none.
    fields = _stage(cfg, "corpus")["fields"]
    assert "unit" not in fields and "members" not in fields


def test_every_standard_stage_sees_the_guideline_and_nothing_else(cfg):
    """Each of the four stages that used to see a principle renders the guideline in its
    `<standard>` slot, and every template variable resolves (a leftover `{trait_text}`
    would be a KeyError at generation time, after money was spent)."""
    (u,), _ = units_from_config(cfg)
    record = {**u.as_dict(), "trait_name": u.name, "trait_text": u.text,
              "scenario_id": "guideline_b00_s000", "situation": "s", "shortcut": "x",
              "domain": "d", "system": "SYS", "user": "USR", "draft_system": "ds",
              "draft_user": "du", "draft_reasoning": "dr", "draft_response": "dp",
              "reasoning": "r", "response": "p", "changes": "c"}
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=Path("."), smoke=True,
              vars={"constitution": document_text(cfg), "style_guidance": ""})
    for sc in cfg["stages"]:
        for role, text in (sc.get("prompts") or {}).items():
            # Rendering IS the check: an unresolved placeholder raises KeyError here
            # rather than mid-run, after the earlier stages were paid for.
            out = _render(text, record, ctx, n=2, avoid="", overrepresented="")
            if sc["name"] in _STANDARD_STAGES:
                if "{trait_text}" in text:
                    assert cfg["guideline"]["text"].strip() in out
                    assert f'<standard name="{cfg["guideline"]["name"]}">' in out
    for name in _STANDARD_STAGES:
        joined = " ".join(_stage(cfg, name)["prompts"].values())
        assert "{trait_text}" in joined, f"{name} no longer sees the standard at all"


def test_everything_else_matches_the_baseline(cfg, baseline):
    """The two recipes must differ ONLY in what the generator is shown, or the ODCV
    contrast is confounded by whatever else moved."""
    assert [s["kind"] for s in cfg["stages"]] == [s["kind"] for s in baseline["stages"]]
    assert [s["name"] for s in cfg["stages"]][1:] == \
        [s["name"] for s in baseline["stages"]][1:]
    assert cfg["models"] == baseline["models"]
    assert cfg["defaults"] == baseline["defaults"]
    assert cfg["seed"] == baseline["seed"]
    assert cfg["total_scenarios"] == baseline["total_scenarios"]
    assert cfg["scenarios_per_call"] == baseline["scenarios_per_call"]
    for name in ("draft_responses", "revise_responses"):
        assert _stage(cfg, name)["lint"] == _stage(baseline, name)["lint"], (
            f"{name}: the recital ban must not move, or a change in how often it fires "
            f"cannot be read against the baseline")
    assert _stage(cfg, "write_scenarios")["diversity"] == \
        _stage(baseline, "write_scenarios")["diversity"]
    assert _stage(cfg, "export_sft")["metadata"] == _stage(baseline, "export_sft")["metadata"]
    # The corpus checks and their judge wording are the same instrument, so a number
    # from either corpus means the same thing.
    for key in ("properties", "rubrics", "on_fail"):
        assert _stage(cfg, "corpus")[key] == _stage(baseline, "corpus")[key], key
    assert _stage(cfg, "corpus_scenarios") == _stage(baseline, "corpus_scenarios")
    assert _stage(cfg, "draft_responses")["prompts"]["user"] == \
        _stage(baseline, "draft_responses")["prompts"]["user"]
    assert _stage(cfg, "revise_responses")["ablate_with"] == \
        _stage(baseline, "revise_responses")["ablate_with"]
    # The smoke block must shrink a `total_scenarios` recipe (the 2026-08-13 lesson).
    assert cfg["smoke"]["total_scenarios"] <= 8


def test_the_estimator_prices_a_single_unit(cfg):
    assert pipeline.n_units(cfg) == 1
    est = pipeline.estimate(cfg)
    assert est["planned_documents"] == cfg["total_scenarios"]
    calls = {r["stage"]: r["calls"] for r in est["per_stage"]}
    assert calls["scenarios"] == -(-cfg["total_scenarios"] // cfg["scenarios_per_call"])
    assert calls["refine"] == calls["rewrite"] == cfg["total_scenarios"]
    assert est["total_usd"] > 0


def test_a_none_run_records_no_document_sha_and_publishes_the_guideline(tmp_path, cfg):
    """The manifest must say there was no document (None, not the sha of ''), and stage
    1's snapshot is the guideline row every scenario will inherit its fields from."""
    mini = {
        "pipeline": "none_probe",
        "constitution": NO_CONSTITUTION,
        "guideline": cfg["guideline"],
        "output_dir": str(tmp_path),
        "hf_repo": None,
        "hf_private": False,
        "workers": 1,
        "models": {},
        "stages": [{"name": "declare_guideline", "kind": "segment"}],
    }
    m = pipeline.run(mini)
    assert m["constitution_sha256"] is None
    assert m["counts"]["declare_guideline"] == 1
    (run_dir,) = [d for d in tmp_path.iterdir() if d.is_dir()]
    rows = [json.loads(line) for line in
            (run_dir / "stage_1_declare_guideline.jsonl").open(encoding="utf-8")]
    assert rows == [units_from_config(cfg)[0][0].as_dict()]
    assert rows[0]["trait_id"] == GUIDELINE_UNIT_ID
    assert rows[0]["text"] == cfg["guideline"]["text"].strip()
    assert m["config"]["constitution"] == NO_CONSTITUTION
