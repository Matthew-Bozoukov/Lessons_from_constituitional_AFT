# ABOUTME: Offline tests for `extend_from`: the prior corpus seeds the scenario gate, its
# ABOUTME: rows ride into dataset.jsonl, and an incompatible prior is refused before spend.
from __future__ import annotations

import hashlib
import json

import pytest

from src.data.synth.ours import extend, pipeline, stage_operators as ops
from src.data.synth.ours.extend import Prior, check_compatible, merged_rows, parse_spec
from src.data.synth.ours.stage_runtime import Ctx, Stage, Usage

CONSTITUTION = "constitutions/archive/claude_distilled_8_principles_v1/constitution.md"
TRAITS = [{"trait_id": "t1", "index": 0, "name": "Oversight", "text": "..."},
          {"trait_id": "t2", "index": 1, "name": "Honesty", "text": "..."}]
DISTINCT = ["A nurse must decide whether to log an overtime shift she did not work",
            "A cyclist is offered a courier contract that requires ignoring red lights",
            "A baker is asked to relabel yesterday's bread as fresh for a food festival",
            "A violinist is pressed to claim a prize entry was composed unaided",
            "A surveyor is told to sign off on a boundary he never physically walked",
            "A translator is asked to soften a witness statement for a sympathetic client",
            "A brewer is offered cheaper hops if he keeps the origin off the label",
            "A librarian is asked to quietly discard donations from a disliked patron"]


def _prior(rows=None, scenarios=None, **kw) -> Prior:
    rows = rows if rows is not None else [
        {"messages": [], "metadata": {"scenario_id": "t1_b00_s000", "trait_id": "t1"}},
        {"messages": [], "metadata": {"scenario_id": "t2_b00_s000", "trait_id": "t2"}}]
    scenarios = scenarios if scenarios is not None else [
        {"scenario_id": "t1_b00_s000", "domain": "nursing", "situation": DISTINCT[0]}]
    ids = frozenset(r["metadata"]["scenario_id"] for r in rows)
    return Prior(repo="org/prior-synth", revision="abc123def456", rows=rows,
                 scenarios=scenarios, scenario_file="stages/stage_3_dedupe_scenarios.jsonl",
                 ids=ids, **kw)


# --- the spec and the compatibility gate -------------------------------------------

def test_parse_spec_splits_repo_and_revision():
    assert parse_spec("org/repo@deadbeef") == ("org/repo", "deadbeef")
    assert parse_spec("org/repo") == ("org/repo", None)
    with pytest.raises(AssertionError, match="not an HF dataset repo id"):
        parse_spec("output/synthdoc_v3/dataset.jsonl")


def test_compatibility_refuses_style_constitution_and_prefix_faults():
    sha = hashlib.sha256(b"c").hexdigest()
    ok = {"pipeline": "da", "id_prefix": "r2_"}
    check_compatible(_prior(pipeline="da", constitution_sha256=sha), ok, sha)
    with pytest.raises(ValueError, match="style-types"):
        check_compatible(_prior(pipeline="dat"), ok, sha)
    with pytest.raises(ValueError, match="different constitution"):
        check_compatible(_prior(constitution_sha256="0" * 64), ok, sha)
    with pytest.raises(ValueError, match="id_prefix"):
        check_compatible(_prior(), {"pipeline": "da"}, sha)
    with pytest.raises(ValueError, match="already in use"):
        check_compatible(_prior(), {"pipeline": "da", "id_prefix": "t1_"}, sha)


def test_merged_rows_puts_prior_first_and_refuses_an_id_in_both():
    new = [{"messages": [], "metadata": {"scenario_id": "r2_t1_b00_s000"}}]
    out = merged_rows(_prior(), new)
    assert [r["metadata"]["scenario_id"] for r in out] == \
        ["t1_b00_s000", "t2_b00_s000", "r2_t1_b00_s000"]
    with pytest.raises(ValueError, match="already exist"):
        merged_rows(_prior(), [{"messages": [], "metadata": {"scenario_id": "t1_b00_s000"}}])


# --- the scenario stage dedupes against the prior --------------------------------

def _run_scenarios(monkeypatch, tmp_path, replies, prior, capture):
    box = list(replies)

    def fake_call_json(client, usage, model, system, user, temp, max_tokens, stage=None,
                       extra=None):
        capture.append({"system": system, "user": user})
        batch = box.pop(0) if box else []
        return [{"domain": f"d{i}", "situation": s, "shortcut": "x"}
                for i, s in enumerate(batch)], {}

    monkeypatch.setattr(ops, "call_json", fake_call_json)
    stage = {"name": "scenarios", "model": "scenarios",
             "prompts": {"system": "sys {avoid}", "user": "make {n} {avoid} {overrepresented}"},
             "diversity": {"wave_size": 1, "reject_cosine": 0.9, "max_regen_rounds": 2}}
    cfg = {"seed": 0, "scenarios_per_trait": 2, "scenarios_per_call": 2, "id_prefix": "r2_",
           "models": {"scenarios": {"model": "m", "temperature": 1.0, "max_tokens": 100}}}
    st = ops.OPERATORS["scenarios"](stage, cfg)
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False,
              prior=prior, _client=object())
    return st.fn(ctx, TRAITS, None), ctx


def test_prior_scenarios_are_banned_in_the_prompt_and_refused_by_the_gate(monkeypatch, tmp_path):
    """A generator that re-writes a PRIOR run's scenario verbatim: the ban list named it
    from the first call, and the gate drops it even so, then refills."""
    capture: list[dict] = []
    replies = [[DISTINCT[0], DISTINCT[1]], DISTINCT[2:4], DISTINCT[4:6], DISTINCT[6:8]]
    out, ctx = _run_scenarios(monkeypatch, tmp_path, replies, _prior(), capture)

    assert "nursing: A nurse must decide" in capture[0]["user"], \
        "the prior's gist must reach the very first call"
    entry = ctx.manifest_extra["scenario_diversity"]["scenarios"]
    assert entry["prior_seeded"] == 1 and entry["rejected"] >= 1, entry
    assert DISTINCT[0] not in [r["situation"] for r in out], "the prior's scenario got in"
    assert all(r["scenario_id"].startswith("r2_") for r in out)


def test_without_a_prior_nothing_changes(monkeypatch, tmp_path):
    capture: list[dict] = []
    out, ctx = _run_scenarios(monkeypatch, tmp_path,
                              [DISTINCT[0:2], DISTINCT[2:4]], None, capture)
    assert ctx.manifest_extra["scenario_diversity"]["scenarios"]["prior_seeded"] == 0
    assert len(out) == 4 and "nursing" not in capture[0]["user"]


# --- the pipeline carries the prior's rows and records it ---------------------------

def _fake_operator(sc, cfg):
    def fn(ctx, records, ckpt):
        return [{"messages": [], "metadata": {"scenario_id": f"{cfg.get('id_prefix', '')}t1_b00_s{i:03d}"}}
                for i in range(2)]
    return Stage(sc["name"], fn)


@pytest.fixture(autouse=True)
def _register_fake():
    ops.OPERATORS["fake"] = _fake_operator
    yield
    ops.OPERATORS.pop("fake", None)


def _cfg(tmp_path, **extra):
    return {"pipeline": "fake-type", "hf_push": False, "constitution": CONSTITUTION,
            "output_dir": str(tmp_path), "workers": 1,
            "stages": [{"name": "make", "kind": "fake"}], **extra}


def test_pipeline_merges_the_prior_into_dataset_jsonl_and_the_manifest(tmp_path, monkeypatch):
    seen = {}

    def fake_load(spec):
        seen["spec"] = spec
        return _prior(pipeline="fake-type", run_id="20260923", git_sha="1111111")

    monkeypatch.setattr(pipeline, "load_prior", fake_load)
    m = pipeline.run(_cfg(tmp_path, extend_from="org/prior-synth@abc123def456", id_prefix="r2_"))

    assert seen["spec"] == "org/prior-synth@abc123def456"
    rows = [json.loads(l) for l in (tmp_path / m["run_id"] / "dataset.jsonl").open()]
    assert [r["metadata"]["scenario_id"] for r in rows] == \
        ["t1_b00_s000", "t2_b00_s000", "r2_t1_b00_s000", "r2_t1_b00_s001"]
    assert m["counts"] == {"make": 2, "dataset": 4}, "stage counts stay this run's own"
    assert m["extend_from"] == {"repo": "org/prior-synth", "revision": "abc123def456",
                                "run_id": "20260923", "git_sha": "1111111", "rows": 2,
                                "scenario_file": "stages/stage_3_dedupe_scenarios.jsonl"}
    # The stage snapshot is this run's alone; only the published dataset is the union.
    snap = [json.loads(l) for l in (tmp_path / m["run_id"] / "stage_1_make.jsonl").open()]
    assert len(snap) == 2


def test_pipeline_refuses_a_missing_prefix_before_any_stage_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_prior", lambda spec: _prior(pipeline="fake-type"))
    ran = []
    ops.OPERATORS["fake"] = lambda sc, cfg: Stage(sc["name"], lambda c, r, k: ran.append(1) or [])
    with pytest.raises(ValueError, match="id_prefix"):
        pipeline.run(_cfg(tmp_path, extend_from="org/prior-synth"))
    assert not ran and not list(tmp_path.iterdir()), "nothing ran, nothing was written"


def test_pipeline_without_extend_from_records_none(tmp_path):
    m = pipeline.run(_cfg(tmp_path))
    assert m["extend_from"] is None and "dataset" not in m["counts"]
