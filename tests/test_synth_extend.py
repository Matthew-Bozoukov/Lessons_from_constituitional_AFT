# ABOUTME: Offline tests for `extend_from`: the prior corpus seeds the scenario gate, its
# ABOUTME: rows ride into dataset.jsonl, and an incompatible prior is refused before spend.
from __future__ import annotations

import dataclasses
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


def test_check_compatible_accepts_the_pinned_equivalent_constitution():
    from src.data.synth.ours.constitution import EQUIVALENT_SHA256, same_constitution
    old, new = sorted(EQUIVALENT_SHA256[0])  # 7726f860..., 8e273b47...
    ok = {"pipeline": "da", "id_prefix": "r2_"}
    check_compatible(_prior(pipeline="da", constitution_sha256=old), ok, new)
    check_compatible(_prior(pipeline="da", constitution_sha256=new), ok, old)
    with pytest.raises(ValueError, match="different constitution"):
        check_compatible(_prior(pipeline="da", constitution_sha256=old), ok, "f" * 64)
    assert same_constitution("a", "a") and not same_constitution("a", "b")


def test_merged_rows_puts_prior_first_and_refuses_an_id_in_both():
    new = [{"messages": [], "metadata": {"scenario_id": "r2_t1_b00_s000"}}]
    out = merged_rows(_prior(), new)
    assert [r["metadata"]["scenario_id"] for r in out] == \
        ["t1_b00_s000", "t2_b00_s000", "r2_t1_b00_s000"]
    with pytest.raises(ValueError, match="already exist"):
        merged_rows(_prior(), [{"messages": [], "metadata": {"scenario_id": "t1_b00_s000"}}])


def test_prior_scenarios_cover_rows_an_earlier_run_left_out_of_the_snapshot():
    """An extended prior: dataset.jsonl holds run 1 + run 2, stages/ holds run 2 only."""
    rows = [{"messages": [], "metadata": {"scenario_id": "t1_b00_s000", "domain": "nursing",
                                          "situation": DISTINCT[0], "shortcut": "x"}},
            {"messages": [], "metadata": {"scenario_id": "r2_t1_b00_s000", "situation": DISTINCT[1]}}]
    snapshot = [{"scenario_id": "r2_t1_b00_s000", "situation": DISTINCT[1]},
                {"scenario_id": "r2_t1_b00_s001", "situation": DISTINCT[2]}]  # dropped downstream
    got = extend.prior_scenarios(rows, snapshot)
    assert sorted(s["situation"] for s in got) == sorted(DISTINCT[:3]), \
        "run 1's row, run 2's row and run 2's dropped scenario, each once"
    assert got[-1] == {"scenario_id": "t1_b00_s000", "domain": "nursing",
                       "situation": DISTINCT[0], "shortcut": "x"}


def test_prior_scenarios_refuse_a_row_nothing_can_dedupe_against():
    rows = [{"messages": [], "metadata": {"scenario_id": "t1_b00_s000"}}]
    with pytest.raises(ValueError, match="dedupe cannot cover"):
        extend.prior_scenarios(rows, [{"scenario_id": "r2_t1_b00_s000", "situation": DISTINCT[1]}])
    # covered by the snapshot, so no metadata is needed
    assert len(extend.prior_scenarios(rows, [{"scenario_id": "t1_b00_s000",
                                              "situation": DISTINCT[0]}])) == 1


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


def test_a_load_source_run_arm_extends_without_a_prefix():
    from src.data.synth.ours.extend import loads_source_ids
    sha = hashlib.sha256(b"c").hexdigest()
    cfg = {"pipeline": "da-grok-resp", "stages": [{"name": "load_prompts", "kind": "load_source_run"}]}
    assert loads_source_ids(cfg) and not loads_source_ids({"stages": [{"kind": "write_scenarios"}]})
    # the ids are the source's, so no prefix is minted and none is demanded ...
    check_compatible(_prior(pipeline="da-grok-resp", constitution_sha256=sha), cfg, sha)
    # ... but a prior whose rows cannot be told apart is still refused
    with pytest.raises(ValueError, match="disjointness"):
        bad = _prior(pipeline="da-grok-resp", constitution_sha256=sha)
        check_compatible(dataclasses.replace(bad, ids=frozenset({"t1_b00_s000", ""})), cfg, sha)


# --- the name lock at launch ------------------------------------------------------------

class _FakeCache:
    """StageCache without the Hub: records which repo the run would publish to."""
    made: list[str] = []

    def __init__(self, run_dir, repo_id, **_kw):
        self.run_dir, self.repo_id = run_dir, repo_id
        _FakeCache.made.append(repo_id)
        run_dir.mkdir(parents=True, exist_ok=True)

    def has(self, *_a):
        return False

    def save(self, *_a):
        return None

    def publish_final(self, _rows):
        return None

    def save_json(self, *_a):
        return None


def test_a_fresh_launch_onto_a_taken_name_stops_before_any_stage_runs(tmp_path, monkeypatch):
    """2026-10-02: an aborted launch pushed 1,978 scenarios into the canonical repo."""
    import src.infra.huggingface as hf

    monkeypatch.setenv("HF_ORG", "org")
    monkeypatch.setattr(hf, "hf_api", lambda: type("A", (), {
        "repo_exists": lambda self, rid, repo_type=None: True})())
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    ran = []
    ops.OPERATORS["fake"] = lambda sc, cfg: Stage(sc["name"], lambda c, r, k: ran.append(1) or [])
    with pytest.raises(hf.NameTaken, match="name_suffix"):
        pipeline.run(_cfg(tmp_path, hf_push=True))
    assert not ran


def test_the_lock_claims_a_suffixed_name_and_a_resume_publishes_where_it_started(tmp_path, monkeypatch):
    import src.infra.huggingface as hf

    monkeypatch.setenv("HF_ORG", "org")
    monkeypatch.setattr(pipeline, "StageCache", _FakeCache)
    taken = {f"org/{pipeline.synth_name('fake-type')}"}
    monkeypatch.setattr(hf, "hf_api", lambda: type("A", (), {
        "repo_exists": lambda self, rid, repo_type=None: rid in taken})())
    m = pipeline.run(_cfg(tmp_path, hf_push=True, name_suffix="nika"))
    want = f"org/{pipeline.synth_name('fake-type', suffix='nika')}"
    assert m["hf_repo"] == want and _FakeCache.made[-1] == want
    run_dir = tmp_path / m["run_id"]
    assert json.loads((run_dir / "hf_repo.json").read_text()) == {"repo": want}
    # The resumed run publishes to the repo its launch claimed, even though it now exists.
    taken.add(want)
    m2 = pipeline.run(_cfg(tmp_path, hf_push=True), resume=str(run_dir))
    assert m2["hf_repo"] == want


# --- continuing a deal from the prior, and planning the extension --------------------------

SECTORS = {f"s{i}": 1 for i in range(6)}


def test_a_continued_axis_deals_each_unit_the_labels_it_has_seen_least():
    """Run 1 dealt u1 s0-s2 and u2 s3-s5; run 2 rotated its list by hand. Continuing from
    the prior, u1 meets s3-s5 and u2 s0-s2 with no hand edit."""
    axes = {"sector": {"weights": SECTORS}}
    fresh = ops.deal_unit_axes(axes, ["u1", "u2"], {0: 3, 1: 3})
    assert fresh[("sector", 0)] == ["s0", "s1", "s2"] and fresh[("sector", 1)] == ["s3", "s4", "s5"]
    prior = {"sector": {"u1": {"s0": 1, "s1": 1, "s2": 1}, "u2": {"s3": 1, "s4": 1, "s5": 1}}}
    cont = ops.deal_unit_axes(axes, ["u1", "u2"], {0: 3, 1: 3}, prior)
    assert sorted(cont[("sector", 0)]) == ["s3", "s4", "s5"]
    assert sorted(cont[("sector", 1)]) == ["s0", "s1", "s2"]


def test_an_axis_that_does_not_continue_is_dealt_exactly_as_a_fresh_run_deals_it():
    """da: sectors continue, AI labels do not. The AI deal must not move."""
    axes = {"ai": {"weights": {"self": 1, "other": 1, "none": 4, "open": 4},
                   "unit_weights": {"u2": {"self": 1, "other": 1}}},
            "sector": {"weights": SECTORS}}
    plan = {0: 5, 1: 5}
    fresh = ops.deal_unit_axes(axes, ["u1", "u2"], plan)
    prior = {"sector": {"u1": {"s0": 3}, "u2": {"s5": 2}}}
    cont = ops.deal_unit_axes(axes, ["u1", "u2"], plan, prior)
    assert cont[("ai", 0)] == fresh[("ai", 0)] and cont[("ai", 1)] == fresh[("ai", 1)]
    assert "s0" not in cont[("sector", 0)] and "s5" not in cont[("sector", 1)]
    assert "s0" in fresh[("sector", 0)]


def test_prior_deals_count_calls_once_from_scenarios_and_rows_alike():
    rows = [{"messages": [], "metadata": {"scenario_id": f"t1_b00_s00{j}", "trait_id": "t1",
                                          "sector": "law", "ai": "none", "situation": DISTINCT[j]}}
            for j in range(3)]
    rows.append({"messages": [], "metadata": {"scenario_id": "r2_t1_b03_s000", "trait_id": "t1",
                                              "sector": "farming", "situation": DISTINCT[5]}})
    snap = [{"scenario_id": "r2_t1_b03_s001", "trait_id": "t1", "sector": "farming",
             "situation": DISTINCT[6]},
            {"scenario_id": "r2_t2_b00_s000", "trait_id": "t2", "sector": "law",
             "situation": DISTINCT[7]}]
    got = extend.prior_deals(_prior(rows=rows, scenarios=snap), ["sector"])
    assert got == {"sector": {"t1": {"law": 1, "farming": 1}, "t2": {"law": 1}}}
    bad = snap + [{"scenario_id": "r2_t2_b00_s001", "trait_id": "t2", "sector": "farming"}]
    with pytest.raises(ValueError, match="two sector labels"):
        extend.prior_deals(_prior(rows=rows, scenarios=bad), ["sector"])


def test_ids_name_their_call_and_their_run():
    assert extend.call_id("r2_t6_b11_s003") == "r2_t6_b11"
    assert extend.run_prefix("r2_t6_b11_s003", "t6") == "r2_"
    assert extend.run_prefix("t6_b11_s003", "t6") == ""
    rows = [{"metadata": {"scenario_id": s, "trait_id": "t1"}}
            for s in ("t1_b00_s000", "t1_b00_s001", "r2_t1_b00_s000")]
    assert extend.kept_by_run(rows) == {"": {"t1": 2}, "r2_": {"t1": 1}}


def test_loss_is_measured_on_the_runs_that_extended_a_corpus():
    first = extend.PriorRun("", {"t1": 72, "t6": 72}, {"t1": 72, "t6": 72}, extended=False)
    second = extend.PriorRun("r2_", {"t1": 73, "t6": 72}, {"t1": 73, "t6": 63}, extended=True)
    assert extend.loss_rates([second, first]) == {"t1": 0.0, "t6": 0.125}
    assert extend.loss_rates([first]) == {"t1": 0.0, "t6": 0.0}


def test_requested_by_unit_reads_a_weighted_ask_back_exactly():
    cfg = {"total_scenarios": 736, "scenarios_per_call": 4,
           "trait_weights": {"t1": 79, "t2": 76, "t3": 79, "t4": 78, "t5": 78, "t6": 100,
                             "t7": 87, "t8": 81, "t9": 78}}
    units = [f"t{i}" for i in range(1, 10)]
    assert extend.requested_by_unit(cfg, units) == cfg["trait_weights"]
    assert extend.requested_by_unit({"total_scenarios": 650}, units) == \
        {"t1": 73, "t2": 73, **{f"t{i}": 72 for i in range(3, 10)}}


def test_plan_reproduces_the_2026_10_30_da_numbers():
    """633 unused rows of 2026-10-02-da-synth; two new pools of 75 a trait."""
    unused = {"t1": 71, "t2": 74, "t3": 71, "t4": 72, "t5": 72, "t6": 63, "t7": 67, "t8": 71, "t9": 72}
    loss = {**{t: 0.0 for t in unused}, "t6": 0.125, "t7": 0.0417, "t8": 0.0139}
    plan = extend.plan_extension(unused, 2, 75, loss, tokens_per_row={t: 1164.0 for t in unused},
                                 prior_tokens=1_493_867, budget_tokens=5_000_005, pool_pct=15,
                                 goal_share=0.45)
    assert plan["need"] == {"t1": 79, "t2": 76, "t3": 79, "t4": 78, "t5": 78, "t6": 87,
                            "t7": 83, "t8": 79, "t9": 78}
    assert plan["new_rows"] == 717
    assert plan["request"]["t6"] == 100 and plan["request"]["t1"] == 79
    assert plan["total_scenarios"] == sum(plan["request"].values())
    assert plan["overrides"].startswith(f"total_scenarios={plan['total_scenarios']},trait_weights.t1=79")
    assert 0.46 < plan["expected_share"] < 0.47
    with pytest.raises(ValueError, match="raise per_pool"):
        extend.plan_extension(unused, 2, 60, loss, tokens_per_row={t: 1164.0 for t in unused},
                              prior_tokens=1_493_867, budget_tokens=5_000_005, pool_pct=15,
                              goal_share=0.45)


def test_the_writer_continues_sectors_from_the_prior_and_records_it(monkeypatch, tmp_path):
    monkeypatch.setattr(ops, "call_json", lambda *a, **k: (
        [{"domain": "d", "situation": "s", "shortcut": "x"}], {}))
    rotate = {"ai": {"per_trait": True, "weights": {"none": 1, "open": 1}},
              "sector": {"per_trait": True, "continue_from_prior": True,
                         "weights": {"s0": 1, "s1": 1, "s2": 1, "s3": 1}}}
    stage = {"name": "write", "model": "scenarios", "rotate": rotate,
             "prompts": {"system": "sys", "user": "make {n}{sector_text}{ai_text}"}}
    cfg = {"seed": 0, "scenarios_per_trait": 2, "scenarios_per_call": 1, "id_prefix": "r3_",
           "models": {"scenarios": {"model": "m", "temperature": 1.0, "max_tokens": 100}}}
    snap = [{"scenario_id": f"t1_b0{i}_s000", "trait_id": "t1", "sector": f"s{i}",
             "situation": DISTINCT[i]} for i in range(2)]
    prior = _prior(scenarios=snap)

    def run(p):
        ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False,
                  prior=p, _client=object())
        out = ops.OPERATORS["scenarios"](stage, cfg).fn(ctx, TRAITS, None)
        return {r["trait_id"]: sorted(x["sector"] for x in out if x["trait_id"] == r["trait_id"])
                for r in out}, ctx

    fresh, _ = run(None)
    assert fresh["t1"] == ["s0", "s1"]
    cont, ctx = run(prior)
    assert cont["t1"] == ["s2", "s3"], "t1 already met s0 and s1 in the prior"
    assert ctx.manifest_extra["continued_axes"]["write"] == {
        "sector": {"prior_calls": 2, "prior_units": 1}}
    with pytest.raises(ValueError, match="continue_from_prior"):
        ops.OPERATORS["scenarios"]({**stage, "rotate": {"sector": {
            "continue_from_prior": True, "weights": {"s0": 1}}}}, cfg)
