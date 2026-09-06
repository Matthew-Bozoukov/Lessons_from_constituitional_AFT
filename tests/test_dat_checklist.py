# ABOUTME: dat's per-scenario checklist: `rotate_per: scenario` deals every axis per scenario,
# ABOUTME: independently and with exact counts, and the prompt/rows carry the dealt lines.
"""The 2026-09-06 change: scenarios are conditioned on a dealt checklist, not steered by waves.

Same scripted-generator harness as tests/test_good_ai_fiction.py -- the operator is real,
only the model call is faked -- so these pin BEHAVIOUR: what the generator is shown, how
its reply is matched back to the slots it was dealt, and that the deal is what the papers
this copies rely on (independent axes with the declared marginals).
"""

from __future__ import annotations

import collections
import itertools

import pytest
import yaml

from src.data.synth import stage_operators as ops
from src.data.synth.stage_runtime import Ctx, Usage

CFG_PATH = "configs/data/synth/dat.yaml"
CFG = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
AXES = ("sector", "task_shape", "environment", "pressure", "framing", "visibility")
MIN_OPTIONS = {"sector": 10, "task_shape": 10, "environment": 10, "pressure": 20,
               "framing": 2, "visibility": 3}


def _stage(name: str) -> dict:
    return next(s for s in CFG["stages"] if s["name"] == name)


# --- the config -------------------------------------------------------------------------

def test_the_checklist_axes_are_declared_with_text_for_every_option() -> None:
    st = _stage("write_scenarios")
    assert st["rotate_per"] == "scenario"
    assert "diversity" not in st, "the checklist replaces the waves; both is a contradiction"
    assert tuple(st["rotate"]) == AXES
    for axis, spec in st["rotate"].items():
        assert len(spec["weights"]) >= MIN_OPTIONS[axis], f"{axis}: too few options"
        assert set(spec["text"]) == set(spec["weights"]), f"{axis}: text/weights disagree"
        assert len(set(spec["weights"].values())) == 1, f"{axis}: the deal is uniform"
    # `domain` is what the generator writes (a specific one- or two-word setting); the dealt
    # sector must not overwrite it, so the axis carries a different name.
    assert "domain" not in st["rotate"]


def test_the_prompts_read_the_checklist_and_nothing_hashes_framing_any_more() -> None:
    assert "{checklist}" in _stage("write_scenarios")["prompts"]["user"]
    assert '"slot"' in _stage("write_scenarios")["prompts"]["user"]
    for name in ("draft_environment", "revise_environment"):
        assert "{checklist}" in _stage(name)["prompts"]["user"], f"{name} never sees the checklist"
    assert not any(s.get("kind") == "assign" and "framing" in (s.get("fields") or {})
                   for s in CFG["stages"])
    export = next(s for s in CFG["stages"] if s["kind"] == "chat_export")
    assert set(AXES) <= set(export["metadata"])


# --- the deal ---------------------------------------------------------------------------

def test_sample_labels_keeps_exact_counts_and_decorrelates_axes() -> None:
    st = _stage("write_scenarios")
    n = 2000
    seqs = {axis: ops.sample_labels(st["rotate"][axis]["weights"], n, 0, axis) for axis in AXES}
    for axis, seq in seqs.items():
        counts = collections.Counter(seq)
        assert len(seq) == n and max(counts.values()) - min(counts.values()) <= 1, axis
    # The batch walk locked axes whose label counts share a factor (20 sectors fixed every
    # slot's parity). Dealt per scenario, every sector meets both framings, every
    # visibility and most pressures.
    for sector in st["rotate"]["sector"]["weights"]:
        idx = [i for i, v in enumerate(seqs["sector"]) if v == sector]
        assert {seqs["framing"][i] for i in idx} == {"mandated", "incentivized"}
        assert len({seqs["visibility"][i] for i in idx}) == 5
        assert len({seqs["pressure"][i] for i in idx}) >= 18
    assert ops.sample_labels({"a": 1, "b": 1}, 10, 0, "x") != ops.sample_labels({"a": 1, "b": 1}, 10, 1, "x")
    # A tiny deal (the three-row smoke) does not always draw the first labels of the list.
    smokes = {tuple(sorted(ops.sample_labels(st["rotate"]["sector"]["weights"], 3, seed, "sector")))
              for seed in range(6)}
    assert len(smokes) > 1


# --- the operator, driven offline ------------------------------------------------------

_UNITS = [{"trait_id": "t1", "index": 0, "name": "Honesty", "text": "..."},
          {"trait_id": "t2", "index": 1, "name": "Care", "text": "..."}]


def _mini_stage(**over):
    real = _stage("write_scenarios")
    stage = {"name": "write_scenarios", "model": "scenarios", "rotate_per": "scenario",
             "rotate": real["rotate"],
             "prompts": {"system": "sys", "user": "{trait_name} {n}\n{checklist}"}}
    return {**stage, **over}


def _cfg(**over):
    return {"seed": 0, "total_scenarios": 6, "scenarios_per_call": 3,
            "models": {"scenarios": {"model": "m", "temperature": 1.0, "max_tokens": 100}},
            **over}


def _drive(monkeypatch, tmp_path, stage, cfg, reply):
    seen: list[dict] = []

    def fake_call_json(client, usage, model, system, user, temp, max_tokens, stage=None,
                       extra=None):
        seen.append({"system": system, "user": user})
        return reply(user), {}

    monkeypatch.setattr(ops, "call_json", fake_call_json)
    st = ops.OPERATORS["scenarios"](stage, cfg)
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False)
    return st.fn(ctx, _UNITS, None), seen


def test_each_scenario_in_a_call_is_dealt_its_own_checklist_and_matched_by_slot(monkeypatch, tmp_path):
    # The reply comes back out of order, with one object naming a slot that does not exist
    # and one repeating a slot: those two are dropped, the three real ones land on theirs.
    def reply(_user):
        return [{"slot": 2, "domain": "d2", "situation": "two", "shortcut": "x"},
                {"slot": 9, "domain": "d9", "situation": "nine", "shortcut": "x"},
                {"slot": 1, "domain": "d1", "situation": "one", "shortcut": "x"},
                {"slot": 3, "domain": "d3", "situation": "three", "shortcut": "x"},
                {"slot": 3, "domain": "dup", "situation": "again", "shortcut": "x"}]

    rows, seen = _drive(monkeypatch, tmp_path, _mini_stage(), _cfg(), reply)
    assert len(rows) == 6 and len(seen) == 2
    st = _stage("write_scenarios")
    for r in rows:
        for axis in AXES:
            assert r[axis] in st["rotate"][axis]["weights"], (axis, r[axis])
            assert f"{axis}: {st['rotate'][axis]['text'][r[axis]]}" in r["checklist"]
        assert "slot" not in r
    # Slot, not position: "two" was first in the reply and sits in slot 2.
    by_sit = {r["situation"]: r for r in rows if r["trait_id"] == "t1"}
    assert by_sit["one"]["scenario_id"].endswith("_s000")
    assert by_sit["two"]["scenario_id"].endswith("_s001")
    assert by_sit["three"]["scenario_id"].endswith("_s002")
    # Per SCENARIO: the three in one call do not share a checklist.
    assert len({r["checklist"] for r in rows if r["trait_id"] == "t1"}) == 3
    # And the checklist the row carries is the one its slot was shown in the prompt.
    prompt = seen[0]["user"]
    assert prompt.startswith("Honesty 3\nSituation 1:\n")
    for k, sit in ((1, "one"), (2, "two"), (3, "three")):
        block = prompt.split(f"Situation {k}:\n")[1].split("\n\nSituation")[0]
        assert block == "\n".join(f"  {line}" for line in by_sit[sit]["checklist"].splitlines())


def test_a_one_slot_call_implies_its_slot(monkeypatch, tmp_path):
    def reply(_user):
        return [{"domain": "d", "situation": "only", "shortcut": "x"}]

    rows, seen = _drive(monkeypatch, tmp_path, _mini_stage(),
                        _cfg(total_scenarios=2, scenarios_per_call=1), reply)
    assert [r["situation"] for r in rows] == ["only", "only"] and len(seen) == 2
    assert all(r["scenario_id"].endswith("_b00_s000") for r in rows)


def test_the_deal_is_reproducible_and_seed_dependent(monkeypatch, tmp_path):
    def reply(user):
        return [{"slot": k, "domain": "d", "situation": f"s{k}", "shortcut": "x"}
                for k in (1, 2, 3)]

    a, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _cfg(), reply)
    b, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _cfg(), reply)
    c, _ = _drive(monkeypatch, tmp_path, _mini_stage(), _cfg(seed=7), reply)
    key = lambda rows: [tuple(r[a_] for a_ in AXES) for r in rows]  # noqa: E731
    assert key(a) == key(b) and key(a) != key(c)


def test_per_scenario_dealing_refuses_what_only_makes_sense_per_batch() -> None:
    with pytest.raises(AssertionError, match="rotate_per"):
        ops.OPERATORS["scenarios"](_mini_stage(rotate_per="row"), _cfg())
    with pytest.raises(AssertionError, match="library"):
        ops.OPERATORS["scenarios"](_mini_stage(library={"file": "x", "var": "v", "item": "i"}), _cfg())
    with pytest.raises(AssertionError, match="needs a `rotate:` block"):
        ops.OPERATORS["scenarios"](_mini_stage(rotate={}), _cfg())


def test_batch_dealing_is_untouched(monkeypatch, tmp_path):
    """good-ai-fiction's per-batch contract: every scenario from one call shares the labels,
    the prompt gets `{<axis>}` / `{<axis>_text}`, and no `checklist` field appears."""
    stage = _mini_stage(rotate_per="batch",
                        prompts={"system": "sys", "user": "{n} {sector} {sector_text}"})

    def reply(_user):
        return [{"domain": "d", "situation": f"s{k}", "shortcut": "x"} for k in range(3)]

    rows, seen = _drive(monkeypatch, tmp_path, stage, _cfg(), reply)
    assert len(rows) == 6 and "checklist" not in rows[0]
    assert len({r["sector"] for r in rows if r["trait_id"] == "t1"}) == 1
    assert any(_stage("write_scenarios")["rotate"]["sector"]["text"][rows[0]["sector"]] in p["user"]
               for p in seen)


def test_every_axis_pair_is_covered_at_corpus_scale() -> None:
    """2,000 slots reach (nearly) every pairwise cell: the product space is what the
    checklist buys over a per-batch deal. Independent draws, so the sparsest pair
    (20 x 13 = 260 cells, ~8 rows each) is allowed the odd empty cell."""
    st = _stage("write_scenarios")
    seqs = {axis: ops.sample_labels(st["rotate"][axis]["weights"], 2000, 0, axis) for axis in AXES}
    for a, b, floor in (("sector", "task_shape", 0.98), ("pressure", "framing", 1.0),
                        ("environment", "visibility", 1.0)):
        seen = set(zip(seqs[a], seqs[b]))
        want = set(itertools.product(st["rotate"][a]["weights"], st["rotate"][b]["weights"]))
        assert len(seen & want) >= floor * len(want), f"{a} x {b}: {len(want - seen)} cells never dealt"
