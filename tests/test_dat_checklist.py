# ABOUTME: dat's scenario checklist: a free `checklist` stage deals every scenario one value per
# ABOUTME: axis (uniform, exact, independent) and a per-record llm_json writes one situation from it.
"""The 2026-09-06 overhaul: scenarios are conditioned on a dealt checklist, not batched per call
and steered by waves. The deal operator is real and driven offline; the config is read as it is.
"""

from __future__ import annotations

import collections
import itertools

import pytest
import yaml

from src.data.synth import pipeline
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

def test_the_axes_are_uniform_label_to_text_maps_with_enough_options() -> None:
    st = _stage("deal_checklists")
    assert st["kind"] == "checklist" and tuple(st["axes"]) == AXES
    for axis, options in st["axes"].items():
        assert len(options) >= MIN_OPTIONS[axis], f"{axis}: too few options"
        assert all(isinstance(t, str) and t for t in options.values()), f"{axis}: every value has text"
    # `domain` is what the writer returns (a specific one- or two-word setting); no axis may
    # shadow it or any other scenario field.
    assert not set(st["axes"]) & ops._CHECKLIST_RESERVED


def test_none_of_the_batch_era_machinery_survives_in_dat() -> None:
    blob = yaml.safe_dump(CFG)
    for key in ("scenarios_per_call", "scenarios_per_trait", "rotate", "diversity:", "library:",
                "honest_cost", "{avoid}", "{overrepresented}"):
        assert key not in blob, f"{key} is still in dat.yaml"
    assert all(s["kind"] != "scenarios" for s in CFG["stages"])
    assert "total_scenarios" in CFG and CFG["smoke"]["total_scenarios"] == 3


def test_the_writer_is_an_ordinary_per_record_call_that_reads_the_checklist() -> None:
    st = _stage("write_scenarios")
    assert st["kind"] == "llm_json" and st["checkpoint"] == "scenario_id"
    assert set(st["save"]) == {"domain", "situation", "shortcut"}
    user = st["prompts"]["user"]
    for var in ("{checklist}", "{trait_name}", "{trait_text}"):
        assert var in user
    assert "one situation" in user and "{n}" not in user
    for name in ("draft_environment", "revise_environment"):
        assert "{checklist}" in _stage(name)["prompts"]["user"], f"{name} never sees the checklist"
    assert not any(s.get("kind") == "assign" and "framing" in (s.get("fields") or {})
                   for s in CFG["stages"])
    export = next(s for s in CFG["stages"] if s["kind"] == "chat_export")
    assert set(AXES) <= set(export["metadata"])


def test_the_estimator_prices_one_call_per_scenario() -> None:
    cfg = {**CFG, **CFG["smoke"]}
    assert pipeline._calls(cfg)["scenarios"] == 3
    assert pipeline._calls(dict(CFG))["scenarios"] == CFG["total_scenarios"]


# --- the deal ---------------------------------------------------------------------------

def test_sample_labels_keeps_exact_counts_and_decorrelates_axes() -> None:
    axes = _stage("deal_checklists")["axes"]
    n = 2000
    seqs = {axis: ops.sample_labels(list(axes[axis]), n, 0, axis) for axis in AXES}
    for axis, seq in seqs.items():
        counts = collections.Counter(seq)
        assert len(seq) == n and set(counts) == set(axes[axis])
        assert max(counts.values()) - min(counts.values()) <= 1, axis
    # A round-robin deal locked axes whose label counts share a factor (20 sectors fixed
    # every slot's parity). Shuffled, every sector meets both framings, every visibility
    # and most pressures.
    for sector in axes["sector"]:
        idx = [i for i, v in enumerate(seqs["sector"]) if v == sector]
        assert {seqs["framing"][i] for i in idx} == {"mandated", "incentivized"}
        assert len({seqs["visibility"][i] for i in idx}) == 5
        assert len({seqs["pressure"][i] for i in idx}) >= 18
    assert ops.sample_labels(["a", "b"], 10, 0, "x") != ops.sample_labels(["a", "b"], 10, 1, "x")
    # A tiny deal (the three-row smoke) does not always draw the first labels of the list.
    smokes = {tuple(sorted(ops.sample_labels(list(axes["sector"]), 3, seed, "sector")))
              for seed in range(6)}
    assert len(smokes) > 1


def test_every_axis_pair_is_covered_at_corpus_scale() -> None:
    """2,000 rows reach (nearly) every pairwise cell: the product space is what the checklist
    buys. Independent draws, so the sparsest pair (20 x 13 = 260 cells, ~8 rows each) is
    allowed the odd empty cell."""
    axes = _stage("deal_checklists")["axes"]
    seqs = {axis: ops.sample_labels(list(axes[axis]), 2000, 0, axis) for axis in AXES}
    for a, b, floor in (("sector", "task_shape", 0.98), ("pressure", "framing", 1.0),
                        ("environment", "visibility", 1.0)):
        seen = set(zip(seqs[a], seqs[b]))
        want = set(itertools.product(axes[a], axes[b]))
        assert len(seen & want) >= floor * len(want), f"{a} x {b}: {len(want - seen)} cells never dealt"


# --- the operator, driven offline ------------------------------------------------------

_UNITS = [{"trait_id": "t1", "index": 0, "name": "Honesty", "text": "Be honest.",
           "chunk_ids": ["c1"], "granularity": "principle", "grouping_strategy": "single",
           "n_chunks": 1},
          {"trait_id": "t2", "index": 1, "name": "Care", "text": "Take care.",
           "chunk_ids": ["c2"], "granularity": "principle", "grouping_strategy": "single",
           "n_chunks": 1}]


def _deal(tmp_path, total=6, seed=0, stage=None):
    st = ops.OPERATORS["checklist"](stage or _stage("deal_checklists"), {})
    assert not st.paid
    ctx = Ctx(cfg={"seed": seed, "total_scenarios": total}, usage=Usage(), workers=1,
              run_dir=tmp_path, smoke=False)
    return st.fn(ctx, _UNITS, None), ctx


def test_each_unit_gets_its_share_and_each_record_its_own_checklist(tmp_path):
    rows, ctx = _deal(tmp_path, total=7)
    assert [r["scenario_id"] for r in rows] == \
        ["t1_s0000", "t1_s0001", "t1_s0002", "t1_s0003", "t2_s0000", "t2_s0001", "t2_s0002"]
    axes = _stage("deal_checklists")["axes"]
    for r in rows:
        assert (r["trait_name"], r["chunk_ids"], r["n_chunks"]) in (("Honesty", ["c1"], 1), ("Care", ["c2"], 1))
        for axis in AXES:
            assert r[axis] in axes[axis], (axis, r[axis])
        assert r["checklist"] == "\n".join(f"{axis}: {axes[axis][r[axis]]}" for axis in AXES)
        assert "domain" not in r and "situation" not in r
    assert len({r["checklist"] for r in rows}) == 7
    assert set(ctx.manifest_extra["dealt_axes"]["deal_checklists"]) == set(AXES)


def test_the_deal_is_reproducible_and_seed_dependent(tmp_path):
    key = lambda rows: [tuple(r[a] for a in AXES) for r in rows]  # noqa: E731
    a, _ = _deal(tmp_path)
    b, _ = _deal(tmp_path)
    c, _ = _deal(tmp_path, seed=7)
    assert key(a) == key(b) and key(a) != key(c)


def test_the_writer_prompt_renders_from_the_dealt_record(tmp_path):
    rows, ctx = _deal(tmp_path, total=2)
    user = ops._render(_stage("write_scenarios")["prompts"]["user"], rows[0], ctx)
    assert rows[0]["checklist"] in user and 'name="Honesty"' in user and "Be honest." in user
    assert '{"domain":' in user  # the JSON contract survived the brace escaping


def test_the_operator_refuses_a_bad_axis_or_an_unsized_run(tmp_path):
    with pytest.raises(AssertionError, match="shadow"):
        ops.OPERATORS["checklist"]({"name": "d", "axes": {"domain": {"a": "x", "b": "y"}}}, {})
    with pytest.raises(AssertionError, match="at least two"):
        ops.OPERATORS["checklist"]({"name": "d", "axes": {"sector": {"a": "x"}}}, {})
    with pytest.raises(AssertionError, match="axes"):
        ops.OPERATORS["checklist"]({"name": "d"}, {})
    st = ops.OPERATORS["checklist"](_stage("deal_checklists"), {})
    with pytest.raises(AssertionError, match="total_scenarios"):
        st.fn(Ctx(cfg={}, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False), _UNITS, None)
