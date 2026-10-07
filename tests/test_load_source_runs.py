# ABOUTME: Offline tests for load_source_run's multi-run form: several pinned sources merged, duplicate ids
# ABOUTME: refused, and `sample: {total, by, seed}` drawing an even per-group sample recorded in the manifest.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.synth.ours.stage_operators import op_load_source_run


def _run_dir(tmp_path: Path, name: str, ids, trait_cycle=("t1", "t2", "t3")) -> Path:
    d = tmp_path / name
    d.mkdir()
    rows = [{"scenario_id": i, "trait_id": trait_cycle[k % len(trait_cycle)], "situation": f"s{i}"}
            for k, i in enumerate(ids)]
    (d / "stage_2_write_scenarios.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"git_sha": name, "constitution_sha256": None}))
    return d


def _ctx(tmp_path: Path, source: dict):
    return SimpleNamespace(cfg={"source": {"snapshot": "stage_2_write_scenarios.jsonl", **source}},
                           constitution="the constitution", run_dir=tmp_path, manifest_extra={})


def test_several_runs_merge_and_sample_evenly_by_group(tmp_path):
    a = _run_dir(tmp_path, "run_a", [f"a{i}" for i in range(9)])
    b = _run_dir(tmp_path, "run_b", [f"b{i}" for i in range(9)])
    ctx = _ctx(tmp_path, {"runs": [{"local_dir": str(a)}, {"local_dir": str(b)}],
                          "sample": {"total": 7, "by": "trait_id", "seed": 0}})
    out = op_load_source_run({"name": "load"}, ctx.cfg).fn(ctx, [], None)
    assert len(out) == 7
    per = {t: sum(r["trait_id"] == t for r in out) for t in ("t1", "t2", "t3")}
    assert sorted(per.values()) == [2, 2, 3]            # 7 over 3 groups: the remainder goes to the first groups
    assert len({r["scenario_id"] for r in out}) == 7
    meta = ctx.manifest_extra["source"]
    assert [m["n_records"] for m in meta["runs"]] == [9, 9]
    assert meta["sample"]["n_drawn"] == 7 and sorted(meta["sample"]["drawn_ids"]) == sorted(r["scenario_id"] for r in out)
    # the same seed draws the same ids
    again = op_load_source_run({"name": "load"}, ctx.cfg).fn(_ctx(tmp_path, ctx.cfg["source"]), [], None)
    assert [r["scenario_id"] for r in again] == [r["scenario_id"] for r in out]


def test_duplicate_ids_across_runs_are_refused(tmp_path):
    a = _run_dir(tmp_path, "run_a", ["x1", "x2", "x3"])
    b = _run_dir(tmp_path, "run_b", ["x3", "x4", "x5"])
    ctx = _ctx(tmp_path, {"runs": [{"local_dir": str(a)}, {"local_dir": str(b)}]})
    with pytest.raises(AssertionError, match="share scenario ids"):
        op_load_source_run({"name": "load"}, ctx.cfg).fn(ctx, [], None)


def test_a_group_too_small_for_its_share_fails_loudly(tmp_path):
    a = _run_dir(tmp_path, "run_a", ["a1", "a2", "a3", "a4"], trait_cycle=("t1", "t1", "t1", "t2"))
    ctx = _ctx(tmp_path, {"local_dir": str(a), "sample": {"total": 4, "by": "trait_id"}})
    with pytest.raises(ValueError, match="fewer than"):
        op_load_source_run({"name": "load"}, ctx.cfg).fn(ctx, [], None)


def test_single_source_without_sample_is_unchanged(tmp_path):
    a = _run_dir(tmp_path, "run_a", ["a1", "a2"])
    ctx = _ctx(tmp_path, {"local_dir": str(a)})
    out = op_load_source_run({"name": "load"}, ctx.cfg).fn(ctx, [], None)
    assert [r["scenario_id"] for r in out] == ["a1", "a2"]
    assert ctx.manifest_extra["source"]["source_run"] == str(a) and ctx.manifest_extra["source"]["sample"] is None


def test_extend_from_sets_aside_the_prompts_the_prior_already_answered(tmp_path):
    from src.data.synth.ours.extend import Prior
    a = _run_dir(tmp_path, "run_a", [f"a{i}" for i in range(9)])
    answered = {"a0", "a1", "a2", "a3", "a4", "a5"}
    prior = Prior(repo="org/prior-synth", revision="abc123def456",
                  rows=[{"messages": [], "metadata": {"scenario_id": i}} for i in answered],
                  scenarios=[], scenario_file="stages/stage_4_revise_prompts.jsonl", ids=frozenset(answered))
    ctx = _ctx(tmp_path, {"local_dir": str(a), "sample": {"total": 3, "by": "trait_id", "seed": 1}})
    ctx.prior = prior
    out = op_load_source_run({"name": "load"}, ctx.cfg).fn(ctx, [], None)
    assert sorted(r["scenario_id"] for r in out) == ["a6", "a7", "a8"]   # only what the prior has not answered
    meta = ctx.manifest_extra["source"]
    assert meta["set_aside_prior_ids"] == 6 and meta["sample"]["n_drawn"] == 3
