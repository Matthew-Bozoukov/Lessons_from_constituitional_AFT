# ABOUTME: The corpus_filter stage: it applies a corpus check's removal set, and every
# ABOUTME: guard that stops a failed check from turning into a deleted corpus.

"""What these pin down.

`corpus_filter` is the only stage allowed to delete records, so the tests that matter are
the ones asserting it REFUSES. The design invariant is that every failure mode of the
check it reads must degrade toward keeping records:

- a check that died writes no labels          -> nothing is dropped
- a check that sampled part of the corpus     -> error, not a partial sweep sold as full
- a removal set bigger than declared          -> error, not a mass delete
- `keep_when` (the phrasing that deletes on failure) -> rejected at build time
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.synth.stage_operators import OPERATORS
from src.data.synth.stage_runtime import Ctx, Usage


def _ctx(tmp: Path) -> Ctx:
    return Ctx(cfg={"seed": 0}, usage=Usage(), workers=1, run_dir=tmp, smoke=False)


def _records(n: int = 10) -> list[dict]:
    return [{"scenario_id": f"s{i}", "situation": f"situation {i}"} for i in range(n)]


def _write_check(tmp: Path, stage: str, *, n_records: int = 10,
                 labels: dict[str, dict] | None = None,
                 sampled: int | None = None) -> None:
    """Write the two artefacts a corpus_check leaves behind."""
    metrics: dict = {}
    if sampled is not None:
        metrics["sampled"] = sampled
    (tmp / f"{stage}_report.json").write_text(json.dumps({
        "stage": stage, "n_records": n_records, "fields": {"id": "scenario_id"},
        "properties": {"embedding_dedup": {"status": "reported", "gate": False,
                                           "metrics": metrics, "findings": []}},
        "pass": True,
    }), encoding="utf-8")
    with (tmp / f"{stage}_labels.jsonl").open("w", encoding="utf-8") as f:
        for rid, lab in (labels or {}).items():
            f.write(json.dumps({"record_id": rid, **lab}) + "\n")


def _stage(**over):
    # These fixtures are 10 records, so a 1-2 record removal is 10-20% and would trip the
    # shipped 5% ceiling. Tests about the ceiling itself set it explicitly.
    return {"name": "dedupe", "kind": "corpus_filter", "from": "chk",
            "drop_when": ["embedding_dup"], "max_drop_share": 0.25, **over}


def _run(tmp: Path, spec: dict, records: list[dict]):
    stage = OPERATORS["corpus_filter"](spec, {})
    ctx = _ctx(tmp)
    return stage.fn(ctx, records, None), ctx


def test_drops_exactly_the_labelled_records(tmp_path):
    _write_check(tmp_path, "chk",
                 labels={"s3": {"embedding_dup": True}, "s7": {"embedding_dup": True}})
    kept, ctx = _run(tmp_path, _stage(), _records())

    assert [r["scenario_id"] for r in kept] == [
        "s0", "s1", "s2", "s4", "s5", "s6", "s8", "s9"]
    entry = ctx.manifest_extra["corpus_filters"]["dedupe"]
    assert (entry["n_before"], entry["n_after"], entry["dropped"]) == (10, 8, 2)
    assert entry["by_reason"] == {"embedding_dup": 2}


def test_dropped_records_are_written_aside_with_their_reason(tmp_path):
    _write_check(tmp_path, "chk", labels={"s3": {"embedding_dup": True}})
    _run(tmp_path, _stage(), _records())

    rows = [json.loads(x) for x in
            (tmp_path / "dedupe_dropped.jsonl").read_text().splitlines() if x.strip()]
    assert [r["scenario_id"] for r in rows] == ["s3"]
    assert rows[0]["_dropped_for"] == ["embedding_dup"]


def test_a_check_that_wrote_no_labels_drops_nothing(tmp_path):
    """The dead-API-key case. No labels must mean no deletions, never the inverse."""
    _write_check(tmp_path, "chk", labels={})
    kept, ctx = _run(tmp_path, _stage(), _records())

    assert len(kept) == 10
    assert ctx.manifest_extra["corpus_filters"]["dedupe"]["dropped"] == 0


def test_labels_from_an_unrelated_property_are_ignored(tmp_path):
    _write_check(tmp_path, "chk", labels={"s1": {"quality_verdict": "drop"},
                                          "s2": {"embedding_dup": True}})
    kept, _ = _run(tmp_path, _stage(), _records())

    assert "s1" in [r["scenario_id"] for r in kept]
    assert "s2" not in [r["scenario_id"] for r in kept]


def test_refuses_a_sampled_property(tmp_path):
    _write_check(tmp_path, "chk", sampled=4,
                 labels={"s1": {"embedding_dup": True}})
    with pytest.raises(AssertionError, match="never saw"):
        _run(tmp_path, _stage(), _records())


def test_allow_partial_accepts_a_sampled_property(tmp_path):
    _write_check(tmp_path, "chk", sampled=4,
                 labels={"s1": {"embedding_dup": True}})
    kept, _ = _run(tmp_path, _stage(allow_partial=True), _records())
    assert len(kept) == 9


def test_refuses_a_removal_set_over_the_ceiling(tmp_path):
    _write_check(tmp_path, "chk",
                 labels={f"s{i}": {"embedding_dup": True} for i in range(6)})
    with pytest.raises(AssertionError, match="over max_drop_share"):
        _run(tmp_path, _stage(max_drop_share=0.25), _records())


def test_the_shipped_ceiling_is_five_percent(tmp_path):
    """No `max_drop_share:` in the config must still bound the damage."""
    _write_check(tmp_path, "chk", n_records=100,
                 labels={f"s{i}": {"embedding_dup": True} for i in range(6)})
    spec = {"name": "dedupe", "from": "chk", "drop_when": ["embedding_dup"]}
    with pytest.raises(AssertionError, match="over max_drop_share=5.0%"):
        _run(tmp_path, spec, _records(100))

    _write_check(tmp_path, "chk", n_records=100,
                 labels={f"s{i}": {"embedding_dup": True} for i in range(4)})
    kept, _ = _run(tmp_path, spec, _records(100))
    assert len(kept) == 96


def test_keep_when_is_rejected_at_build_time(tmp_path):
    with pytest.raises(AssertionError, match="deliberately unsupported"):
        OPERATORS["corpus_filter"](_stage(keep_when=["good"]), {})


def test_empty_drop_when_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match="at least one label key"):
        OPERATORS["corpus_filter"]({"name": "d", "from": "chk"}, {})


def test_refuses_when_the_check_saw_a_different_corpus(tmp_path):
    _write_check(tmp_path, "chk", n_records=99,
                 labels={"s1": {"embedding_dup": True}})
    with pytest.raises(AssertionError, match="cannot be matched"):
        _run(tmp_path, _stage(), _records())


def test_missing_report_names_the_ordering_mistake(tmp_path):
    with pytest.raises(AssertionError, match="must name a corpus_check stage"):
        _run(tmp_path, _stage(), _records())
