# ABOUTME: MASK runner: the even subsample (stratified_counts / sample_data), the batched-judge
# ABOUTME: exchange (batch_judge over a stubbed run_batch), and the config keys the runner reads.
"""Offline tests for the MASK eval's src-side logic (the vendored harness is exercised by the
smoke on a served target, not here)."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from src.eval.misalignment.mask import runner

CFG = yaml.safe_load(open("configs/eval/mask.yaml", encoding="utf-8"))
TEST_DATA = runner._HARNESS / "test_csv_data"
FULL_DATA = runner._HARNESS / "csv_data"


# --- stratified_counts ------------------------------------------------------------------

def test_even_split_caps_the_small_archetype_and_spreads_the_rest():
    available = {"continuations": 854, "disinformation": 201, "doubling_down_known_facts": 225,
                 "known_facts": 316, "provided_facts": 891, "statistics": 108}
    counts = runner.stratified_counts(available, 1000)
    assert sum(counts.values()) == 1000
    assert counts["statistics"] == 108            # all it has
    rest = [v for a, v in counts.items() if a != "statistics"]
    assert max(rest) - min(rest) <= 1 and min(rest) >= 178
    assert all(counts[a] <= available[a] for a in available)


def test_two_small_archetypes_cap_in_turn():
    counts = runner.stratified_counts({"a": 5, "b": 20, "c": 100, "d": 100}, 100)
    assert counts == {"a": 5, "b": 20, "c": 38, "d": 37} or counts == {"a": 5, "b": 20, "c": 37, "d": 38}
    assert sum(counts.values()) == 100


def test_fewer_rows_than_archetypes_and_overflow():
    assert sum(runner.stratified_counts({"a": 9, "b": 9, "c": 9}, 2).values()) == 2
    assert runner.stratified_counts({"a": 9, "b": 9}, 18) == {"a": 9, "b": 9}
    with pytest.raises(ValueError, match="exceeds"):
        runner.stratified_counts({"a": 9, "b": 9}, 19)


def test_the_real_data_holds_what_the_config_asks_for():
    available = {p.stem: sum(1 for _ in csv.DictReader(p.open(encoding="utf-8", newline="")))
                 for p in FULL_DATA.glob("*.csv") if p.stem in runner.ARCHETYPES}
    assert set(available) == set(runner.ARCHETYPES)
    counts = runner.stratified_counts(available, int(CFG["subsample"]))
    assert sum(counts.values()) == int(CFG["subsample"])


# --- sample_data ------------------------------------------------------------------------

def test_sample_data_draws_evenly_seeded_and_verbatim(tmp_path):
    rec = runner.sample_data(TEST_DATA, tmp_path / "d", 12, seed=0)
    assert set(rec) == set(runner.ARCHETYPES)
    assert sum(v["n"] for v in rec.values()) == 12 and all(v["n"] == 2 for v in rec.values())
    assert all(v["of"] == 5 for v in rec.values())
    # Rows are the original rows, in the original order, with the original header.
    src = list(csv.DictReader((TEST_DATA / "continuations.csv").open(encoding="utf-8", newline="")))
    out = list(csv.DictReader((tmp_path / "d" / "continuations.csv").open(encoding="utf-8", newline="")))
    assert [r["task_id"] for r in out] == rec["continuations"]["task_ids"]
    assert all(r in src for r in out) and [src.index(r) for r in out] == sorted(src.index(r) for r in out)
    assert list(out[0].keys()) == list(src[0].keys())
    # Same seed, same draw; a different seed, a different one (somewhere).
    again = runner.sample_data(TEST_DATA, tmp_path / "e", 12, seed=0)
    other = runner.sample_data(TEST_DATA, tmp_path / "f", 12, seed=1)
    assert [v["task_ids"] for v in again.values()] == [v["task_ids"] for v in rec.values()]
    assert [v["task_ids"] for v in other.values()] != [v["task_ids"] for v in rec.values()]


def test_sample_data_without_n_copies_everything(tmp_path):
    rec = runner.sample_data(TEST_DATA, tmp_path / "d", None, seed=0)
    assert all(v["n"] == v["of"] == 5 for v in rec.values())


# --- batch_judge ------------------------------------------------------------------------

def test_batch_judge_writes_what_the_batch_answered_and_counts_the_rest(tmp_path, monkeypatch):
    exchange = tmp_path / "judge_exchange"
    reqs = [{"custom_id": f"k{i}", "body": {"messages": [{"role": "user", "content": str(i)}],
                                             "max_completion_tokens": 500}} for i in range(4)]
    (tmp_path / "judge_exchange.requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in reqs))
    seen = {}

    def fake_run_batch(model, requests, stage, state_path, collect, **kw):
        seen.update(model=model, n=len(requests), stage=stage)
        for cid in list(requests)[:3]:      # the batch answers three of four
            collect(cid, {"choices": [{"message": {"content": f"Answer: {cid}"}}]})

    monkeypatch.setattr(runner, "run_batch", fake_run_batch)
    stats = runner.batch_judge(exchange, "google/gemini-3-flash-preview")
    assert seen == {"model": "google/gemini-3-flash-preview", "n": 4, "stage": "mask_judge"}
    assert stats == {"requested": 4, "returned": 3, "fallback_live": 1}
    got = [json.loads(l) for l in (tmp_path / "judge_exchange.results.jsonl").read_text().splitlines()]
    assert {g["custom_id"]: g["content"] for g in got} == {f"k{i}": f"Answer: k{i}" for i in range(3)}


# --- the config -------------------------------------------------------------------------

def test_the_config_declares_the_runner_knobs():
    assert CFG["judge_batch"] is False and int(CFG["subsample"]) == 1000
    assert int(CFG["gen_concurrency"]) >= 10 and "seed" in CFG
    # The generation ceiling is the family's verified cap: asking for more only queues.
    qwen36 = yaml.safe_load(open("configs/models/qwen36.yaml", encoding="utf-8"))
    assert int(CFG["gen_concurrency"]) <= int(qwen36["serving"]["max_num_seqs"])


def test_every_harness_patch_is_listed():
    listed = (runner._HARNESS_ROOT.parent / "VENDORED_FROM.txt").read_text()
    for stage in ("generate_responses.py", "evaluate.py", "metric.py", "process_metrics.py"):
        src = (runner._HARNESS / stage).read_text()
        assert "PATCH (teaching_claude_why)" in src, f"{stage} carries no patch marker"
        assert stage in listed, f"{stage} is patched but not listed in VENDORED_FROM.txt"
    for var in ("MASK_DATA_DIR", "MASK_GEN_CONCURRENCY", "MASK_JUDGE_TRANSPORT", "MASK_JUDGE_EXCHANGE"):
        assert var in listed
