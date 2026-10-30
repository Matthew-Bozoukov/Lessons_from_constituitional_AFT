# ABOUTME: Offline tests for corpus pools (src/data/mixture/split.py): the rows an earlier mixture took
# ABOUTME: are found by content, the rest is dealt evenly per stratum, and a mixture source selects one pool.
from __future__ import annotations

from collections import Counter

import pytest

from src.data.mixture import split


def _row(sid: str, trait: str, ai: str, text: str | None = None) -> dict:
    return {"messages": [{"role": "user", "content": text or sid},
                         {"role": "assistant", "content": "a", "reasoning_content": "r"}],
            "metadata": {"scenario_id": sid, "trait_id": trait, "ai": ai}}


def _corpus() -> list[dict]:
    rows = []
    for t in ("t1", "t6"):
        for prefix in ("", "r2_", "r3_"):
            for j in range(7):
                ai = ("self", "other")[j % 2] if t == "t6" else ("none", "open", "self")[j % 3]
                rows.append(_row(f"{prefix}{t}_b{j:02d}_s000", t, ai))
    return rows


def test_rows_taken_by_a_mixture_are_found_by_their_conversation():
    corpus = _corpus()
    mixture = [{"messages": corpus[3]["messages"], "source": "da"},
               {"messages": corpus[10]["messages"], "source": "da"},
               {"messages": [{"role": "user", "content": "replay"}], "source": "nemotron_chat"}]
    assert split.rows_taken_by(corpus, mixture, "da") == {
        corpus[3]["metadata"]["scenario_id"], corpus[10]["metadata"]["scenario_id"]}
    stranger = mixture + [{"messages": [{"role": "user", "content": "elsewhere"}], "source": "da"}]
    with pytest.raises(ValueError, match="matches 0 corpus rows"):
        split.rows_taken_by(corpus, stranger, "da")


def test_pools_are_disjoint_cover_the_corpus_and_split_every_stratum_within_one():
    corpus = _corpus()
    taken = {r["metadata"]["scenario_id"] for r in corpus[::4]}
    labels = split.assign_pools(corpus, taken, ("B", "C"), seed=0)
    assert set(labels) == {r["metadata"]["scenario_id"] for r in corpus}
    assert {s for s, lab in labels.items() if lab == "A"} == taken
    counts = split.split_counts(corpus, labels)
    for trait in ("t1", "t6"):
        assert abs(counts["by_trait"]["B"][trait] - counts["by_trait"]["C"][trait]) <= 1
    strata = set(counts["by_stratum"]["B"]) | set(counts["by_stratum"]["C"])
    for s in strata:
        assert abs(counts["by_stratum"]["B"].get(s, 0) - counts["by_stratum"]["C"].get(s, 0)) <= 1, s
    assert abs(counts["rows"]["B"] - counts["rows"]["C"]) <= 1
    # deterministic, and the seed is what moves it
    assert split.assign_pools(corpus, taken, ("B", "C"), seed=0) == labels
    assert split.assign_pools(corpus, taken, ("B", "C"), seed=1) != labels


def test_the_split_round_trips_and_selects_one_pool(tmp_path):
    corpus = _corpus()
    labels = split.assign_pools(corpus, set(), ("B", "C"))
    path = split.write_split(tmp_path / "splits" / "x.json", labels, {"seed": 0})
    back, meta = split.load_split(path)
    assert back == labels and meta == {"seed": 0}
    b = split.select(corpus, back, "B", name="da")
    assert {split.row_key(r["messages"]) for r in b}.isdisjoint(
        split.row_key(r["messages"]) for r in split.select(corpus, back, "C", name="da"))
    assert len(b) == Counter(labels.values())["B"]
    with pytest.raises(ValueError, match="carry no split label"):
        split.select(corpus + [_row("r9_t1_b00_s000", "t1", "none")], back, "B", name="da")
    with pytest.raises(ValueError, match="no row is in pool"):
        split.select(corpus, back, "D", name="da")


def test_a_mixture_source_draws_only_from_its_pool(monkeypatch, tmp_path):
    """`split: {file, keep}` on a `dataset:` source: the labels are read from the same repo
    at the same pinned revision, and only that pool's rows reach the fill."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    import src.infra.huggingface as hf
    from src.data.mixture import build_mixture as bm

    corpus = _corpus()
    labels = split.assign_pools(corpus, {corpus[0]["metadata"]["scenario_id"]}, ("B", "C"))
    path = split.write_split(tmp_path / "x.json", labels, {"seed": 0})
    monkeypatch.setattr(hf, "hf_api", lambda: SimpleNamespace(
        repo_info=lambda repo, repo_type, revision: SimpleNamespace(sha="c0ffee" * 6)))
    monkeypatch.setattr(hf, "hf_token", lambda: "offline")
    asked = {}
    monkeypatch.setattr(hf, "hf_download", lambda repo, f, repo_type, revision:
                        asked.update(f=f, revision=revision) or str(path))
    monkeypatch.setattr(bm, "load_dataset", lambda repo, revision, split, token: corpus)

    class _Tok:
        def apply_chat_template(self, msgs, **kw):
            return {"input_ids": [0] * 10}

    spec = {"dataset": "org/2026-10-30-da-synth", "reasoning": "native", "all": True,
            "balance_by": "trait_id", "split": {"file": "splits/x.json", "keep": "C"}}
    got, _ = bm._take_interchange(_Tok(), OmegaConf.create({"max_seq_len": 99}), "da", spec,
                                  budget=("all", 0), seed=0, render_kwargs={})
    want = {split.row_key(r["messages"]) for r in corpus
            if labels[r["metadata"]["scenario_id"]] == "C"}
    assert {split.row_key(r["messages"]) for r in got} == want
    assert asked == {"f": "splits/x.json", "revision": "c0ffee" * 6}
    assert spec["split"]["selected_rows"] == len(want)
    # An unpublished split (`path:`) is read from disk, never from the repo.
    asked.clear()
    local = {**spec, "split": {"path": str(path), "keep": "B"}}
    got, _ = bm._take_interchange(_Tok(), OmegaConf.create({"max_seq_len": 99}), "da", local,
                                  budget=("all", 0), seed=0, render_kwargs={})
    assert not asked and len(got) == sum(1 for v in labels.values() if v == "B")
    with pytest.raises(ValueError, match="needs `dataset:`"):
        bm._take_interchange(_Tok(), OmegaConf.create({"max_seq_len": 99}), "da",
                             {"path": "x.jsonl", "reasoning": "native",
                              "split": {"file": "x", "keep": "B"}},
                             budget=("all", 0), seed=0, render_kwargs={})


def test_a_mixture_from_an_unpublished_split_is_never_published(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from src.data.mixture import build_mixture as bm

    monkeypatch.setattr(bm.AutoTokenizer, "from_pretrained", lambda *_a, **_k: object())
    monkeypatch.setattr(bm, "model_profile", lambda *_a: SimpleNamespace(render_kwargs={}))
    over = ["sources.da.split.path=x.json", "sources.da.split.keep=B",
            f"output_dir={tmp_path}"]
    with pytest.raises(ValueError, match="LOCAL split file"):
        bm.main("configs/data/mixture/da.yaml", *over)
