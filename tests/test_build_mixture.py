# ABOUTME: Offline unit tests for the mixture builder's sampling and config schema:
# ABOUTME: budget fill behaviour, pre-rendered source loading, and the sources mapping.

import json

from omegaconf import OmegaConf

from src.data.build_mixture import _fill, _take_rendered


def _rows(sizes):
    return [{"text": f"t{i}", "source": "s", "n_tokens": n} for i, n in enumerate(sizes)]


def test_fill_respects_budget():
    out = _fill(_rows([40, 40, 40]), budget=100, seed=0)
    assert sum(r["n_tokens"] for r in out) <= 100
    assert len(out) == 2


def test_fill_skips_oversized_rows_but_keeps_filling():
    out = _fill(_rows([500, 10, 10]), budget=25, seed=0)
    assert sum(r["n_tokens"] for r in out) == 20


def test_fill_is_seed_deterministic():
    a = _fill(_rows(range(1, 30)), budget=100, seed=7)
    b = _fill(_rows(range(1, 30)), budget=100, seed=7)
    assert a == b


def test_take_rendered_labels_source_and_respects_budget(tmp_path):
    path = tmp_path / "corpus.jsonl"
    with path.open("w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"doc{i}", "n_tokens": 30}) + "\n")
    out = _take_rendered(path, budget=100, seed=0, source="embodied")
    assert out and all(r["source"] == "embodied" for r in out)
    assert sum(r["n_tokens"] for r in out) <= 100


def test_mixture_configs_share_one_schema():
    for name in ("mixture_qwen36_20_80", "mixture_qwen36_10_90",
                 "mixture_qwen36_40_60", "mixture_qwen36_three_way"):
        cfg = OmegaConf.load(f"configs/data/{name}.yaml")
        sources = OmegaConf.to_container(cfg.sources, resolve=True)
        assert sources, name
        for spec in sources.values():
            assert set(spec) == {"path", "format", "tokens"}, name
            assert spec["format"] in ("messages", "rendered"), name
        assert int(cfg.tulu3_tokens) > 0, name
        assert int(cfg.max_seq_len) > 0, name
