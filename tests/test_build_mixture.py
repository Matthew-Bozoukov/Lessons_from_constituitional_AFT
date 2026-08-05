# ABOUTME: Offline unit tests for the mixture builder's sampling and config schema:
# ABOUTME: budget fill behaviour, source loading and dispatch, and the sources mapping.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.mixture.build_mixture import _fill, _take_rendered, _usable, main


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


def test_take_rendered_passes_supervise_through(tmp_path):
    path = tmp_path / "corpus.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"text": "self-reflect", "n_tokens": 10,
                            "supervise": "final"}) + "\n")
        f.write(json.dumps({"text": "plain", "n_tokens": 10}) + "\n")
    out = {r["text"]: r for r in _take_rendered(path, budget=100, seed=0, source="model_eval_model")}
    assert out["self-reflect"]["supervise"] == "final"
    assert "supervise" not in out["plain"], "rows without the field stay unchanged"


def test_usable_accepts_wellformed_and_rejects_malformed():
    good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert _usable(good)
    assert not _usable([])                                              # too short
    assert not _usable(good[:1])                                        # no assistant end
    assert not _usable(good + [{"role": "user", "content": "q2"}])      # ends on user
    assert not _usable([{"role": "user", "content": "q"},
                        {"role": "assistant", "content": ""}])          # empty content
    assert not _usable([{"role": "tool", "content": "x"},
                        {"role": "assistant", "content": "a"}])         # unsupported role


def test_main_rejects_legacy_tulu3_schema(tmp_path):
    cfg = tmp_path / "legacy.yaml"
    cfg.write_text(
        "seed: 0\ntokenizer: t\nsources: {}\n"
        "tulu3_repo: allenai/tulu-3-sft-mixture\ntulu3_tokens: 1000\n"
        "max_seq_len: 2048\noutput_dir: out\n")
    with pytest.raises(AssertionError, match="folded into"):
        main(config=str(cfg))


def test_mixture_configs_share_one_schema():
    # tulu_control.yaml is prepare_tulu's config, not a build_mixture config.
    configs = [p for p in sorted(Path("configs/data/mixture").glob("*.yaml"))
               if p.name != "tulu_control.yaml"]
    assert configs, "no mixture configs found"
    for path in configs:
        cfg = OmegaConf.load(path)
        name = path.name
        assert "tulu3_repo" not in cfg and "tulu3_tokens" not in cfg, name
        sources = OmegaConf.to_container(cfg.sources, resolve=True)
        assert sources, name
        for sname, spec in sources.items():
            if "repo" in spec:
                assert set(spec) <= {"repo", "split", "tokens", "shuffle_buffer",
                                     "reasoning"}, (name, sname)
                # What the data carries is part of the scientific record, never guessed.
                assert spec.get("reasoning") in ("native", "none", "strip"), (name, sname)
            else:
                assert set(spec) == {"path", "format", "tokens"}, (name, sname)
                assert spec["format"] in ("messages", "rendered"), (name, sname)
            assert int(spec["tokens"]) > 0, (name, sname)
        assert int(cfg.max_seq_len) > 0, name
