# ABOUTME: Offline unit tests for the mixture builder's sampling and config schema:
# ABOUTME: budget fill behaviour, source loading and dispatch, and the sources mapping.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.mixture.build_mixture import (
    _budget,
    _fill,
    _fill_budget,
    _source_stats,
    _take_interchange,
    _validate_interchange,
    _write_rows,
    main,
    stratified_subset,
)


class _StubTok:
    """Counts tokens as words — enough to exercise budgets and caps offline.

    Mirrors the real contract: tokenize=True returns a BatchEncoding-like MAPPING
    (whose len() is its key count, not the token count — the bug the 2026-08-06 smoke
    run caught), so callers must index ["input_ids"].
    """

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            return_dict=False, **kw):
        assert tokenize is True and return_dict is True
        words = " ".join(m.get("content") or "" for m in messages).split()
        return {"input_ids": words, "attention_mask": [1] * len(words)}


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




def test_budget_requires_exactly_one_of_tokens_or_examples():
    assert _budget("s", {"tokens": 100}, scale=1) == ("tokens", 100)
    assert _budget("s", {"examples": 40}, scale=1) == ("examples", 40)
    with pytest.raises(ValueError, match="exactly one"):
        _budget("s", {"tokens": 100, "examples": 40}, scale=1)
    with pytest.raises(ValueError, match="exactly one"):
        _budget("s", {}, scale=1)


def test_budget_smoke_scale_divides_and_floors_at_one():
    assert _budget("s", {"examples": 100}, scale=20) == ("examples", 5)
    assert _budget("s", {"examples": 3}, scale=20) == ("examples", 1)


def test_fill_budget_examples_takes_exactly_n_and_is_deterministic():
    a = _fill_budget(_rows([10] * 8), ("examples", 5), seed=7)
    b = _fill_budget(_rows([10] * 8), ("examples", 5), seed=7)
    assert len(a) == 5 and a == b


def test_fill_budget_examples_fails_loudly_when_source_is_short():
    with pytest.raises(AssertionError, match="only 3"):
        _fill_budget(_rows([10, 10, 10]), ("examples", 5), seed=0)



def test_source_stats_reports_example_and_token_shares_separately():
    # A 20/80-by-examples mixture whose synthetic rows are 4x longer: the example share
    # must read 20/80 even though the token share reads 50/50 — both are recorded.
    rows = ([{"source": "mem", "n_tokens": 400}] * 2
            + [{"source": "replay", "n_tokens": 100}] * 8)
    stats = _source_stats(rows)
    assert stats["mem"]["share_pct_examples"] == 20.0
    assert stats["replay"]["share_pct_examples"] == 80.0
    assert stats["mem"]["share_pct_tokens"] == 50.0
    assert stats["replay"]["share_pct_tokens"] == 50.0
    assert stats["mem"]["examples"] == 2 and stats["mem"]["tokens"] == 800



def test_main_rejects_legacy_tulu3_schema(tmp_path):
    cfg = tmp_path / "legacy.yaml"
    cfg.write_text(
        "seed: 0\ntokenizer: t\nsources: {}\n"
        "tulu3_repo: allenai/tulu-3-sft-mixture\ntulu3_tokens: 1000\n"
        "max_seq_len: 2048\noutput_dir: out\n")
    with pytest.raises(AssertionError, match="folded into"):
        main(config=str(cfg))


def test_mixture_configs_share_one_schema():
    from src.data.mixture.sources import SOURCES

    configs = sorted(Path("configs/data/mixture").glob("*.yaml"))
    assert configs, "no mixture configs found"
    for path in configs:
        cfg = OmegaConf.load(path)
        name = path.name
        assert "tulu3_repo" not in cfg and "tulu3_tokens" not in cfg, name
        sources = OmegaConf.to_container(cfg.sources, resolve=True)
        assert sources, name
        for sname, spec in sources.items():
            assert set(spec) <= {"source", "repo", "path", "config", "split", "tokens",
                                 "examples", "shuffle_buffer", "reasoning", "synthetic",
                                 "balance_by"}, (name, sname)
            # What the data carries is part of the scientific record, never guessed —
            # and the legacy kinds (strip / format: rendered) are gone (2026-08-07).
            assert spec.get("reasoning") in ("native", "none"), (name, sname)
            if not ("repo" in spec or "path" in spec):
                assert (spec.get("source") or sname) in SOURCES, (name, sname)
            # Exactly one budget kind per source (the builder's _budget contract).
            declared = [k for k in ("tokens", "examples") if spec.get(k) is not None]
            assert len(declared) == 1, (name, sname, declared)
            assert int(spec[declared[0]]) > 0, (name, sname)
        assert int(cfg.max_seq_len) > 0, name


# --------------------------------------------------------------------------------------
# Interchange mode
# --------------------------------------------------------------------------------------

def _icfg(tmp_path, max_seq_len=100):
    return OmegaConf.create({"max_seq_len": max_seq_len, "shuffle_buffer": 10})


def test_take_interchange_local_messages_path(tmp_path):
    path = tmp_path / "rows.jsonl"
    with path.open("w") as f:
        for i in range(6):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"}]}) + "\n")
    rows, kind = _take_interchange(
        _StubTok(), _icfg(tmp_path), "plain", {"path": str(path), "reasoning": "none"},
        ("examples", 4), seed=0, render_kwargs={})
    assert kind == "none" and len(rows) == 4
    assert all(r["source"] == "plain" and "messages" in r and "text" not in r for r in rows)


def test_take_interchange_resolves_adapter_by_name_and_path(tmp_path):
    # The difficult_advice adapter maps synth stage-6 records to messages.
    path = tmp_path / "stage6.jsonl"
    with path.open("w") as f:
        for i in range(3):
            f.write(json.dumps({"system": "s", "user": f"u{i}", "reasoning": f"r{i}",
                                "response": f"a{i}", "trait_id": "t"}) + "\n")
    rows, kind = _take_interchange(
        _StubTok(), _icfg(tmp_path), "difficult_advice",
        {"source": "difficult_advice", "path": str(path), "reasoning": "native"},
        ("examples", 3), seed=0, render_kwargs={})
    assert kind == "native" and len(rows) == 3
    assert rows[0]["messages"][-1]["reasoning_content"].startswith("r")


def test_take_interchange_refuses_legacy_kinds_and_unknown_adapters(tmp_path):
    with pytest.raises(ValueError, match="removed 2026-08-07"):
        _take_interchange(_StubTok(), _icfg(tmp_path), "s",
                          {"path": "x", "reasoning": "strip"}, ("examples", 1), 0, {})
    with pytest.raises(ValueError, match="removed 2026-08-07"):
        _take_interchange(_StubTok(), _icfg(tmp_path), "s",
                          {"path": "x", "format": "rendered", "reasoning": "none"},
                          ("examples", 1), 0, {})
    with pytest.raises(ValueError, match="unknown adapter"):
        _take_interchange(_StubTok(), _icfg(tmp_path), "not_a_source",
                          {"reasoning": "none"}, ("examples", 1), 0, {})


def test_take_interchange_caps_on_rendered_length(tmp_path):
    path = tmp_path / "rows.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"messages": [
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "ok"}]}) + "\n")
        f.write(json.dumps({"messages": [
            {"role": "user", "content": "w " * 300},
            {"role": "assistant", "content": "ok"}]}) + "\n")
    rows, _ = _take_interchange(
        _StubTok(), _icfg(tmp_path, max_seq_len=50), "s",
        {"path": str(path), "reasoning": "none"}, ("examples", 1), 0, {})
    assert len(rows) == 1 and rows[0]["messages"][0]["content"] == "short"


def test_validate_interchange_enforces_declarations():
    native_ok = [{"messages": [{"role": "user", "content": "q"},
                               {"role": "assistant", "content": "a",
                                "reasoning_content": "because"}], "source": "s"}]
    _validate_interchange("s", "native", native_ok)
    with pytest.raises(AssertionError, match="no real"):
        _validate_interchange("s", "native", [{"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}], "source": "s"}])
    with pytest.raises(AssertionError, match="mislabels"):
        _validate_interchange("s", "none", native_ok)


def test_write_rows_messages_roundtrip_keeps_supervise(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}],
             "source": "s", "n_tokens": 5, "supervise": "final"},
            {"messages": [{"role": "user", "content": "q2"},
                          {"role": "assistant", "content": "a2"}],
             "source": "s", "n_tokens": 5}]
    path = tmp_path / "m.jsonl"
    _write_rows(path, rows)
    written = [json.loads(line) for line in path.open()]
    assert written[0]["supervise"] == "final" and "supervise" not in written[1]
    assert all("n_tokens" not in w for w in written), "counters must not leak into artifacts"


def test_stratified_subset_holds_proportions_and_is_deterministic():
    rows = ([{"source": "a", "n_tokens": 1}] * 60
            + [{"source": "b", "n_tokens": 1}] * 30
            + [{"source": "c", "n_tokens": 1}] * 10)
    picked, quota = stratified_subset(list(rows), 50, seed=3)
    again, _ = stratified_subset(list(rows), 50, seed=3)
    assert len(picked) == 50 and [r["source"] for r in picked] == [r["source"] for r in again]
    counts = {s: sum(1 for r in picked if r["source"] == s) for s in "abc"}
    assert counts == {"a": 30, "b": 15, "c": 5}
    assert quota == {"a": 30, "b": 15, "c": 5}
    with pytest.raises(AssertionError):
        stratified_subset(rows[:5], 10, seed=0)


def test_main_requires_filter_for_synthetic_flags(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "seed: 0\ntokenizer: t\nmax_seq_len: 100\noutput_dir: out\n"
        "sources:\n  da:\n    path: x\n    reasoning: native\n    synthetic: true\n"
        "    examples: 1\n")
    with pytest.raises(ValueError, match="no `filter:` block"):
        main(config=str(cfg))


# --------------------------------------------------------------------------------------
# The render bridge (real tokenizer, cached-only): moving rendering from build time to
# train time must not change a byte of what the model trains on.
# --------------------------------------------------------------------------------------

@pytest.mark.tokenizer
def test_train_time_render_matches_legacy_build_time_render():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.6-27B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.6 tokenizer not in the local HF cache")
    from src.train.masking import build_labels
    from src.model_profile import model_profile, think_census

    profile = model_profile("Qwen/Qwen3.6-27B")
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},                       # no reasoning
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2",
             "reasoning_content": "thinking hard"}]                       # real trace

    # The reference render: what the legacy build-time path produced (verified when the
    # two paths coexisted; the legacy renderer is deleted, the byte-contract remains).
    legacy = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                     **profile.render_kwargs)
    assert legacy.count("<think>") == 2, "preserve policy: a think block on EVERY turn" 
    # train_lora's map: strip the None padding HF's json loader adds, then render with
    # the profile kwargs — the exact expression in src/train/train_lora.py.
    padded = [{**m, "reasoning_content": m.get("reasoning_content"),
               "tool_calls": None} for m in msgs]
    train_time = tok.apply_chat_template(
        [{k: v for k, v in m.items() if v is not None} for m in padded],
        tokenize=False, add_generation_prompt=False, **profile.render_kwargs)
    assert train_time == legacy, "train-time render diverged from the legacy build-time render"

    census = think_census([train_time])
    assert census == {"turns": 2, "real": 1, "empty": 1, "absent": 0}

    # The generation-boundary mask on the rendered row: the whole empty marker is
    # forced context (never supervised); the real trace and its close are supervised.
    out = build_labels(train_time, tok, max_length=100000, profile=profile)
    supervised = tok.decode([i for i, l in zip(out["input_ids"], out["labels"])
                             if l != -100])
    assert profile.empty_think not in supervised and "thinking hard" in supervised
    assert "a1" in supervised and "a2" in supervised


def test_take_interchange_lifts_supervise_from_metadata(tmp_path):
    # Synthdoc stage-5 exports carry supervise under metadata; dropping it would
    # silently train the flawed first response of self-reflection records.
    path = tmp_path / "stage5.jsonl"
    path.write_text(json.dumps({
        "messages": [{"role": "user", "content": "q"},
                     {"role": "assistant", "content": "flawed",
                      "reasoning_content": "r1"},
                     {"role": "user", "content": "reflect"},
                     {"role": "assistant", "content": "better",
                      "reasoning_content": "r2"}],
        "metadata": {"supervise": "final", "record_id": "x"}}) + "\n")
    rows, _ = _take_interchange(
        _StubTok(), _icfg(tmp_path), "self_reflection",
        {"path": str(path), "reasoning": "native"}, ("examples", 1), 0, {})
    assert rows[0]["supervise"] == "final"


def test_balance_by_takes_even_quotas_and_lifts_metadata_key(tmp_path):
    # Absorbs balanced_subset.py: 7 examples over 3 traits -> quotas 3/2/2, exact and
    # deterministic; the group key reads top-level or from metadata.
    path = tmp_path / "pool.jsonl"
    with path.open("w") as f:
        for t, n in (("t1", 5), ("t2", 4), ("t3", 3)):
            for i in range(n):
                where = {"trait_id": t} if i % 2 else {"metadata": {"trait_id": t}}
                f.write(json.dumps({**where, "messages": [
                    {"role": "user", "content": f"{t}q{i}"},
                    {"role": "assistant", "content": f"{t}a{i}"}]}) + "\n")
    spec = {"path": str(path), "reasoning": "none", "balance_by": "trait_id"}
    rows, _ = _take_interchange(_StubTok(), _icfg(tmp_path), "da", spec,
                                ("examples", 7), seed=1, render_kwargs={})
    again, _ = _take_interchange(_StubTok(), _icfg(tmp_path), "da", spec,
                                 ("examples", 7), seed=1, render_kwargs={})
    assert [r["messages"] for r in rows] == [r["messages"] for r in again]
    counts = {}
    for r in rows:
        t = r["messages"][0]["content"][:2]
        counts[t] = counts.get(t, 0) + 1
    assert sorted(counts.values()) == [2, 2, 3] and len(rows) == 7


def test_balance_by_fails_loudly_when_a_group_is_short(tmp_path):
    path = tmp_path / "pool.jsonl"
    with path.open("w") as f:
        for t, n in (("t1", 10), ("t2", 1)):
            for i in range(n):
                f.write(json.dumps({"trait_id": t, "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"}]}) + "\n")
    with pytest.raises(AssertionError, match="quota"):
        _take_interchange(_StubTok(), _icfg(tmp_path), "da",
                          {"path": str(path), "reasoning": "none",
                           "balance_by": "trait_id"},
                          ("examples", 8), seed=0, render_kwargs={})


def test_balance_by_refuses_streams_and_token_budgets(tmp_path):
    with pytest.raises(ValueError, match="whole pool"):
        _take_interchange(_StubTok(), _icfg(tmp_path), "no_robots",
                          {"reasoning": "none", "balance_by": "trait_id"},
                          ("examples", 4), seed=0, render_kwargs={})
    path = tmp_path / "pool.jsonl"
    path.write_text(json.dumps({"trait_id": "t", "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}]}) + "\n")
    with pytest.raises(AssertionError, match="examples"):
        _take_interchange(_StubTok(), _icfg(tmp_path), "da",
                          {"path": str(path), "reasoning": "none",
                           "balance_by": "trait_id"},
                          ("tokens", 100), seed=0, render_kwargs={})


def test_the_base_blend_keeps_its_proportions_as_the_synthetic_share_grows():
    """The whole point of a fixed base: only the synthetic share varies across a ladder.

    Earlier arms replaced the replay portion with a single source, so `da-10` and `da-40`
    differed in their replay composition too and neither was a clean control for the
    other. Here a source that is 27.79% of the base is 27.79% x (100-pct)% of every
    mixture built from it.
    """
    from src.data.mixture.build_mixture import _base_sources, blend

    base = _base_sources("configs/data/mixture/0.yaml")
    base_total = sum(s["examples"] for s in base.values())
    base_share = base["no_robots"]["examples"] / base_total

    for pct in (0, 10, 20, 40):
        synth = {"da": {"path": "x", "reasoning": "native", "examples": 1}} if pct else {}
        out = blend(base, synth, pct, 10_000)
        total = sum(s["examples"] for s in out.values())
        assert abs(total - 10_000) <= len(out)          # rounding only
        assert round(100 * out.get("da", {}).get("examples", 0) / total) == pct
        assert abs(out["no_robots"]["examples"] / total
                   - base_share * (100 - pct) / 100) < 0.001


def test_a_synthetic_share_and_synthetic_sources_must_agree():
    from src.data.mixture.build_mixture import blend

    base = {"tulu3": {"examples": 100}}
    with pytest.raises(AssertionError, match="do not agree"):
        blend(base, {}, 10, 1000)                        # a share with nothing to fill it
    with pytest.raises(AssertionError, match="do not agree"):
        blend(base, {"da": {"examples": 1}}, 0, 1000)     # a source with no share


def test_several_synthetic_styles_split_the_share_by_their_declared_ratio():
    """`da-par-20` is 20% synthetic; the styles' own budgets set the split within it."""
    from src.data.mixture.build_mixture import blend

    out = blend({"tulu3": {"examples": 100}},
                {"da": {"examples": 3}, "par": {"examples": 1}}, 20, 1000)
    assert out["da"]["examples"] == 150 and out["par"]["examples"] == 50
    assert out["tulu3"]["examples"] == 800


def test_a_base_config_may_not_itself_carry_a_synthetic_share(tmp_path):
    from src.data.mixture.build_mixture import _base_sources

    bad = tmp_path / "bad.yaml"
    bad.write_text("sources:\n  da:\n    examples: 10\n    synthetic: true\n")
    with pytest.raises(AssertionError, match="used as a BASE blend"):
        _base_sources(str(bad))


def test_a_build_refuses_a_stem_its_synthetic_sources_do_not_spell(tmp_path, monkeypatch):
    """Belt and braces with the lint: an ad-hoc config cannot build a mixture named for
    corpora it does not contain."""
    from src.data.mixture import build_mixture as bm

    cfg = tmp_path / "da-par.yaml"
    cfg.write_text(
        "seed: 0\ntokenizer: x\nmax_seq_len: 8\ntotal_examples: 100\nsynthetic_pct: 20\n"
        "base: configs/data/mixture/0.yaml\noutput_dir: " + str(tmp_path) + "\n"
        "sources:\n  da:\n    path: x.jsonl\n    reasoning: native\n    examples: 1\n")
    # stop before any real loading: the name check is the first thing after config parse
    monkeypatch.setattr(bm, "AutoTokenizer", None)
    with pytest.raises(AssertionError, match="stem must be `da`"):
        bm.main(str(cfg))
