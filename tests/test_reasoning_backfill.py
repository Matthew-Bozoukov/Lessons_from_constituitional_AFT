# ABOUTME: The base blend's reasoning traces belong to one family: the record a mixture carries,
# ABOUTME: how an arm inherits it from its base, and the train-time refusal of a mismatch.
from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from src.data.mixture import build_mixture as bm
from src.data.mixture import reasoning_backfill as rb
from src.model_profile import ModelProfile
from src.train.launch import check_trace_family


class _Tok:
    def apply_chat_template(self, messages, **kw):
        n = sum(len(str(m.get("content", ""))) + len(str(m.get("reasoning_content", ""))) for m in messages)
        return {"input_ids": list(range(n))}


def _rows():
    return [
        {"source": "lima", "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
        {"source": "lima", "messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
                                        {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]},
        {"source": "no_robots", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    ]


def test_family_is_read_through_the_model_registry():
    assert rb.family_of("qwen/qwen3.6-27b") == "qwen36"
    assert rb.family_of("Qwen/Qwen3.6-27B") == "qwen36"
    with pytest.raises(ValueError, match="no model profile"):
        rb.family_of("nobody/unregistered-9b")


def test_validate_backfill_names_the_sources_and_a_registered_generator():
    ok = {"model": "qwen/qwen3.6-27b", "judge": "google/gemini-3-flash-preview",
          "sources": ["lima"], "fraction": 0.5, "max_tokens": 6144}
    sources = {"lima": {"reasoning": "none"}, "no_robots": {"reasoning": "none"}}
    assert rb.validate_backfill(ok, sources)["family" if False else "model"] == "qwen/qwen3.6-27b"
    for bad, why in [({**ok, "sources": ["nope"]}, "unknown"), ({**ok, "fraction": 0}, "fraction"),
                     ({**ok, "max_tokens": 10}, "max_tokens"), ({**ok, "extra": 1}, "exactly"),
                     ({**ok, "model": "x/unknown"}, "no model profile")]:
        with pytest.raises(ValueError, match=why):
            rb.validate_backfill(bad, sources)


def test_select_draws_a_seeded_fraction_and_every_untraced_turn_of_a_drawn_row():
    rows = _rows()
    picked = rb.select(rows, ["lima"], 1.0, seed=0)
    assert picked == [(0, 1), (1, 1), (1, 3)], "every assistant turn of every lima row; no_robots untouched"
    assert rb.select(rows, ["lima"], 0.5, seed=0) == rb.select(rows, ["lima"], 0.5, seed=0)
    rows[1]["messages"][1]["reasoning_content"] = "already"
    assert (1, 1) not in rb.select(rows, ["lima"], 1.0, seed=0)


def test_splice_keeps_only_complete_judged_fitting_traces_and_never_mutates_input():
    rows = _rows()
    picked = [(0, 1), (1, 1), (1, 3), (2, 1)]
    generated = {
        (0, 1): {"status": "ok", "verdict": "yes", "trace": "think one"},
        (1, 1): {"status": "ok", "verdict": "no", "trace": "wrong way"},
        (1, 3): {"status": "truncated", "trace": "cut"},
        (2, 1): {"status": "ok", "verdict": "yes", "trace": "x" * 500},   # over the cap
    }
    out, report = rb.splice(rows, picked, generated, _Tok(), max_seq_len=40, render_kwargs={})
    assert out[0]["messages"][1]["reasoning_content"] == "think one"
    assert "reasoning_content" not in out[1]["messages"][1] and "reasoning_content" not in out[1]["messages"][3]
    assert "reasoning_content" not in out[2]["messages"][1]
    assert report["n_accepted"] == 1 and report["outcomes"] == {"accepted": 1, "judge_rejected": 1, "truncated": 1, "over_max_seq_len": 1}
    assert all("reasoning_content" not in m for r in rows for m in r["messages"]), "input rows untouched"
    block = rb.traces_block({"model": "qwen/qwen3.6-27b", "judge": "j", "sources": ["lima"], "fraction": 1.0}, out, report, 0)
    assert block["family"] == "qwen36" and block["turns"] == 1 and block["rows"] == 1


def test_a_published_base_says_whose_traces_it_carries_with_the_pre_record_fallback(monkeypatch):
    files = {}
    monkeypatch.setattr(rb, "_fetch_json", lambda repo, f, rev: files.get(f))
    assert rb.base_reasoning_traces("org/base", "sha") is None, "no record, no traces, no claim"
    files["enrichment_report.json"] = {"gen_model": "qwen/qwen3.6-27b", "judge_model": "j",
                                       "target_sources": ["lima"], "n_reasoning_turns": 1135}
    got = rb.base_reasoning_traces("org/base", "sha0123456789abc")
    assert got["family"] == "qwen36" and got["turns"] == 1135 and "pre-record" in got["provenance"]
    files["mixture_stats.json"] = {"reasoning_traces": {"model": "qwen/qwen3.6-27b", "family": "qwen36", "turns": 7}}
    assert rb.base_reasoning_traces("org/base", "sha")["turns"] == 7, "the built record wins over the fallback"
    inherited = rb.inherited_block(got, "org/base", "sha")
    assert inherited["inherited_from"] == {"repo": "org/base", "revision": "sha"}


def test_the_card_names_the_trace_generator_beside_the_filter_judge():
    cfg = OmegaConf.create({"seed": 0, "max_seq_len": 8192, "tokenizer": "Qwen/Qwen3.6-27B",
                            "hf": {"experiment": "e"}})
    traces = {"model": "qwen/qwen3.6-27b", "family": "qwen36", "turns": 1135, "judge": "google/gemini-3-flash-preview"}
    fields = bm._card_fields(cfg, "configs/data/mixture/x.yaml", "final", "files", None, None, traces)
    assert fields["models"] == ("base-blend reasoning traces: on-policy qwen/qwen3.6-27b (family qwen36, "
                                "1135 turns, judged by google/gemini-3-flash-preview)")
    assert json.loads(fields["generation_config"])["reasoning_traces"]["family"] == "qwen36"
    assert bm._card_fields(cfg, "x.yaml", "final", "files", None, None)["models"] == "base-blend reasoning traces: none"


def test_train_refuses_another_familys_traces_unless_told_to():
    qwen36 = ModelProfile.from_dict({"model": "Qwen/Qwen3.6-27B", "match": "qwen36"}, key="qwen36")
    other = ModelProfile.from_dict({"model": "openai/gpt-oss-20b", "match": "gptoss20b"}, key="gptoss20b")
    stats = {"reasoning_traces": {"model": "qwen/qwen3.6-27b", "family": "qwen36", "turns": 1135}}
    assert check_trace_family(stats, qwen36)["family_mismatch_allowed"] is False
    with pytest.raises(ValueError, match="on-policy for 'qwen36'.*being trained is 'gptoss20b'"):
        check_trace_family(stats, other)
    assert check_trace_family(stats, other, allow_mismatch=True)["family_mismatch_allowed"] is True
    assert check_trace_family(None, other) is None and check_trace_family({"reasoning_traces": None}, other) is None
    # a record without `family` (hand-written) is resolved through the registry
    assert check_trace_family({"reasoning_traces": {"model": "Qwen/Qwen3.6-27B"}}, qwen36)["family_mismatch_allowed"] is False


def test_the_base_blend_config_declares_its_generator_and_arm_configs_do_not():
    from pathlib import Path
    nosynth = OmegaConf.load("configs/data/mixture/nosynth.yaml")
    spec = rb.validate_backfill(OmegaConf.to_container(nosynth.reasoning_backfill, resolve=True),
                                OmegaConf.to_container(nosynth.sources, resolve=True))
    assert rb.family_of(spec["model"]) == "qwen36" and set(spec["sources"]) == {"tulu3_if", "self_oss_instruct", "lima"}
    for path in Path("configs/data/mixture").glob("*.yaml"):
        cfg = OmegaConf.load(path)
        if cfg.get("base_mixture") or cfg.get("base"):
            assert cfg.get("reasoning_backfill") is None, f"{path.name}: an arm inherits its base's traces"
