# ABOUTME: Offline tests for two `llm_tagged` engine features added 2026-08-26: a fallback
# ABOUTME: model on provider rejection (with the answering model stamped on the record), and
# ABOUTME: the per-stage failure tally by exception type that lands in the run manifest.
# ABOUTME: Run: uv run pytest tests/test_llm_tagged_fallback.py -q

from __future__ import annotations

import pytest

from src.data.synth.stage_operators import op_llm_tagged
from src.data.synth.stage_runtime import Ctx, Usage, resilient, run_items
from src.infra.endpoints.openrouter import ChatResult, ProviderRejectionError


class _Client:
    """Answers for the fallback model; the primary is 'moderated' on chosen prompts."""

    def __init__(self, reject_when):
        self.reject_when = reject_when
        self.calls: list[tuple[str, str]] = []

    def chat(self, model, messages, temperature=1.0, max_tokens=4096, **kw):
        user = messages[-1]["content"]
        self.calls.append((model, user))
        if model == "primary/model" and self.reject_when(user):
            raise ProviderRejectionError("input flagged for moderation")
        return ChatResult(f"<reply>\n{model} declines.\n</reply>", 10, 5, "stop")


def _ctx(tmp_path, client) -> Ctx:
    cfg = {
        "models": {
            "main": {"model": "primary/model", "max_tokens": 200},
            "spare": {"model": "spare/model", "max_tokens": 200},
        },
        "max_fail_pct": 100.0,
    }
    ctx = Ctx(cfg=cfg, usage=Usage(), workers=1, run_dir=tmp_path, smoke=False)
    ctx._client = client
    return ctx


STAGE = {
    "name": "draft_first_turn",
    "kind": "llm_tagged",
    "model": "main",
    "tags": ["reply"],
    "save": {"first_turn": "reply"},
    "also": {"supervise": "final"},
    "stamp_model": "first_turn_source",
    "fallback_model": "spare",
    "prompts": {"system": "sys", "user": "{user}"},
}


def test_provider_rejection_falls_back_and_the_record_says_who_answered(tmp_path):
    client = _Client(reject_when=lambda u: "distress" in u)
    ctx = _ctx(tmp_path, client)
    records = [
        {"scenario_id": "a", "user": "an ordinary ask"},
        {"scenario_id": "b", "user": "a person in distress"},
    ]
    out = op_llm_tagged(STAGE, ctx.cfg).fn(ctx, records, None)
    by = {r["scenario_id"]: r for r in out}
    assert by["a"]["first_turn_source"] == "primary/model"
    assert by["b"]["first_turn_source"] == "spare/model"
    assert by["b"]["first_turn"] == "spare/model declines."
    assert all(r["supervise"] == "final" for r in out)
    # The fallback is only reached after the primary rejected: two calls for b, one for a.
    assert [m for m, _ in client.calls].count("spare/model") == 1
    # Nothing failed, so the manifest's failure block for the stage is empty.
    assert ctx.manifest_extra["failures"]["draft_first_turn"] == {}


def test_without_a_fallback_the_rejection_is_a_counted_loss(tmp_path):
    client = _Client(reject_when=lambda u: "distress" in u)
    ctx = _ctx(tmp_path, client)
    stage = {k: v for k, v in STAGE.items() if k != "fallback_model"}
    records = [
        {"scenario_id": "a", "user": "an ordinary ask"},
        {"scenario_id": "b", "user": "a person in distress"},
    ]
    out = op_llm_tagged(stage, ctx.cfg).fn(ctx, records, None)
    assert [r["scenario_id"] for r in out] == ["a"]
    assert out[0]["first_turn_source"] == "primary/model"
    tally = ctx.manifest_extra["failures"]["draft_first_turn"]
    assert tally["n"] == 1 and tally["of"] == 2
    assert tally["by_type"] == {"ProviderRejectionError": 1}
    assert "moderation" in tally["examples"][0]


def test_resilient_tallies_failures_by_type_and_still_gates():
    def fn(i):
        if i % 3 == 0:
            raise ValueError("lint")
        if i == 1:
            raise KeyError("shape")
        return i

    tally: dict = {}
    ok = resilient(fn, 6, 1, "t", max_fail_pct=100.0, failures=tally)
    assert ok == [2, 4, 5]
    assert tally["by_type"] == {"ValueError": 2, "KeyError": 1}
    assert tally["n"] == 3 and tally["of"] == 6
    with pytest.raises(RuntimeError, match="above max_fail_pct"):
        resilient(fn, 6, 1, "t", max_fail_pct=10.0, failures={})


def test_run_items_forwards_the_tally():
    tally: dict = {}
    out = run_items(
        [{"x": 1}, {"x": 0}],
        lambda r: {"y": 1 / r["x"]},
        1,
        "t",
        max_fail_pct=100.0,
        failures=tally,
    )
    assert out == [{"y": 1.0}]
    assert tally["by_type"] == {"ZeroDivisionError": 1}
