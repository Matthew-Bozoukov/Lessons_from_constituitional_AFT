# ABOUTME: Optional batch mode: batched requests must be byte-identical to interactive
# ABOUTME: ones, results judged by one rule set, and every reject mopped up interactively.

"""What has to hold.

Batching is a billing optimisation, so — like the cache marker — the one thing it may
never do is change the data. The load-bearing tests are the parity ones: a batched
request body must be exactly what `chat` would have sent (same cache blocks, same
provider pin), and a batched completion must pass through the same typed-error
classification as an interactive one. Everything else guards the seams: resume must
not resubmit paid-for jobs, a dead job's records must fall to the interactive path
rather than vanish, and a stage too small (or unkeyed) to batch must quietly run the
old code path.
"""

from __future__ import annotations

import json

import pytest

from src.data.synth import stage_runtime
from src.data.synth.stage_runtime import (
    BATCH_MIN_ITEMS,
    Checkpoint,
    Usage,
    run_batch,
    run_items_batched,
)
from src.endpoints.openrouter import (
    CACHE_MARK,
    EmptyCompletionError,
    ProviderRejectionError,
    build_request_body,
    provider_pin,
    result_from_payload,
)

ANTHROPIC = "anthropic/claude-sonnet-5"
PINNED_OTHER = "qwen/qwen3-32b"


# --- request parity -------------------------------------------------------------------


def test_body_applies_cache_blocks_and_pin_for_anthropic():
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": f"prefix{CACHE_MARK}suffix"}]
    body = build_request_body(ANTHROPIC, msgs, 0.7, 6144)
    assert body["provider"] == provider_pin(ANTHROPIC)
    assert body["temperature"] == 0.7 and body["max_tokens"] == 6144
    blocks = body["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "prefix",
                        "cache_control": {"type": "ephemeral"}}
    assert blocks[1] == {"type": "text", "text": "suffix"}


def test_body_strips_marker_for_non_anthropic_and_merges_extra():
    msgs = [{"role": "user", "content": f"prefix{CACHE_MARK}suffix"}]
    body = build_request_body(PINNED_OTHER, msgs, 1.0, 100,
                              extra_body={"reasoning": {"enabled": False}})
    assert body["messages"][0]["content"] == "prefixsuffix"
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"] == provider_pin(PINNED_OTHER)


def test_body_never_includes_model():
    """The batch API takes the model once per job; a per-request model would let the
    two drift apart."""
    body = build_request_body(ANTHROPIC, [{"role": "user", "content": "x"}], 1.0, 10)
    assert "model" not in body


# --- result parity --------------------------------------------------------------------


def _payload(content="ok", finish="stop", cached=0):
    return {"provider": "anthropic",
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def test_payload_roundtrips_usage_and_cache_tokens():
    res = result_from_payload(ANTHROPIC, _payload(cached=90))
    assert (res.content, res.prompt_tokens, res.completion_tokens,
            res.cached_tokens, res.provider) == ("ok", 100, 20, 90, "anthropic")


def test_payload_content_filter_is_a_rejection():
    with pytest.raises(ProviderRejectionError):
        result_from_payload(ANTHROPIC, _payload(finish="content_filter"))


def test_payload_no_choices_classifies_by_error_code():
    with pytest.raises(ProviderRejectionError):
        result_from_payload(ANTHROPIC, {"error": {"code": 403, "message": "blocked"}})
    with pytest.raises(EmptyCompletionError):
        result_from_payload(ANTHROPIC, {"error": {"code": 500, "message": "oops"}})
    with pytest.raises(EmptyCompletionError):
        result_from_payload(ANTHROPIC, {})


def test_payload_empty_content_is_retryable_blank():
    with pytest.raises(EmptyCompletionError):
        result_from_payload(ANTHROPIC, _payload(content=""))


# --- run_batch transport (stubbed HTTP) -----------------------------------------------


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def test_run_batch_submits_polls_collects_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    posts, gets = [], []

    def fake_post(url, headers=None, json=None):
        posts.append(json)
        return _Resp({"id": f"b{len(posts)}"})

    def fake_get(url, headers=None):
        gets.append(url)
        bid = url.rsplit("/", 1)[1]
        ids = next(p for i, p in enumerate(posts) if f"b{i + 1}" == bid)
        return _Resp({"status": "completed", "results": [
            {"custom_id": r["custom_id"],
             "response": {"status_code": 200,
                          "body": _payload(content=f"c-{r['custom_id']}")}}
            for r in ids["requests"]]})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    got = {}
    state = tmp_path / "state.json"
    run_batch(ANTHROPIC, {"a": {"m": 1}, "b": {"m": 2}}, "stg", state,
              lambda cid, body: got.__setitem__(cid, body), chunk=1, poll_s=0)

    assert len(posts) == 2  # chunk=1 -> one job per request
    assert posts[0]["endpoint"] == "/v1/chat/completions"
    assert posts[0]["model"] == ANTHROPIC
    assert {r["custom_id"] for p in posts for r in p["requests"]} == {"a", "b"}
    assert got["a"]["choices"][0]["message"]["content"] == "c-a"
    assert not state.exists()  # terminal -> state removed


def test_run_batch_resumes_without_resubmitting(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"batches": [{"batch_id": "old1", "ids": ["a"]}]}))
    posts = []

    def fake_post(url, headers=None, json=None):
        posts.append(json)
        return _Resp({"id": "new1"})

    def fake_get(url, headers=None):
        return _Resp({"status": "completed", "results": [
            {"custom_id": cid,
             "response": {"status_code": 200, "body": _payload(content=cid)}}
            for cid in ("a", "b")]})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    got = {}
    run_batch(ANTHROPIC, {"a": {}, "b": {}}, "stg", state,
              lambda cid, body: got.__setitem__(cid, body), poll_s=0)
    # "a" was already submitted as job old1; only "b" goes out again.
    assert [r["custom_id"] for p in posts for r in p["requests"]] == ["b"]
    assert set(got) == {"a", "b"}


def test_run_batch_dead_job_leaves_requests_uncollected(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("requests.post",
                        lambda url, headers=None, json=None: _Resp({"id": "b1"}))
    monkeypatch.setattr("requests.get",
                        lambda url, headers=None: _Resp({"status": "expired"}))
    got = {}
    run_batch(ANTHROPIC, {"a": {}}, "stg", tmp_path / "s.json",
              lambda cid, body: got.__setitem__(cid, body), poll_s=0)
    assert got == {}  # no exception: the caller's interactive mop-up owns "a"


# --- run_items_batched orchestration (stubbed run_batch) ------------------------------


def _items(n):
    return [{"scenario_id": f"s{i}", "x": i} for i in range(n)]


def test_batched_stage_warms_batches_and_mops_up(tmp_path, monkeypatch):
    items = _items(BATCH_MIN_ITEMS + 2)
    interactive: list[str] = []

    def one(r):
        interactive.append(r["scenario_id"])
        return {**r, "out": "interactive"}

    def fake_run_batch(model, requests, stage, state_path, collect, **kw):
        # Deliver every batched request except s3 (simulating a reject).
        for cid in requests:
            if cid != "s3":
                collect(cid, _payload(content=f"batched-{cid}"))

    monkeypatch.setattr(stage_runtime, "run_batch", fake_run_batch)
    ckpt = Checkpoint(tmp_path / "p.jsonl", key="scenario_id")
    usage = Usage()
    out = run_items_batched(
        items, one, lambda r: {"body": r["scenario_id"]},
        lambda r, res: {**r, "out": res.content},
        usage=usage, model=ANTHROPIC, stage="mk", key="scenario_id",
        run_dir=tmp_path, workers=2, desc="stg", ckpt=ckpt)

    assert [r["scenario_id"] for r in out] == [r["scenario_id"] for r in items]
    by_id = {r["scenario_id"]: r["out"] for r in out}
    assert by_id["s0"] == "interactive"          # the warming call
    assert by_id["s1"] == "batched-s1"
    assert by_id["s3"] == "interactive"          # the reject, mopped up
    assert set(interactive) == {"s0", "s3"}      # batched successes never re-called
    # Batched completions are tallied at the batch API's 50% rate.
    stg = usage.by_stage["mk"]
    assert stg["calls"] == BATCH_MIN_ITEMS  # all but warm (interactive) and s3 (reject)
    assert stg["usd"] == pytest.approx(
        BATCH_MIN_ITEMS * 0.5 * (100 / 1e6 * 2.00 + 20 / 1e6 * 10.00))


def test_small_or_unkeyed_stages_fall_back_to_interactive(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("run_batch must not be called")

    monkeypatch.setattr(stage_runtime, "run_batch", boom)
    calls = []

    def one(r):
        calls.append(r["scenario_id"])
        return {**r, "out": "i"}

    small = _items(BATCH_MIN_ITEMS - 1)
    out = run_items_batched(small, one, None, None, usage=Usage(), model=ANTHROPIC,
                            stage="mk", key="scenario_id", run_dir=tmp_path,
                            workers=2, desc="stg", ckpt=None)
    assert len(out) == len(small) and len(calls) == len(small)

    dup = _items(BATCH_MIN_ITEMS + 2)
    for r in dup:
        r["scenario_id"] = "same"
    calls.clear()
    out = run_items_batched(dup, one, None, None, usage=Usage(), model=ANTHROPIC,
                            stage="mk", key="scenario_id", run_dir=tmp_path,
                            workers=2, desc="stg", ckpt=None)
    assert len(calls) == len(dup)  # duplicate keys -> interactive, nothing dropped
