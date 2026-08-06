# ABOUTME: Offline unit tests for the spec-alignment filter: verdict parsing, sample
# ABOUTME: serialization, the keep-on-doubt rules, and checkpoint resume — no network.

import json

from src.data.mixture.spec_filter import (
    Checkpoint,
    parse_verdict,
    run_filter,
    serialize_sample,
)


class _FakeResult:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    """Scripted judge: replies by sample text, recording every prompt it sees."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def chat(self, model, messages, **kw):
        sample = messages[1]["content"]
        self.calls.append(sample)
        for key, reply in self.replies.items():
            if key in sample:
                return _FakeResult(reply)
        return _FakeResult('{"verdict": "keep", "category": "none", "why": "fine"}')


def _reject(category="toxic"):
    return json.dumps({"verdict": "reject", "category": category, "why": "bad"})


def _rows(n, source="s"):
    return [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"answer-{i}"}],
             "source": source} for i in range(n)]


def test_parse_verdict_keeps_on_unparseable_output():
    ok = parse_verdict('noise {"verdict": "reject", "category": "toxic", "why": "w"} tail')
    assert ok == {"verdict": "reject", "category": "toxic", "why": "w", "parsed": True}
    bad = parse_verdict("I cannot judge this")
    assert bad["verdict"] == "keep" and bad["category"] == "unparsed" and not bad["parsed"]


def test_serialize_sample_labels_roles_reasoning_and_tool_calls():
    calls = [{"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]
    out = serialize_sample({"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r",
         "tool_calls": calls}]}, limit=1000)
    assert "[user]: q" in out and "[assistant reasoning]: r" in out
    assert '"name": "f"' in out
    # Legacy text rows pass through; long samples are middle-elided, keeping both ends.
    assert serialize_sample({"text": "rendered"}, limit=100) == "rendered"
    long = serialize_sample({"text": "S" + "x" * 500 + "E"}, limit=90)
    assert len(long) <= 120 and long.startswith("S") and long.endswith("E")
    assert "[elided]" in long


def test_run_filter_rejects_only_parsed_rejects(tmp_path):
    client = _FakeClient({"answer-1": _reject("toxic"),
                          "answer-3": "gibberish, no json"})
    keep, report = run_filter(_rows(5), constitution_text="const", model="m",
                              dest=tmp_path, workers=2, client=client)
    assert keep == [True, False, True, True, True]
    assert report["samples_kept"] == 4 and report["rejected_by_category"] == {"toxic": 1}
    assert report["unparsed_judge_replies"] == 1, "gibberish must be kept AND counted"
    assert report["by_source"]["s"] == {"total": 5, "kept": 4, "reject_pct": 20.0}
    # The system prompt (constitution + rubric) is byte-identical across calls — that is
    # what makes the provider's prompt cache apply.
    verdicts = [json.loads(line) for line in (tmp_path / "verdicts.jsonl").open()]
    assert len(verdicts) == 5 and {v["idx"] for v in verdicts} == set(range(5))


def test_run_filter_resumes_from_checkpoint(tmp_path):
    rows = _rows(4)
    (tmp_path / "verdicts.jsonl").write_text(
        json.dumps({"idx": 0, "source": "s", "verdict": "reject", "category": "toxic",
                    "why": "", "parsed": True}) + "\n"
        + json.dumps({"idx": 1, "source": "s", "verdict": "keep", "category": "none",
                      "why": "", "parsed": True}) + "\n"
        + "{torn line")
    client = _FakeClient({})
    keep, report = run_filter(rows, constitution_text="c", model="m", dest=tmp_path,
                              client=client)
    # Only the unjudged rows (2, 3) are paid for again; the torn line re-runs nothing
    # already recorded.
    assert len(client.calls) == 2
    assert keep == [False, True, True, True]
    assert report["samples_rejected"] == 1


def test_run_filter_limit_marks_partial_pass(tmp_path):
    client = _FakeClient({})
    keep, report = run_filter(_rows(6), constitution_text="c", model="m", dest=tmp_path,
                              limit=2, client=client)
    assert len(client.calls) == 2 and all(keep)
    assert report["unjudged_kept"] == 4, "a smoke pass must never look like a full filter"


def test_checkpoint_survives_torn_and_blank_lines(tmp_path):
    p = tmp_path / "verdicts.jsonl"
    p.write_text('{"idx": 3, "verdict": "keep"}\n\n{"idx": 4, "verdict"\n')
    ckpt = Checkpoint(p)
    assert set(ckpt.done) == {3}
    ckpt.record({"idx": 4, "verdict": "keep"})
    assert set(Checkpoint(p).done) == {3, 4}
