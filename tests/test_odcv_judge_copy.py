# ABOUTME: The judge reads a capped COPY of a transcript whose single line exceeds the cap
# ABOUTME: (a whole-file `cat`), and the original file otherwise; the rollout is never edited.

from __future__ import annotations

from src.eval.misalignment.odcv.odcv_judge import JUDGE_LINE_CAP, judge_copy


def test_judge_budget_reserves_before_dispatch_and_persists_uncertain_call(tmp_path):
    import json
    import pytest
    from src.eval.misalignment.odcv.odcv_judge import JudgeBudget
    model = 'google/gemini-3-flash-preview'
    path = tmp_path/'ledger.json'
    budget = JudgeBudget(path, .03, 8192)
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        raise TimeoutError('uncertain')
    with pytest.raises(TimeoutError):
        budget.create(send, model=model, messages=[dict(role='user', content='Judge this')])
    assert calls[0]['max_tokens'] == 8192
    assert json.loads(path.read_text())[0]['status'] == 'reserved'
    budget = JudgeBudget(path, .03, 8192)
    with pytest.raises(RuntimeError, match='cap reached'):
        budget.create(send, model=model, messages=[dict(role='user', content='Judge this')])
    assert len(calls) == 1


def test_judge_budget_settles_tokens_but_refuses_truncated_verdict(tmp_path):
    import json
    import pytest
    from types import SimpleNamespace
    from src.eval.misalignment.odcv.odcv_judge import JudgeBudget
    path = tmp_path/'ledger.json'
    budget = JudgeBudget(path, 1, 8192)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                               choices=[SimpleNamespace(finish_reason='length')])
    with pytest.raises(RuntimeError, match='truncated'):
        budget.create(lambda **kw: response, model='google/gemini-3-flash-preview',
                      messages=[dict(role='user', content='Judge this')])
    entry = json.loads(path.read_text())[0]
    assert entry['status'] == 'settled'
    assert entry['charged_or_reserved_usd'] == pytest.approx(.000065)


def test_short_transcript_is_handed_over_unchanged(tmp_path):
    rec = tmp_path / "S" / "rollout_000" / "messages_record.txt"
    rec.parent.mkdir(parents=True)
    rec.write_text("== Step 1 ==\nrole: user\ncontent: hi\n")
    path, cut = judge_copy(rec, tmp_path / "copies")
    assert path == rec and cut == 0 and not (tmp_path / "copies").exists()


def test_oversized_tool_line_is_cut_only_in_the_judges_copy(tmp_path):
    rec = tmp_path / "S" / "rollout_000" / "messages_record.txt"
    rec.parent.mkdir(parents=True)
    big = "content: " + "x" * (JUDGE_LINE_CAP * 3)
    original = f"== Step 1 ==\nrole: tool\n{big}\n== Step 2 ==\nrole: assistant\ncontent: done\n"
    rec.write_text(original)
    path, cut = judge_copy(rec, tmp_path / "copies")
    assert cut == 1 and path != rec and path.name == "S__rollout_000__judge_copy.txt"
    copy = path.read_text()
    assert "chars of tool output cut for the judge" in copy
    assert len(max(copy.split("\n"), key=len)) < JUDGE_LINE_CAP + 200
    assert copy.endswith("role: assistant\ncontent: done\n"), "everything else is intact"
    assert rec.read_text() == original, "the rollout on disk is never edited"
