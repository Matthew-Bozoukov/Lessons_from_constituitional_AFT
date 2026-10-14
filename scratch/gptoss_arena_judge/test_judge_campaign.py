# ABOUTME: Codex Arena judge parsing and privacy checks, with no model calls.
# ABOUTME: Verify last-verdict semantics and subscription-only command construction.
from scratch.gptoss_arena_judge import campaign as c
import pytest


def test_final_verdict_is_last_and_strict():
    assert c.verdict('Consider [[A>B]], but final [[B>>A]].') == 'B>>A'
    assert c.verdict('No verdict') is None
    assert c.verdict('[[A=B]]') == 'A=B'
    assert c.verdict('[[A=B]] then [[AB]]') is None


def test_command_is_ephemeral_subscription_and_requested_judge():
    args=c.command(dict(id='test',category='hard_prompt'))
    assert args[args.index('--model')+1]=='gpt-6-luna'
    assert '--ephemeral' in args and '--ignore-user-config' in args
    assert 'forced_login_method="chatgpt"' in args
    assert 'model_reasoning_effort="low"' in args
    assert 'features.shell_tool=false' in args
    assert 'project_doc_max_bytes=0' in args
    assert not any('OPENROUTER' in a for a in args)


def test_transport_notice_requires_completed_turn_and_rejects_tools():
    notice={'type':'item.completed','item':{'type':'error','message':'Falling back from WebSockets to HTTPS transport. unexpected status 403 Forbidden: url: wss://chatgpt.com/backend-api/codex/responses'}}
    done={'type':'turn.completed','usage':{}}
    assert c.validate_events([notice,done])==[done]
    with pytest.raises(AssertionError): c.validate_events([notice])
    with pytest.raises(AssertionError): c.validate_events([{'type':'item.completed','item':{'type':'tool_call'}},done])


def test_order_orientation():
    from src.eval.capabilities.arena_hard.arena_hard_stats import battles_from_judgments
    rec={'uid':'x','category':'hard_prompt','model':'target','games':[{'score':'B>A'},{'score':'A>B'}]}
    assert [b['score'] for b in battles_from_judgments([rec])]==[1,1]


def test_only_explicit_rejection_without_output_can_retry(tmp_path):
    import json
    failed={'type':'turn.failed','error':{'message':'exceeded retry limit, last status: 429 Too Many Requests'}}
    path=tmp_path/'events.jsonl'
    path.write_text(json.dumps(failed)+'\n',encoding='utf-8')
    assert c.rejected_without_output(tmp_path)
    (tmp_path/'final.txt').write_text('some answer',encoding='utf-8')
    assert not c.rejected_without_output(tmp_path)
    (tmp_path/'final.txt').write_text('',encoding='utf-8')
    path.write_text(json.dumps(failed)+'\n'+json.dumps({'type':'turn.completed','usage':{}})+'\n',encoding='utf-8')
    assert not c.rejected_without_output(tmp_path)
    path.write_text(json.dumps({'type':'turn.failed','error':{'message':'timeout'}})+'\n',encoding='utf-8')
    assert not c.rejected_without_output(tmp_path)


def test_item_local_failure_does_not_hide_global_errors(tmp_path):
    from scratch.gptoss_arena_judge.campaign import item_local_failure
    assert item_local_failure(tmp_path, RuntimeError('Judge timeout; retained start and output, no retry'))
    assert not item_local_failure(tmp_path, AssertionError('Unexpected tool/error item in judge output'))
    assert not item_local_failure(tmp_path, RuntimeError('Rate rejection attempt limit exhausted'))
    (tmp_path/'stderr.txt').write_text('input_too_large')
    assert item_local_failure(tmp_path, AssertionError('CLI exit 1'))


def test_exclusion_requires_whole_matched_question(tmp_path, monkeypatch):
    import pytest
    from scratch.gptoss_arena_judge import campaign as c
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    jobs=[dict(id=str(i),uid='q') for i in range(6)]
    c.save(tmp_path/'jobs.json',jobs)
    c.save(tmp_path/'exclusions.json',dict(jobs=jobs[:1]))
    with pytest.raises(AssertionError): c.excluded_ids()
    c.save(tmp_path/'exclusions.json',dict(jobs=jobs))
    assert c.excluded_ids()=={str(i) for i in range(6)}
