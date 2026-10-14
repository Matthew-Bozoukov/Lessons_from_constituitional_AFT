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
