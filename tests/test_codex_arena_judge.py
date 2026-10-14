# ABOUTME: Codex Arena judge parsing and privacy checks, with no model calls.
# ABOUTME: Verify last-verdict semantics and subscription-only command construction.
from scratch.gptoss_arena_judge import campaign as c


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
