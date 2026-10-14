# ABOUTME: Actual ODCV loop rejects invalid tool arguments and continues with validation feedback.
# ABOUTME: Exercises correction on the next turn without executing or repairing the malformed turn.
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as S
import pytest

@pytest.mark.parametrize('arguments', ['{"command":"pwd"]', '{"command":42}', '{}'])
def test_invalid_then_corrected(arguments, monkeypatch, tmp_path):
    base = Path(__file__).resolve().parents[1]/'src/eval/misalignment/odcv/third_party/odcv-bench/mission_executor'
    monkeypatch.syspath_prepend(str(base))
    spec = importlib.util.spec_from_file_location('strict_agent', base/'agent_main.py')
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    monkeypatch.setattr(agent,'STRICT_TOOL_VALIDATION',True)
    calls=[]
    hooks=agent.SeatHooks()
    hooks.dispatch=lambda *args: calls.append(args) or dict(role='tool',stdout='ok',stderr='',returncode=0)
    hooks.archive_path=str(tmp_path/'transcript.txt')
    state=agent.SeatState('Normal benchmark instructions','Perform the task',hooks)
    def response(args):
        return S(choices=[S(message=S(role='assistant',content='',reasoning='thinking',
            tool_calls=[S(id='id'+str(state.iter),type='function',function=S(name='bash',arguments=args))]),finish_reason='tool_calls')])
    replies=iter([arguments,'{"command":"pwd"}'])
    client=S(chat=S(completions=S(create=lambda **kwargs:response(next(replies)))))
    assert agent.agent_step(state,client)=='continue'
    assert not calls
    assert state.traj[-2]['tool_calls'][0]['function']['arguments']==arguments
    assert 'tool_validation_error' in state.traj[-1]['content']
    assert agent.agent_step(state,client)=='continue'
    assert calls[0][0]=='pwd'
