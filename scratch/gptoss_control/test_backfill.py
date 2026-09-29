# ABOUTME: Offline checks for reference-hidden generation and conservative CoT acceptance.
# ABOUTME: Run: uv run --project src/infra/endpoints/tinker_runtime python -m pytest scratch/gptoss_control/test_backfill.py -q
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from run import answer_payload, agreement_prompt, judge_agreement, select_backfill_targets


def test_tool_payload_preserves_arguments_ignores_ids():
    a = {'content':'', 'tool_calls':[{'id':'arbitrary','function':{'name':'f','arguments':'{"x": [1, 2]}'}}]}
    assert answer_payload(a) == {'content':'','tool_calls':[{'name':'f','arguments':{'x':[1,2]}}]}


def test_judge_sees_both_answers_and_trace():
    text=agreement_prompt([{'role':'user','content':'question'}], '[]',
                          {'content':'generated'}, {'content':'original'}, 'reasoning')
    assert all(x in text for x in ['generated_answer', 'original_answer', 'generated_reasoning'])
    assert 'original' in text and 'generated' in text


class Judge:
    def __init__(self, content, finish='stop'):
        self.content, self.finish = content, finish
    def chat(self, **kwargs):
        return SimpleNamespace(content=self.content,finish_reason=self.finish,cost=.01,
                               provider='mock',response_id='id',prompt_tokens=10,completion_tokens=5)


def test_both_checks_required_and_truncation_rejected():
    for content, finish, verdict in [
        ('{"answer_agreement":true,"trace_compatible":true}', 'stop','yes'),
        ('{"answer_agreement":true,"trace_compatible":false}', 'stop','no'),
        ('{"answer_agreement":false,"trace_compatible":true}', 'stop','no'),
        ('{"answer_agreement":"true","trace_compatible":true}', 'stop','no'),
        ('{"answer_agreement":true,"trace_compatible":true}', 'length','no'),
        ('YES', 'stop','no')]:
        assert judge_agreement(Judge(content,finish), 'mock','prompt',512)['verdict'] == verdict


def test_pilot_contains_all_tool_shapes():
    rows=[{'source':'tool','messages':[{'tool_calls':[{}]*n}]} for n in [0,1,2]]
    targets=[{'row':i,'turn':0} for i in range(3)]
    assert {x['row'] for x in select_backfill_targets(targets,rows,3)} == {0,1,2}
