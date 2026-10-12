# ABOUTME: Guard backend parity and reject silent tokenization, truncation and stop changes.
# ABOUTME: Exercises raw completion responses without paid generation or GPU rental.
from types import SimpleNamespace
import pytest
from scratch.gptoss_swe.mxfp4_bridge import request_body, sample_result

def fixture():
    body=request_body([200006,42],SimpleNamespace(stop=[200002,200012],max_tokens=9,temperature=.7,top_p=.95,top_k=20,seed=None))
    response=dict(choices=[dict(token_ids=[80,200012],prompt_token_ids=body['prompt'],finish_reason='stop',stop_reason=200012)],
        usage=dict(prompt_tokens=2,completion_tokens=2,prompt_tokens_details=dict(cached_tokens=1)))
    return body,response

def test_exact_tokens_and_controls():
    body,response=fixture()
    result=sample_result(body,response)
    assert result.sequences[0].tokens==[80,200012]
    assert result.prompt_cache_hit_tokens==1
    assert body['top_k']==20 and body['temperature']==.7 and body['top_p']==.95
    assert body['ignore_eos'] and not body['add_special_tokens']

@pytest.mark.parametrize('mutation', ['prompt','missing_ids','missing_stop','extra_tokens','accounting','unexpected_end'])
def test_refuse_transport_changes(mutation):
    body,response=fixture();c=response['choices'][0]
    if mutation=='prompt':c['prompt_token_ids']=[0]+body['prompt']
    if mutation=='missing_ids':c.pop('token_ids')
    if mutation=='missing_stop':c['token_ids']=[80,81]
    if mutation=='extra_tokens':body['max_tokens']=1
    if mutation=='accounting':response['usage']['completion_tokens']=3
    if mutation=='unexpected_end':c['finish_reason']='abort'
    with pytest.raises(ValueError):sample_result(body,response)

def test_length_is_not_repaired():
    body,response=fixture();response['choices'][0].update(token_ids=[80,81],finish_reason='length')
    assert sample_result(body,response).sequences[0].stop_reason=='length'

def test_disabled_topk_and_seed():
    b=request_body([1],SimpleNamespace(stop=[200002],max_tokens=1,temperature=1,top_p=1,top_k=-1,seed=7))
    assert b['top_k']==-1 and b['seed']==7
