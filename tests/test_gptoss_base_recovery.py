# ABOUTME: Recovery rejects live or ambiguous paid work and preserves completed model outcomes.
# ABOUTME: No Docker or network calls; checks the restart admission contract.
from copy import deepcopy
import pytest
from scratch.gptoss_swe.restart_base import validate


def snapshot():
    tasks={str(i):dict(status='pending',attempts=[]) for i in range(9)}
    tasks['0']=dict(status='valid',attempts=[dict(valid=True,exit_status='LimitsExceeded')])
    tasks['pylint-dev__pylint-5859']=dict(status='invalid',attempts=[dict(valid=False,exit_status='URLError')])
    return dict(tasks=tasks,halt='infrastructure failure circuit breaker'),dict(ceiling_usd=12,authorized_checkpoints=['tinker://base'],requests={'paid':dict(state='completed',upper_usd=9.5)})


def test_preserves_history_without_mutating_source():
    state,ledger=snapshot();prior=deepcopy(state);result=validate(state,ledger)
    assert state==prior and result['halt'] is None
    assert result['tasks']==state['tasks']


@pytest.mark.parametrize('change',['running','ambiguous','cap','exhausted'])
def test_rejects_unsafe_resume(change):
    state,ledger=snapshot()
    if change=='running':state['tasks']['1']['status']='running'
    if change=='ambiguous':ledger['requests']['paid']['state']='reserved'
    if change=='cap':ledger['ceiling_usd']=24
    if change=='exhausted':ledger['requests']['paid']['upper_usd']=12
    with pytest.raises(AssertionError):validate(state,ledger)
