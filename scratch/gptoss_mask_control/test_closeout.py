# ABOUTME: Validate retry isolation and shared budget accounting without paid requests.
# ABOUTME: Ensure restart cannot issue a second authorized retry or alter original results.
import json
import pytest
from campaign import Ledger, Hold, save
from closeout import Closeout, RetryLedger
from campaign import digest
from score_local import resolve

def test_retry_budget_and_no_repeat(tmp_path):
    ledger=Ledger(tmp_path/'ledger.sqlite',{})
    assert ledger.reserve('generation','abc',{},0.6,1.0)
    ledger.finish('generation','abc','review',0.4)
    retry=RetryLedger(ledger,{'generation':{'abc'},'judge':set()})
    assert retry.reserve('generation','abc',{},0.6,1.0)
    assert not retry.reserve('generation','abc',{},0.6,1.0)
    with pytest.raises(Hold): retry.reserve('generation','def',{},0.1,1.0)
    assert ledger.summary()['generation']['settled_usd']==0.4
    assert ledger.summary()['generation']['reserved_usd']==0.6

def test_paths_and_judge_attempt_ids(tmp_path):
    ledger=Ledger(tmp_path/'ledger.sqlite',{})
    c=Closeout({},tmp_path,ledger,{'generation':{'abc'},'judge':{'xyz'}})
    original=tmp_path/'calls/generation/abc/result.json'
    save(original,{'valid':False})
    save(c.path('generation','abc'),{'valid':True})
    assert json.loads(original.read_text())=={'valid':False}
    assert c.path('generation','new')==tmp_path/'calls/generation/new/result.json'
    assert c.ledger.key('judge','xyz-1')=='xyz-1:authorized-retry-1'
    assert c.ledger.key('judge','new-1')=='new-1'

def test_request_resolution_survives_column_reordering(tmp_path):
    jobs=[dict(id='a',category='known',row=0,body={'prompt':'first'}),dict(id='b',category='known',row=0,body={'prompt':'second'})]
    save(tmp_path/'pre-retry-status/judge-jobs.json',jobs)
    for j in jobs:
        save(tmp_path/'calls/judge'/j['id']/'request.json',j['body'])
        save(tmp_path/'calls/judge'/j['id']/'result.json',dict(valid=True,request_sha256=digest(j['body'])))
    reordered=[dict(jobs[0],body=jobs[1]['body']),dict(jobs[1],body=jobs[0]['body'])]
    mapping=resolve(tmp_path,{},dict(generation=set(),judge=set()),reordered)
    assert mapping=={'a':'calls/judge/b','b':'calls/judge/a'}
    with pytest.raises(AssertionError):
        resolve(tmp_path,{},dict(generation=set(),judge=set()),[dict(reordered[0],body={'prompt':'unknown'}),reordered[1]])
