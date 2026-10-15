# ABOUTME: Offline checks for budget admission, crash recovery, duplicate ownership and MASK workload.
# ABOUTME: All tests use local temporary files and fake outputs; no inference or judging calls.
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys

import pytest
import campaign as c


def test_budget_includes_inflight_and_survives_restart(tmp_path):
    p=tmp_path/'ledger.sqlite'; l=c.Ledger(p,{'id':1})
    assert l.reserve('judge','1',{'a':1},0.7,1)
    with pytest.raises(c.Hold): l.reserve('judge','2',{},0.4,1)
    l.db.close();l=c.Ledger(p,{'id':1})
    assert not l.reserve('judge','1',{'a':1},0.7,1)
    with pytest.raises(AssertionError): l.reserve('judge','1',{'a':2},0.7,1)
    l.finish('judge','1','completed',0.2)
    assert l.reserve('judge','2',{},0.4,1)
    assert l.summary()['judge']['reserved_usd']==0.4


def test_atomic_concurrent_budget(tmp_path):
    p=tmp_path/'ledger.sqlite'; c.Ledger(p,{}).db.close()
    def task(i):
        l=c.Ledger(p,{})
        try: return l.reserve('judge',str(i),{},0.31,1)
        except c.Hold: return False
        finally: l.db.close()
    with ThreadPoolExecutor(max_workers=8) as pool: assert sum(pool.map(task,range(20)))==3


def test_frozen_identity_rejects_change(tmp_path):
    p=tmp_path/'ledger.sqlite';c.Ledger(p,{'cap':12}).db.close()
    with pytest.raises(AssertionError):c.Ledger(p,{'cap':15})


def test_one_owner(tmp_path):
    with c.owner(tmp_path):
        with pytest.raises(OSError):
            with c.owner(tmp_path):pass


def test_generation_jobs_match_protocol(tmp_path):
    cfg=c.load_config();c.freeze(cfg,tmp_path)
    frames,jobs=c.generation_jobs(tmp_path,cfg)
    assert len(frames)==6 and len(jobs)==4438
    assert len({j['id'] for j in jobs})==4438
    assert len({(j['category'],j['row']) for j in jobs if j['pilot']})==12
    # Independent belief replicates must never be merged merely because prompts match.
    examples=[j for j in jobs if j['category']=='statistics' and j['row']==0 and 'belief' in j['column']]
    assert len(examples)==3 and len({j['id'] for j in examples})==3
    assert len({c.digest(j['messages']) for j in examples})==1


def test_judge_count_and_no_private_reasoning(tmp_path):
    cfg=c.load_config();c.freeze(cfg,tmp_path);frames,jobs=c.generation_jobs(tmp_path,cfg)
    for j in jobs:
        df=frames[j['category']]
        df.at[j['row'],j['column']]='A short visible answer.'
        df.at[j['row'],j['column'].replace('generation(','reasoning(')]='SECRET_REASONING_MARKER'
    runner=c.Campaign(cfg,tmp_path,c.Ledger(tmp_path/'ledger.sqlite',{}))
    asyncio.run(runner.evaluator(frames))
    assert len(runner.judge_jobs)==4246
    assert len({j['id'] for j in runner.judge_jobs})==4246
    assert 'SECRET_REASONING_MARKER' not in json.dumps(runner.judge_jobs)


@pytest.mark.parametrize('kind',['invalid','timeout','missing_cost','valid'])
def test_judge_no_ambiguous_retry_and_accounting(tmp_path,monkeypatch,kind):
    import httpx
    monkeypatch.setenv('OPENROUTER_API_KEY','offline-test')
    attempts=[]
    class Client:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,*args,**kw):
            attempts.append(kw)
            if kind=='timeout':raise TimeoutError()
            usage={'prompt_tokens':100,'completion_tokens':20}
            if kind!='missing_cost':usage['cost']=0.001
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'Answer: B' if kind!='invalid' else 'unparseable'}}],'usage':usage})
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    cfg=c.load_config(); runner=c.Campaign(cfg,tmp_path,c.Ledger(tmp_path/'ledger.sqlite',{}))
    jobs=[dict(id='one',numeric=False,body={'messages':[{'role':'user','content':'test'}],'max_completion_tokens':500})]
    asyncio.run(runner.judge(jobs));asyncio.run(runner.judge(jobs))
    assert len(attempts)==1
    rec=json.loads(runner.path('judge','one').read_text())
    assert rec['valid']==(kind in ('valid','missing_cost'))
    if kind=='missing_cost':assert runner.halt and runner.ledger.summary()['judge']['reserved_usd']>0
    if kind=='timeout':assert runner.ledger.summary()['judge']['reserved_usd']>0



