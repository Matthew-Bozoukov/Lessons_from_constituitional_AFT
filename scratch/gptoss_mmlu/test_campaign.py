# ABOUTME: Offline checks for aggregate/stage admission and conservative unknown reservations.
# ABOUTME: Run using the CPU environment pytest; no network or paid API calls.
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from scratch.gptoss_mmlu import bridge

def test_shared_atomic_ceiling_and_retained_unknown(tmp_path,monkeypatch):
    monkeypatch.setenv('MMLU_PHASE','load')
    path=tmp_path/'budget.json'
    budgets=[bridge.CampaignBudget(path,.06,'model',c,['a','b','c'],bridge.RATES) for c in ['a','b','c']]
    barrier=threading.Barrier(3)
    # Three reservations exactly fit; a later settle reduces accounted usage only.
    def call(i):
        barrier.wait()
        return budgets[i].reserve(0,20000)
    with ThreadPoolExecutor(3) as pool: keys=list(pool.map(call,range(3)))
    data=json.loads(path.read_text())
    assert len(data['requests'])==3
    assert sum(r['upper_usd'] for r in data['requests'].values())<=.06
    budgets[0].settle(keys[0],1)
    data=json.loads(path.read_text())
    assert sum(r['state']=='reserved' for r in data['requests'].values())==2
    assert data['requests'][keys[1]]['upper_usd']==.0168

def test_smoke_cap_is_cumulative_across_arms(tmp_path,monkeypatch):
    monkeypatch.setenv('MMLU_PHASE','smoke')
    monkeypatch.setattr(bridge,'ROOT',tmp_path)
    monkeypatch.setattr(bridge,'resources_ok',lambda:True)
    path=tmp_path/'budget.json'
    budgets=[bridge.CampaignBudget(path,15,'model',c,['a','b','c'],bridge.RATES) for c in ['a','b','c']]
    for i in range(17):
        b=budgets[i%3]; k=b.reserve(1000,32768); b.settle(k,32768)
    with pytest.raises(RuntimeError,match='smoke ceiling'): budgets[2].reserve(1000,32768)
    data=json.loads(path.read_text())
    assert len(data['requests'])==17 and sum(r['upper_usd'] for r in data['requests'].values())<.49
    monkeypatch.setenv('MMLU_PHASE','run')
    key=budgets[0].reserve(1000,32768)
    data=json.loads(path.read_text())
    assert len(data['requests'])==18 and data['requests'][key]['phase']=='run'

def test_hold_prevents_new_reservations(tmp_path,monkeypatch):
    monkeypatch.setenv('MMLU_PHASE','run'); monkeypatch.setattr(bridge,'ROOT',tmp_path)
    budget=bridge.CampaignBudget(tmp_path/'budget.json',15,'model','a',['a'],bridge.RATES)
    (tmp_path/'admission-hold.json').write_text('{}')
    with pytest.raises(RuntimeError,match='held'): budget.reserve(10,20)
    assert json.loads((tmp_path/'budget.json').read_text())['requests']=={}
