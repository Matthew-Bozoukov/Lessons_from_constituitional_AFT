# ABOUTME: Exercise shared conversation capacity and budget contention across concurrent arms.
# ABOUTME: Run on Linux with pytest scratch/gptoss_swe/test_parallel.py; no API calls.
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import sys
import threading
import time
import pytest

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Linux advisory locks')


def test_legacy_reservations_and_three_arms_share_one_pool(tmp_path):
    from src.eval.capabilities.swebench_mini.fleet_admission import token_slot
    active=0
    peak=0
    guard=threading.Lock()
    def task(arm):
        nonlocal active,peak
        with token_slot(tmp_path,1,5,expires=time.time()+15,fairness_seconds=0,poll=.005):
            with guard:
                active+=1;peak=max(peak,active)
            time.sleep(.03)
            with guard: active-=1
        return arm
    with ExitStack() as legacy:
        for _ in range(2):
            legacy.enter_context(token_slot(tmp_path,1,5,expires=time.time()+15,fairness_seconds=0))
        with ThreadPoolExecutor(max_workers=12) as pool:
            assert sorted(pool.map(task,['base','control','da15']*4))==sorted(['base','control','da15']*4)
        assert 1<=peak<=3
    assert not list(tmp_path.glob('*.json'))


def test_parallel_reservations_cannot_multiply_the_shared_cap(tmp_path):
    from src.infra.endpoints.tinker_budget import Budget
    arms=['base','control','da15']
    budgets=[Budget(tmp_path/'ledger.json',.05,'openai/gpt-oss-120b:peft:131072',a,arms) for a in arms]
    def request(n):
        try: return budgets[n%3].reserve(1000,1000)
        except RuntimeError: return None
    with ThreadPoolExecutor(max_workers=12) as pool:
        results=list(pool.map(request,range(100)))
    saved=json.loads((tmp_path/'ledger.json').read_text())
    assert len(saved['requests'])==sum(x is not None for x in results)
    assert sum(x['upper_usd'] for x in saved['requests'].values())<=.05
    assert any(x is None for x in results)
    for budget in budgets: budget.owner.close()
