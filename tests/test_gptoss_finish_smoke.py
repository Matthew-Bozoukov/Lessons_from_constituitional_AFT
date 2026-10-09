# ABOUTME: Docker missing-container checks recognize case variants without hiding other errors.
# ABOUTME: Offline coverage for the completion owner's preflight ownership fence.
from types import SimpleNamespace
import pytest
from scratch.gptoss_swe import finish_smoke


@pytest.mark.parametrize('error',['error: no such object: old','Error: No such container: old'])
def test_missing_container_is_stopped(monkeypatch,error):
    monkeypatch.setattr(finish_smoke.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr=error))
    assert finish_smoke.running('old') is False


def test_daemon_error_is_not_stopped(monkeypatch):
    monkeypatch.setattr(finish_smoke.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='cannot connect to daemon'))
    with pytest.raises(AssertionError):finish_smoke.running('old')


def test_parallel_pool_is_four_independent_configs():
    from omegaconf import OmegaConf
    import threading
    barrier=threading.Barrier(4)
    seen=[]
    cfg=OmegaConf.create(dict(max_infrastructure_attempts=99,step_limit=500))
    def consume(endpoint,model,worker,name,allowed,expires,admission):
        barrier.wait(timeout=5)
        seen.append((allowed[0],worker.max_infrastructure_attempts,name,worker.step_limit))
    ids=['a','b','c','d']
    finish_smoke.consume_pool(consume,'http://local',cfg,'base',ids,{i:dict(max_total_attempt_records=n+3) for n,i in enumerate(ids)},{},4)
    assert sorted((i,n) for i,n,_,_ in seen)==[('a',3),('b',4),('c',5),('d',6)]
    assert len({name for _,_,name,_ in seen})==4
    assert all(steps==500 for *_,steps in seen) and cfg.max_infrastructure_attempts==99


def test_pool_rejects_duplicate_task_ids():
    from omegaconf import OmegaConf
    with pytest.raises(AssertionError):
        finish_smoke.consume_pool(None,'local',OmegaConf.create({}),'base',['a','a'],{},{},4)
