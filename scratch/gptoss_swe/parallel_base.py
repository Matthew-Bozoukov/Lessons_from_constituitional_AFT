# ABOUTME: Admission-drain the serial base worker, then complete its same tasks with four workers.
# ABOUTME: Preserve running attempts, original control ownership, ledgers and immutable publication evidence.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ['GPTOSS_FINISH_CONFIG']='scratch/gptoss_swe/parallel_base.yaml'
from scratch.gptoss_swe import finish_smoke as f


def drain():
    from src.eval.capabilities.swebench_mini.fleet_state import State,atomic
    state=State(Path(f.C.base_prior)/'swe')
    receipt=f.ROOT/'admission-drain.json'
    with state.edit() as data:
        assert not receipt.exists() and data['deadline'] is None and not data.get('halt')
        assert len(data['tasks'])==10
        before=json.loads(json.dumps(data))
        # Worker captured each active task's deadline earlier; this only prevents new claims.
        data['deadline']=time.time()
        atomic(receipt,dict(at=time.time(),before=before,new_deadline=data['deadline'],note='Only admission changes; current paid attempt continues. Four workers take remaining tasks after owner exits.'))


def copy_control():
    prior=Path(f.C.previous_root)/'control'
    while True:
        s=json.loads((prior/'status.json').read_text())
        if s['phase']=='graded':break
        if s['phase']=='held':raise RuntimeError('Control held: '+str(s))
        time.sleep(15)
    target=f.ROOT/'control';assert not target.exists()
    hashes={}
    for p in sorted(prior.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(prior);dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
        hashes[rel.as_posix()]=digest
    f.save(f.ROOT/'control-copy-hashes.json',hashes)
    return 0


def run():
    f.ROOT.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    with (f.ROOT/'owner-started.json').open('x') as out:json.dump(dict(pid=os.getpid(),at=time.time(),source=source),out)
    assert (f.ROOT/'admission-drain.json').exists()
    assert (f.ROOT/'finalizer-superseded.json').exists()
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(f.ROOT/'source.tar.gz'),'HEAD'],check=True)
    shutil.copyfile(Path(f.C.previous_root)/'spending-cap-removal.json',f.ROOT/'spending-cap-removal.json')
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    def base():
        with (f.ROOT/'base-owner.log').open('x',encoding='utf-8') as out:
            return subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.finish_smoke','arm','--arm','base'],env=f.environment(),stdout=out,stderr=subprocess.STDOUT).returncode
    try:
        f.save(f.ROOT/'status.json',dict(phase='waiting_for_drain_then_parallel',at=time.time(),workers=4))
        with ThreadPoolExecutor(max_workers=2) as pool:
            b=pool.submit(base);c=pool.submit(copy_control)
            exits=dict(base=b.result(),control=c.result())
        f.save(f.ROOT/'arm-exits.json',exits);assert not any(exits.values()),exits
        f.save(f.ROOT/'status.json',dict(phase='publishing',at=time.time()))
        f.publish()
        ids=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(f.C.campaign)],text=True).split()
        for cid in ids:
            assert not f.running(cid);subprocess.run(['docker','rm',cid],check=True)
        f.save(f.ROOT/'cleanup.json',dict(at=time.time(),owned_remaining=[]))
        f.save(f.ROOT/'status.json',dict(phase='complete',at=time.time()))
    except BaseException as exc:
        f.save(f.ROOT/'status.json',dict(phase='held',at=time.time(),error=repr(exc)));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','drain']);a=p.parse_args();globals()[a.action]()
