# ABOUTME: Finish only unsent MXFP4 ODCV work using a verified short Windows directory alias.
# ABOUTME: Preserve nine cached outcomes and hold the prior GPU poller until safe closeout.
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
from omegaconf import OmegaConf
from scratch.gptoss_swe import mxfp4_pilot as pilot
from scratch.gptoss_swe.mxfp4_odcv_recovery import same_process

ALIAS=Path.home()/'.lasr/mx12'


def short_path(ws):
    target=Path(ws).resolve()
    alias=ALIAS/target.relative_to(Path.cwd().resolve())
    assert alias.resolve()==target and len(str(alias))<240
    return alias


def resume(receipt):
    p=same_process(receipt['prior'])
    if p:p.resume()
    pilot.save(pilot.ROOT/'path-recovery-resumed.json',dict(at=time.time(),prior_alive=bool(p)))


def guard():
    receipt=pilot.read(pilot.ROOT/'path-recovery-handoff.json')
    while same_process(receipt['owner']):time.sleep(5)
    if not (pilot.ROOT/'path-recovery-resumed.json').exists():resume(receipt)


def run():
    from src.infra import runpod
    from src.eval.misalignment.odcv import runner,odcv_rollout,odcv_judge,progress_judge,passes
    os.environ.update(pilot.environment())
    r=pilot.ROOT/'mxfp4';od=r/'odcv'
    pilot.claim(pilot.ROOT/'path-recovery-owner-started.json')
    assert pilot.read(pilot.ROOT/'tinker-user-stop.json')['prohibit_new_inference']
    assert pilot.read(od/'status.json')['phase']=='held'
    assert 'NotADirectoryError' in pilot.read(od/'status.json')['error']
    assert not runpod._parent_alive(pilot.read(r/'odcv-recovery-owner-started.json')['pid'])
    old_passes=runner._prior_passes(str(od),pilot.named('mxfp4','odcv'))
    assert len(old_passes)==1
    records=list(old_passes[0].glob('agent_logs/*/experiments/*/messages_record.txt'))
    assert len(records)==9 and all(passes.cell_usable(p.parent) for p in records)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in records}
    missing=old_passes[0]/'workspaces/incentivized/Ai-Moderator-Content-Censorship-Backdoor'
    assert missing.exists()
    cfg=OmegaConf.load(r/'odcv.yaml')
    expected={f'{v}/{s}' for v,ss in cfg.include_scenarios.items() for s in ss}
    observed={('incentivized' if '-incentivized' in str(p) else 'mandated')+'/'+p.parent.name for p in records}
    assert expected-observed=={'incentivized/Ai-Moderator-Content-Censorship-Backdoor'}
    missing_record=old_passes[0]/'agent_logs'/f'{cfg.model_key}-incentivized'/'experiments/Ai-Moderator-Content-Censorship-Backdoor'
    assert not list(missing_record.glob('*')), 'Unexpected request/log evidence at missing cell'
    subprocess.run(['docker','compose','config','--quiet'],cwd=short_path(missing),capture_output=True,check=True)
    prior_receipt=pilot.read(pilot.ROOT/'odcv-recovery-handoff.json')['recovery']
    prior=same_process(prior_receipt);assert prior is not None
    assert prior.cmdline()[-2:]==['scratch.gptoss_swe.mxfp4_odcv_recovery','run']
    assert not (pilot.ROOT/'gpu-cleanup.json').exists()
    shim=json.loads(subprocess.check_output(['docker','inspect',pilot.named('mxfp4','swe-shim')]))[0]
    assert shim['State']['Running']
    os.environ['MXFP4_API_KEY']=next(x.split('=',1)[1] for x in shim['Config']['Env'] if x.startswith('MXFP4_API_KEY='))
    receipt=dict(at=time.time(),prior=prior_receipt,owner=dict(pid=os.getpid(),created=psutil.Process().create_time()),
        hashes=hashes,alias=str(ALIAS),preserved_outcomes=9,remaining_rollouts=21,never_run_tinker=True)
    pilot.save(pilot.ROOT/'path-recovery-handoff.json',receipt)
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(pilot.ROOT/'path-recovery-source.tar.gz'),'HEAD'],check=True)
    with (pilot.ROOT/'path-recovery-guard.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-m',__spec__.name,'guard'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    pilot.save(pilot.ROOT/'path-recovery-guard.json',dict(pid=p.pid))
    prior.suspend()
    ended=threading.Event();monitor=None;errors=[]
    def gpu_monitor():
        try:
            ended.wait()
            swpid=pilot.read(r/'swe-owner-started.json')['pid']
            while not (r/'swe-inference-ended.json').exists() and runpod._parent_alive(swpid):time.sleep(5)
            pilot.collect_gpu();pod=pilot.read(pilot.ROOT/'pod.json')['id']
            assert runpod.terminate(pod)
            result=dict(pod=pod,terminated=True,at=time.time(),owner='path-recovery')
            pilot.save(pilot.ROOT/'path-recovery-gpu-cleanup.json',result)
            pilot.save(pilot.ROOT/'gpu-cleanup.json',result)
        except BaseException as e:errors.append(repr(e))
    try:
        archive=r/'odcv-path-failure';archive.mkdir()
        for name in ('odcv-recovery-owner.log','odcv-shim.log','odcv_config.yaml'):
            p=r/name if name!='odcv_config.yaml' else od/name
            if p.exists():shutil.copyfile(p,archive/name)
        shutil.copyfile(od/'status.json',archive/'status.json')
        name=pilot.named('mxfp4','odcv-shim')
        obj=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert not obj['State']['Running']
        subprocess.run(['docker','rename',name,name+'-path-failed'],check=True)
        original_compose=odcv_rollout._compose
        odcv_rollout._compose=lambda project,ws,env,args,timeout:original_compose(project,short_path(ws),env,args,timeout)
        # Disable all automatic outcome retries; cached first-pass transcripts are immutable.
        runner.PASS_RETRIES=0;runner._prune_networks=lambda:None
        for module in (odcv_rollout,odcv_judge,progress_judge):module.openrouter_usage=lambda settle_s=0:None
        def preserve_discard(pass_dir):
            assert all(passes.cell_usable(p.parent) for p in pass_dir.glob('agent_logs/*/experiments/*/messages_record.txt'))
            return 0
        runner.discard_unusable=preserve_discard
        audits=[];original_pass=runner._run_pass
        def tracked(*args,**kwargs):
            pilot.phase(od,'inference',pass_number=len(audits)+1,recovery='short-cwd-only')
            result=original_pass(*args,**kwargs);audits.append(result)
            for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
            pilot.save(r/'completed-passes.json',audits)
            if len(audits)==int(pilot.C.odcv_passes):
                pilot.stop_shim('mxfp4','odcv');pilot.save(r/'odcv-inference-ended.json',dict(at=time.time()))
                ended.set();pilot.phase(od,'judging')
            return result
        runner._run_pass=tracked
        cfg.resume_from=str(od)
        pilot.start_shim('mxfp4','odcv')
        monitor=threading.Thread(target=gpu_monitor);monitor.start()
        result=runner.run(SimpleNamespace(model_name='openai/gpt-oss-120b',spec=SimpleNamespace(model_key=cfg.model_key),
            base_url=f'http://127.0.0.1:{pilot.C.ports.mxfp4.odcv}/v1'),cfg,od)
        pilot.save(r/'odcv-summary.json',result);pilot.phase(od,'complete')
    except BaseException as e:
        pilot.phase(od,'held',error=repr(e));raise
    finally:
        pilot.stop_shim('mxfp4','odcv');ended.set()
        if monitor:monitor.join()
        pilot.save(pilot.ROOT/'path-recovery-exit.json',dict(at=time.time(),gpu_errors=errors))
        resume(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','guard']);a=p.parse_args();globals()[a.action]()
