# ABOUTME: Recover only the pilot's proven-unsent ODCV directory-preparation failure.
# ABOUTME: Hold the existing supervisory poller, preserve SWE workers, and release the same GPU promptly.
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
from scratch.gptoss_swe import mxfp4_pilot as pilot


def same_process(identity):
    try:
        p = psutil.Process(identity['pid'])
        return p if p.create_time() == identity['created'] else None
    except psutil.NoSuchProcess:
        return None


def verify_unsent(backend):
    r = pilot.ROOT/backend
    status = pilot.read(r/'odcv/status.json')
    log = (r/'odcv-owner.log').read_text(encoding='utf-8')
    assert status['phase'] == 'held' and 'FileNotFoundError' in status['error']
    assert 'OmegaConf.save(cfg, cfg_path)' in log and 'odcv_config.yaml' in log and 'FileNotFoundError' in log
    assert not list((r/'odcv').rglob('rollout_*.json'))
    assert not (r/'completed-passes.json').exists()
    for folder in ('odcv-traces', 'odcv-raw'):
        assert not list((r/folder).rglob('*.json')), 'ODCV request evidence requires manual review'
    if backend == 'tinker':
        assert pilot.read(r/'odcv-budget.json')['requests'] == {}
    return dict(backend=backend, zero_generation_requests=True, error=status['error'])


def child(backend):
    os.environ.update(pilot.environment())
    original_claim = pilot.claim
    expected = pilot.ROOT/backend/'odcv-owner-started.json'
    def recovery_claim(path):
        assert path == expected
        original_claim(path.with_name('odcv-recovery-owner-started.json'))
    pilot.claim = recovery_claim
    # The stock runner saves its config before creating pass directories.
    (pilot.ROOT/backend/'odcv').mkdir(parents=True, exist_ok=True)
    pilot.odcv(backend)


def resume_original(receipt):
    main = same_process(receipt['original'])
    if main is not None:
        main.resume()
    pilot.save(pilot.ROOT/'odcv-recovery-resumed.json', dict(at=time.time(), original_alive=main is not None))


def guard():
    receipt = pilot.read(pilot.ROOT/'odcv-recovery-handoff.json')
    while same_process(receipt['recovery']) is not None:
        time.sleep(5)
    if not (pilot.ROOT/'odcv-recovery-resumed.json').exists():
        # Existing GPU watchdog and provider expiry remain untouched throughout.
        resume_original(receipt)


def run():
    from src.infra import runpod
    os.environ.update(pilot.environment())
    pilot.claim(pilot.ROOT/'odcv-recovery-coordinator.json')
    evidence = [verify_unsent(b) for b in ('tinker', 'mxfp4')]
    original_id = pilot.read(pilot.ROOT/'owner-started.json')['pid']
    main = psutil.Process(original_id)
    assert main.cmdline()[-2:] == ['scratch.gptoss_swe.mxfp4_pilot', 'run']
    assert pilot.read(pilot.ROOT/'status.json')['phase'] == 'inference'
    assert not (pilot.ROOT/'gpu-cleanup.json').exists()
    for backend in ('tinker','mxfp4'):
        old_id = pilot.read(pilot.ROOT/backend/'odcv-owner-started.json')['pid']
        assert not psutil.pid_exists(old_id), 'Original ODCV worker must have exited'
    # Reuse the current server credential in memory only. Never archive Docker Env.
    server_shim = json.loads(subprocess.check_output(['docker','inspect',pilot.named('mxfp4','swe-shim')]))[0]
    assert server_shim['State']['Running']
    os.environ['MXFP4_API_KEY'] = next(x.split('=',1)[1] for x in server_shim['Config']['Env'] if x.startswith('MXFP4_API_KEY='))
    receipt = dict(at=time.time(), original=dict(pid=main.pid, created=main.create_time()),
                   recovery=dict(pid=os.getpid(), created=psutil.Process().create_time()), evidence=evidence,
                   reason='Only supervisory polling is suspended; all existing SWE inference continues. Original resumes after ODCV judging. GPU released as soon as MXFP4 inference ends.')
    pilot.save(pilot.ROOT/'odcv-recovery-handoff.json', receipt)
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    pilot.save(pilot.ROOT/'odcv-recovery-source.json',dict(commit=source))
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(pilot.ROOT/'odcv-recovery-source.tar.gz'),'HEAD'],check=True)
    with (pilot.ROOT/'odcv-recovery-guard.log').open('x') as log:
        p = subprocess.Popen([sys.executable,'-m',__spec__.name,'guard'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    pilot.save(pilot.ROOT/'odcv-recovery-guard.json',dict(pid=p.pid))
    jobs=[]
    main.suspend()
    try:
        for backend in ('tinker','mxfp4'):
            r=pilot.ROOT/backend; archive=r/'odcv-startup-failure';archive.mkdir()
            for name in ('odcv-owner-started.json','odcv-owner.log','odcv-shim.log','odcv-shim-settings.json','odcv-identity.json','odcv-budget.json'):
                if (r/name).exists():shutil.copyfile(r/name,archive/name)
            shutil.copyfile(r/'odcv/status.json',archive/'status.json')
            pilot.save(archive/'hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir()})
            name=pilot.named(backend,'odcv-shim')
            obj=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            assert not obj['State']['Running']
            subprocess.run(['docker','rename',name,name+'-startup-failed'],check=True)
            with (r/'odcv-recovery-owner.log').open('x',encoding='utf-8') as log:
                p=subprocess.Popen([sys.executable,'-m',__spec__.name,'child','--backend',backend],env=pilot.environment(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            jobs.append((backend,p))
        pilot.save(pilot.ROOT/'odcv-recovery-status.json',dict(phase='inference',at=time.time()))
        mx = next(p for b,p in jobs if b=='mxfp4')
        while True:
            od_done=(pilot.ROOT/'mxfp4/odcv-inference-ended.json').exists() or mx.poll() is not None
            swe_done=(pilot.ROOT/'mxfp4/swe-inference-ended.json').exists()
            swpid=pilot.read(pilot.ROOT/'mxfp4/swe-owner-started.json')['pid']
            swe_done=swe_done or not psutil.pid_exists(swpid)
            if od_done and swe_done:break
            time.sleep(5)
        pilot.collect_gpu()
        pod=pilot.read(pilot.ROOT/'pod.json')['id']
        assert runpod.terminate(pod)
        ended=dict(pod=pod,terminated=True,at=time.time(),owner='odcv-preparation-recovery')
        pilot.save(pilot.ROOT/'odcv-recovery-gpu-cleanup.json',ended)
        pilot.save(pilot.ROOT/'gpu-cleanup.json',ended)
        tunnel=psutil.Process(pilot.read(pilot.ROOT/'tunnel.json')['pid'])
        assert 'ssh' in tunnel.name().lower() and str(pilot.C.raw_port) in ' '.join(tunnel.cmdline())
        tunnel.terminate()
        pilot.save(pilot.ROOT/'odcv-recovery-status.json',dict(phase='judging',at=time.time()))
        exits={b:p.wait() for b,p in jobs}
        pilot.save(pilot.ROOT/'odcv-recovery-exits.json',exits)
        pilot.save(pilot.ROOT/'odcv-recovery-status.json',dict(phase='finished',at=time.time(),exits=exits))
    finally:
        # On ordinary failure allow original supervisor/expiry to retain teardown ownership.
        resume_original(receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','child','guard'])
    p.add_argument('--backend',choices=['tinker','mxfp4']);a=p.parse_args()
    if a.action=='child':child(a.backend)
    else:globals()[a.action]()
