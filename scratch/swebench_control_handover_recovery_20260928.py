# ABOUTME: Recover the diagnosed backup-lock handover failure without replacing surviving workers.
# ABOUTME: Preserve all results and costs; extend only interrupted infrastructure retries from three to four.
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from dotenv import load_dotenv
from omegaconf import OmegaConf
import psutil
from src.eval.capabilities.swebench_mini import fleet, fleet_handover, fleet_session
from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, digest, read, begin_failure_epoch
from src.infra import runpod

ROOT = Path('/srv/lasr/runs/20260928-2026-09-22-qwen36-0-nosynth-lite-v5')
UNIT = 'lasr-swebench-lite.service'
COMMIT = 'dcd20ac8a8c9b4d020a8912728af43dd5465be13'


def main():
    load_dotenv('/srv/lasr/credentials.env')
    cfg=OmegaConf.load(ROOT/'launch.yaml');manifest=read(ROOT/'metadata/manifest.json')
    assert manifest['campaign']=='30564f0317d84edc99b6a802b0316cc5' and manifest['budget_usd']==280
    prior=psutil.Process(int(subprocess.check_output(['systemctl','show',UNIT,'-p','MainPID','--value'],text=True)))
    assert prior.pid==1530789 and prior.status()==psutil.STATUS_STOPPED
    assert 'src.eval.capabilities.swebench_mini.fleet' in prior.cmdline()
    snapshot=ROOT/'metadata'/('budget-handover-recovery-before-'+uuid.uuid4().hex[:8]);snapshot.mkdir(exist_ok=False)
    for name in ['manifest.json','state.json','supervisor.json','coordinator-handover.json','budget-amendment.json','config.yaml']:
        shutil.copyfile(ROOT/'metadata'/name,snapshot/name)
    shutil.copytree(ROOT/'metadata/source',snapshot/'source')
    shutil.copyfile(cfg.recipe_path,snapshot/'recipe.json')
    shutil.copyfile('/srv/lasr/runs/lite-coordinator.log',snapshot/'coordinator.log')
    qualification=Path('/srv/lasr/runs/control-v5-handover-qualification')
    for name in ['backup-lock-regression.log','backup-lock-deployed-regression.log','backup-lock-old-code-regression.log']:
        shutil.copyfile(qualification/name,ROOT/'metadata/budget-amendment-qualification'/name)
    assert '104 passed, 2 subtests passed' in (qualification/'backup-lock-deployed-regression.log').read_text()
    state=State(ROOT);before=read(state.path)
    inventory={p['id']:p for p in runpod.active_pods() if fleet.owned(p,manifest)}
    live=[p for p in before['pods'] if p.get('id') in inventory]
    assert len(live)==5 and all(p['status']=='working' for p in live)
    workers=[]
    for p in live:
        proc=psutil.Process(p['worker_pid']);cmd=proc.cmdline()
        record=dict(p,worker_created=proc.create_time(),server=cmd[cmd.index('--server')+1])
        fleet_handover.validate_worker(record,ROOT/'launch.yaml');workers.append(record)
    guards={}
    for proc in psutil.process_iter(['pid','cmdline']):
        cmd=proc.info['cmdline'] or []
        if 'src.infra.runpod' in cmd and 'watchdog' in cmd:
            i=cmd.index('watchdog')
            if cmd[i+1] in inventory and cmd[i+2]=='0':guards[cmd[i+1]]=proc.pid
    assert set(guards)==set(inventory),'Existing independent deadline guards must remain armed'
    lost=[p for p in before['pods'] if p.get('id') and p['status']=='working' and p['id'] not in inventory]
    assert len(lost)==4
    lost_slots=set(p['slot'] for p in lost)
    for slot in lost_slots:
        record=next(p for p in before['pods'] if p['slot']==slot)
        assert record['id'] not in inventory
        if psutil.pid_exists(record['worker_pid']):
            proc=psutil.Process(record['worker_pid'])
            if proc.status()!=psutil.STATUS_ZOMBIE:
                assert 'src.eval.capabilities.swebench_mini.fleet_worker' in proc.cmdline()
                raise AssertionError('Absent-provider worker still alive; fence only that exact worker first')
        fleet_session.recover_worker(cfg,slot)
    with state.edit() as data:
        assert not data.get('halt')
        for p in data['pods']:
            if p['slot'] in lost_slots:
                p.update(status='terminated',ended=p.get('ended',time.time()),
                    termination_evidence='Verified provider absence after backup-lock handover failure; no model reroll')
        # Diagnosed and repaired coordinator failure, not an unexplained breaker
        # reset: all original errors, attempts and provider costs stay in state.
        begin_failure_epoch(data)
    baseline={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'rollouts').glob('*/*/done.json')}
    cfg.max_infrastructure_attempts=4
    OmegaConf.save(cfg,ROOT/'launch.yaml');shutil.copyfile(ROOT/'launch.yaml',ROOT/'metadata/config.yaml')
    sources=fleet.sources();delta={k:{'before':manifest['source_hashes'].get(k),'after':v} for k,v in sources.items() if manifest['source_hashes'].get(k)!=v}
    assert set(delta)=={'src/eval/capabilities/swebench_mini/fleet.py'}
    manifest.update(config=OmegaConf.to_container(cfg),source_hashes=sources,
        deployment={'git_commit':COMMIT,'scope':'Nonfatal handover checkpoint; infrastructure retry3 to4; model protocol unchanged','previous':manifest['deployment']})
    atomic(ROOT/'metadata/manifest.json',manifest)
    for name in delta:shutil.copyfile(fleet.REPO/name,ROOT/'metadata/source'/name)
    recipe=read(cfg.recipe_path);recipe['source_hashes']=sources
    recipe['qualification']['budget_handover'].update(commit=COMMIT,tests_passed=104,backup_lock_regression=True)
    fleet.validate_recipe(cfg,recipe);atomic(cfg.recipe_path,recipe)
    atomic(ROOT/'metadata/budget-amended-recipe.json',recipe)
    atomic('/srv/lasr/lite-deployment.json',manifest['deployment'])
    audit={'status':'prepared','created':time.time(),'cause':'Required checkpoint collided with surviving publisher; supervisor normal resume fenced four GPUs',
        'commit':COMMIT,'lost_slots':sorted(lost_slots),'preserved_worker_pids':[p['worker_pid'] for p in workers],
        'baseline_completed_hashes':baseline,'source_changes':delta,'original_budget_amendment':'metadata/budget-amendment.json',
        'budget_usd':280,'infrastructure_attempts_before':3,'infrastructure_attempts_after':4,
        'failed_replica_epoch_acknowledgement':read(state.path).get('breaker_failed_replicas_baseline'),
        'tests':104,'subtests':2,'sampling_changes':[],'ledger_preserved':True}
    atomic(ROOT/'metadata/budget-handover-recovery.json',audit)
    now=time.time();atomic(ROOT/'metadata/coordinator-handover.json',{'status':'prepared','campaign':manifest['campaign'],
        'created':now,'expires':now+180,'previous_pid':prior.pid,'previous_created':prior.create_time(),'budget_usd':280,'workers':workers})
    dropin=Path('/run/systemd/system')/(UNIT+'.d')/'90-budget-handover.conf'
    assert not dropin.exists();dropin.parent.mkdir(parents=True,exist_ok=True)
    try:
        dropin.write_text('[Service]\nKillMode=process\nRestart=no\n');subprocess.run(['systemctl','daemon-reload'],check=True)
        prior.kill()
        for _ in range(50):
            if subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode:break
            time.sleep(.1)
        subprocess.run(['systemctl','reset-failed',UNIT],check=True)
        subprocess.run(['systemctl','start','--no-block',UNIT],check=True)
        until=time.time()+120
        while time.time()<until:
            result=read(ROOT/'metadata/coordinator-handover.json')
            if result['status']=='adopted':break
            time.sleep(1)
        assert result['status']=='adopted'
        # Adoption marker alone was insufficient before: require the fleet loop
        # to arm parent-bound monitors, proving the checkpoint was passed.
        while time.time()<until:
            guarded=set()
            for proc in psutil.process_iter(['cmdline']):
                cmd=proc.info['cmdline'] or []
                if 'src.infra.runpod' in cmd and 'watchdog' in cmd:
                    i=cmd.index('watchdog')
                    if cmd[i+2]==str(result['adopted_by']):guarded.add(cmd[i+1])
            if set(inventory)<=guarded:break
            time.sleep(1)
        assert set(inventory)<=guarded,'Adopted workers have not reached monitoring loop'
        for record in workers:fleet_handover.validate_worker(record,ROOT/'launch.yaml')
        for path,sha in baseline.items():assert digest(ROOT/path)==sha
        audit.update(status='applied',applied_at=time.time(),new_coordinator_pid=result['adopted_by'],completed_hashes_preserved=len(baseline))
        atomic(ROOT/'metadata/budget-handover-recovery.json',audit)
        print(json.dumps({'status':'applied','preserved_workers':len(workers),'outcomes':len(baseline),'new_pid':result['adopted_by']}),flush=True)
    finally:
        if dropin.exists():dropin.unlink();subprocess.run(['systemctl','daemon-reload'],check=True)


if __name__=='__main__':main()
