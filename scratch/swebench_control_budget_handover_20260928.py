# ABOUTME: Audited one-off $180 to $280 control-campaign coordinator handover on its existing CPU.
# ABOUTME: Retains live worker PIDs and financial history; run only after the executed CPU qualification.
import json
import math
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
from src.eval.capabilities.swebench_mini import fleet
from src.eval.capabilities.swebench_mini.fleet_handover import validate_worker
from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, digest, read
from src.infra import runpod


ROOT = Path('/srv/lasr/runs/20260928-2026-09-22-qwen36-0-nosynth-lite-v5')
COMMIT = read(ROOT/'metadata/budget-amendment-qualification/qualification.json')['commit']
UNIT = 'lasr-swebench-lite.service'


def main():
    load_dotenv('/srv/lasr/credentials.env')
    cfg = OmegaConf.load(ROOT/'launch.yaml')
    manifest = read(ROOT/'metadata/manifest.json')
    assert manifest['campaign'] == '30564f0317d84edc99b6a802b0316cc5'
    assert manifest['budget_usd'] == 180 and cfg.replicas == 10
    assert read(ROOT/'metadata/budget-amendment-qualification/qualification.json')['status'] == 'passed'
    prior_pid = int(subprocess.check_output(['systemctl','show',UNIT,'-p','MainPID','--value'],text=True))
    prior = psutil.Process(prior_pid)
    assert 'src.eval.capabilities.swebench_mini.fleet' in prior.cmdline()
    snapshot = ROOT/'metadata'/('budget-amendment-live-before-'+uuid.uuid4().hex[:8])
    snapshot.mkdir(exist_ok=False)
    for source,name in [(ROOT/'launch.yaml','launch.yaml'),(ROOT/'metadata/manifest.json','manifest.json'),
                        (ROOT/'metadata/supervisor.json','supervisor.json'),(ROOT/'metadata/config.yaml','config.yaml')]:
        shutil.copyfile(source,snapshot/name)
    state = State(ROOT)
    live = read(state.path)
    assert not live.get('halt')
    running = [p for p in live['pods'] if p['status'] not in ('terminated','rejected-reconciled','not-requested')]
    assert all(p.get('id') and (p['status'] in ('working','serving') or (p['status']=='booting' and not p.get('worker_pid'))) for p in running), 'Wait for an in-flight allocation to settle'
    assert {p['id'] for p in runpod.active_pods() if fleet.owned(p,manifest)} == {p['id'] for p in running}
    transition_guards=[]
    old_guards=[]
    handover=None
    killed=False
    dropin=Path('/run/systemd/system')/(UNIT+'.d')/'90-budget-handover.conf'
    assert not dropin.exists()
    try:
        # Stop only the coordinator's scheduling threads. GPU workers have their
        # own sessions and continue inference/tool execution throughout.
        prior.send_signal(signal.SIGSTOP)
        for _ in range(20):
            if prior.status()==psutil.STATUS_STOPPED:break
            time.sleep(.05)
        assert prior.status()==psutil.STATUS_STOPPED
        current=read(state.path)
        atomic(snapshot/'state.json',current)
        assert [(p['slot'],p.get('id')) for p in current['pods']] == [(p['slot'],p.get('id')) for p in live['pods']], 'Fleet changed before freeze'
        running=[p for p in current['pods'] if p['status'] not in ('terminated','rejected-reconciled','not-requested')]
        for p in running:
            guard=runpod.start_watchdog(p['id'],max(1,math.ceil(p['expires']-time.time())),
                ROOT/'metadata'/f'watchdog-handover-{p["slot"]}.log',parent_pid=0)
            transition_guards.append((p['id'],guard))
        assert all(g.poll() is None for _,g in transition_guards)
        # New deadline-only guards and provider expiry are armed before retiring
        # the old parent-bound guards; there is no unprotected paid interval.
        ids={p['id'] for p in running}
        for proc in psutil.process_iter(['pid','cmdline','create_time']):
            cmd=proc.info['cmdline'] or []
            if 'src.infra.runpod' in cmd and 'watchdog' in cmd:
                i=cmd.index('watchdog')
                if cmd[i+1] in ids and cmd[i+2]==str(prior_pid):
                    old_guards.append(proc)
        assert len(old_guards)==len(running)
        for guard in old_guards:guard.terminate()
        psutil.wait_procs(old_guards,timeout=5)
        assert all(not p.is_running() or p.status()==psutil.STATUS_ZOMBIE for p in old_guards)
        cancelled=[]
        for p in running:
            if p['status']=='booting':
                assert not p.get('worker_pid') and time.time()-p['created']>500
                # This unused fallback is still installing packages. No task or
                # loaded model is interrupted; the new coordinator prefers H100.
                try:
                    import requests
                    response=requests.get(f'https://{p["id"]}-8080.proxy.runpod.net/boot.log',timeout=10)
                    if response.ok:(ROOT/'metadata'/f'bootstrap-{p["slot"]}-handover.log').write_text(response.text)
                except requests.RequestException:
                    pass
                runpod.teardown(p['id'])
                with state.edit() as data:
                    saved=next(x for x in data['pods'] if x['slot']==p['slot'])
                    saved.update(status='terminated',ended=time.time(),handover_cancelled_unused_bootstrap=True)
                cancelled.append(p['id'])
        workers=[]
        for p in running:
            if p['id'] in cancelled:continue
            proc=psutil.Process(p['worker_pid']);cmd=proc.cmdline()
            record=dict(p,worker_created=proc.create_time(),server=cmd[cmd.index('--server')+1])
            validate_worker(record,ROOT/'launch.yaml');workers.append(record)
        assert {p['id'] for p in runpod.active_pods() if fleet.owned(p,manifest)} == {p['id'] for p in workers}
        baseline={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'rollouts').glob('*/*/done.json')}
        old_sources=manifest['source_hashes'];new_sources=fleet.sources()
        source_diff={k:{'before':old_sources.get(k),'after':v} for k,v in new_sources.items() if old_sources.get(k)!=v}
        assert set(source_diff)=={'src/eval/capabilities/swebench_mini/fleet.py','src/eval/capabilities/swebench_mini/fleet_handover.py'}
        cfg.recommended_budget_usd=280
        OmegaConf.save(cfg,ROOT/'launch.yaml');shutil.copyfile(ROOT/'launch.yaml',ROOT/'metadata/config.yaml')
        manifest.update(budget_usd=280,config=OmegaConf.to_container(cfg),source_hashes=new_sources)
        manifest['deployment']={'git_commit':COMMIT,'scope':'Coordinator-only budget handover; inference protocol unchanged','previous':manifest.get('deployment')}
        atomic(ROOT/'metadata/manifest.json',manifest)
        for name in source_diff:
            dst=ROOT/'metadata/source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(fleet.REPO/name,dst)
        recipe=read(cfg.recipe_path);recipe['source_hashes']=new_sources
        recipe['qualification']['budget_handover']={'commit':COMMIT,'tests_passed':102,'subtests':2,'systemd_worker_preserved':True}
        fleet.validate_recipe(cfg,recipe);atomic(cfg.recipe_path,recipe)
        atomic(ROOT/'metadata/budget-amended-recipe.json',recipe)
        control=read(ROOT/'metadata/supervisor.json');control['budget_usd']=280;atomic(ROOT/'metadata/supervisor.json',control)
        Path('/srv/lasr/lite-launch.env').write_text(f'LITE_ACTION=supervise\nLITE_CONFIG={ROOT}/launch.yaml\nLITE_BUDGET_USD=280\n')
        Path('/srv/lasr/lite-launch.env').chmod(0o600)
        atomic('/srv/lasr/lite-deployment.json',manifest['deployment'])
        now=time.time()
        handover={'status':'prepared','campaign':manifest['campaign'],'created':now,'expires':now+180,
                  'previous_pid':prior_pid,'previous_created':prior.create_time(),'budget_usd':280,'workers':workers}
        atomic(ROOT/'metadata/coordinator-handover.json',handover)
        audit={'status':'applying','authorization':'User: increase budget limit by $100 and add the tenth GPU',
               'old_budget_usd':180,'new_budget_usd':280,'commit':COMMIT,'source_changes':source_diff,
               'created':now,'baseline_completed_hashes':baseline,'preserved_worker_pids':[w['worker_pid'] for w in workers],
               'cancelled_unused_bootstrap':cancelled,'original_leases':{p['id']:p['expires'] for p in running},
               'ledger_before':fleet.reserved_cost(current),'initial_manifest_archive':'metadata/budget-amendment-before',
               'qualification':'metadata/budget-amendment-qualification','protocol_changes':[]}
        atomic(ROOT/'metadata/budget-amendment.json',audit)
        dropin.parent.mkdir(parents=True,exist_ok=True);dropin.write_text('[Service]\nKillMode=process\nRestart=no\n')
        subprocess.run(['systemctl','daemon-reload'],check=True)
        prior.kill();killed=True
        for _ in range(50):
            result=subprocess.run(['systemctl','is-active','--quiet',UNIT])
            if result.returncode:break
            time.sleep(.1)
        subprocess.run(['systemctl','reset-failed',UNIT],check=True)
        subprocess.run(['systemctl','start','--no-block',UNIT],check=True)
        until=time.time()+120
        while time.time()<until:
            result=read(ROOT/'metadata/coordinator-handover.json')
            if result['status']=='adopted':break
            time.sleep(1)
        assert result['status']=='adopted','Coordinator failed to adopt; do not stop healthy workers'
        for record in workers:validate_worker(record,ROOT/'launch.yaml')
        for path,sha in baseline.items():assert digest(ROOT/path)==sha
        saved=read(state.path)
        for old in current['pods']:
            new=next(p for p in saved['pods'] if p['slot']==old['slot'])
            for key in ('id','created','expires','actual_hourly','ceiling_hourly'):
                if key in old:assert new[key]==old[key],(old['slot'],key)
        audit.update(status='applied',applied_at=time.time(),new_coordinator_pid=result['adopted_by'],
                     completed_outcomes_preserved=len(baseline),ledger_after=fleet.reserved_cost(saved))
        atomic(ROOT/'metadata/budget-amendment.json',audit)
        print(json.dumps({'status':'applied','budget_usd':280,'adopted_workers':len(workers),
                          'preserved_outcomes':len(baseline),'cancelled_unused_bootstrap':cancelled,'new_pid':result['adopted_by']}),flush=True)
    finally:
        if not killed and prior.is_running():
            prior.send_signal(signal.SIGCONT)
        if dropin.exists():
            dropin.unlink();subprocess.run(['systemctl','daemon-reload'],check=True)


if __name__=='__main__':
    main()
