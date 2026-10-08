# ABOUTME: Three checkpoint arms share 80 whole-task slots and the original $300 Tinker ledger.
# ABOUTME: Run once on the CPU: python -m scratch.gptoss_swe.parallel; preserve original drain and receipts.
from contextlib import ExitStack
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini.fleet_state import atomic, read, lock
from src.eval.capabilities.swebench_mini.fleet_admission import token_slot
from scratch.gptoss_swe.campaign import ROOT, verify


def old_active():
    return subprocess.run(['systemctl','is-active','--quiet','lasr-gptoss-swe-three.service']).returncode == 0


def main():
    from src.infra.huggingface import hf_api, hf_repo_id
    from src.eval.run_eval import _run_repo
    from src.infra.endpoints.tinker import resolve_tinker_target
    manifest = read('scratch/gptoss_swe/manifest.json')
    status_path = ROOT/'campaign-status-parallel.json'
    with lock(ROOT/'.parallel-campaign.lock', nonblocking=True), ExitStack() as reserved:
        assert not status_path.exists(), 'Never restart or erase this campaign'
        assert not (ROOT/'campaign-status-80.json').exists(), 'Sequential continuation already launched'
        assert read(ROOT/'parallel-transition.json')['stopped_only_unstarted_waiter'] is True
        assert (ROOT/'inference-budget.json').exists(), 'Original ledger required'
        prior = read(ROOT/'campaign-status.json')
        assert prior['active_arm'] == 'base' and prior['phase'] in ('running','held')
        base_states = list((ROOT/'base').glob('*/metadata/state.json'))
        assert len(base_states)==1
        old_base = base_states[0].parents[1]
        old_state = read(base_states[0])
        assert old_state['deadline'] is not None and old_state['halt'] is None
        assert all(t['status'] in ('valid','pending','running') for t in old_state['tasks'].values())
        assert sum(t['status']=='running' for t in old_state['tasks'].values()) <= 16
        # Original processes predate the shared task gate. Reserve all 16 of
        # their possible slots until systemd confirms their entire service ended.
        for _ in range(16):
            reserved.enter_context(token_slot(ROOT/'task-slots',1,80,
                expires=time.time()+7*86400,fairness_seconds=0))
        status = dict(phase='running',started=time.time(),global_task_limit=80,
            legacy_reserved_slots=16,cap_usd=300,arms={},source=read('/srv/lasr/gptoss-three-deployment-parallel.json'))
        atomic(status_path,status)
        jobs = {}

        def launch(arm):
            target=manifest['arms'][arm]
            cfg=OmegaConf.load('scratch/gptoss_swe/pilot.yaml')
            cfg.source_deployment=status['source']
            cfg.campaign='gptoss-'+arm+'-20261008'
            cfg.run_name='gptoss120b-'+arm
            cfg.output_root=str(ROOT/(arm+'-parallel'))
            cfg.instance_ids=None
            cfg.workers=80
            cfg.global_task_limit=80
            cfg.grading.max_workers=12
            cfg.qualify_first=True
            cfg.qualification_instance=manifest['qualification_instance']
            cfg.tinker.budget_usd=299.99
            cfg.tinker.budget_ledger=str(ROOT/'inference-budget.json')
            cfg.tinker.budget_checkpoints=list(manifest['arms'].values())
            cfg.subset.fraction=1.0
            cfg.subset.n=None
            profile=Path(cfg.cached_campaign)/'metadata/task-schedule.json'
            cfg.runtime_priority=dict(path=str(profile),sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
            if arm=='base':
                assert not old_active()
                assert read(ROOT/'campaign-status.json')['phase']=='held'
                from scratch.gptoss_swe.resume import validate_snapshot
                validate_snapshot(read(old_base/'metadata/state.json'),read(old_base/'metadata/manifest.json'),
                                  OmegaConf.to_container(cfg,resolve=True),target)
                cfg.resume_from=str(old_base)
            config=ROOT/(arm+'-parallel.yaml')
            OmegaConf.save(cfg,config)
            log=(ROOT/(arm+'-parallel.log')).open('x')
            proc=subprocess.Popen([sys.executable,'-u','-m','scratch.gptoss_swe.run','--name','swebench_mini',
                                   '--config',str(config),'--target',target],stdout=log,stderr=subprocess.STDOUT)
            jobs[arm]=(proc,log,cfg)
            status['arms'][arm]=dict(phase='running',target=target,started=time.time(),pid=proc.pid,config=str(config))
            atomic(status_path,status)

        launch('control')
        launch('da15')
        base_launched=False
        while jobs or not base_launched:
            if not base_launched and not old_active():
                reserved.close()
                status['legacy_reserved_slots']=0
                launch('base')
                base_launched=True
            for arm,(proc,log,cfg) in list(jobs.items()):
                code=proc.poll()
                if code is None:
                    continue
                log.close()
                entry=status['arms'][arm]
                entry.update(eval_exit=code,finished=time.time())
                try:
                    assert code==0, 'Evaluation exited '+str(code)
                    paths=list(Path(cfg.output_root).glob('*/results/results.json'))
                    assert len(paths)==1
                    result=read(paths[0]);assert result['n_valid_rollouts']==result['n_graded']==result['n_total']==300
                    repo=hf_repo_id(_run_repo('swebench_mini',resolve_tinker_target(entry['target']).model_key,
                                             cfg.run_name,cfg.protocol))
                    revision=hf_api().dataset_info(repo).sha
                    count=verify(paths[0].parents[1],repo,revision)
                    entry.update(phase='verified',result=result,repo=repo,revision=revision,verified_files=count)
                except Exception as exc:
                    entry.update(phase='held',error=str(exc))
                del jobs[arm]
                atomic(status_path,status)
            time.sleep(5)
        status.update(phase='complete' if all(a['phase']=='verified' for a in status['arms'].values()) else 'held',
                      finished=time.time())
        atomic(status_path,status)


if __name__=='__main__':
    main()
