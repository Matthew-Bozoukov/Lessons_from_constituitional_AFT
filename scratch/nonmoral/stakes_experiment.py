# ABOUTME: Continues the authorized matched stakes training into two bounded ODCV runs.
# ABOUTME: Uses existing trainers/eval owners, verifies public pinned artifacts and never rents a GPU directly.
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from src.infra.huggingface import hf_api,hf_download,hf_org
from scratch.nonmoral.stakes import write_json,file_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def adopted_eval_status(out, arm, model, frozen_hash):
    """Observe an explicitly transferred evaluation; never launch its replacement."""
    claim=read(out/(arm+'_eval_handoff.json'))
    assert claim['model']=={k:model[k] for k in ('repo','revision')}
    plan=OmegaConf.to_container(OmegaConf.load(claim['plan']),resolve=True)
    assert file_sha256(claim['plan'])==claim['plan_sha256']
    assert plan['target']==model['repo'] and plan['target_revision']==model['revision']
    assert plan['eval_config_sha256']==frozen_hash
    status_path=Path(plan['output_dir'])/'broader_eval_status.json'
    if not status_path.exists():
        raise RuntimeError('Transferred evaluation has no owner state; inspect rather than relaunch')
    status=read(status_path)
    assert status['target']==model['repo'] and status['target_revision']==model['revision']
    if not status.get('termination_verified'):
        if status.get('phase')=='failed' and not status.get('pod_id'):
            raise RuntimeError('Transferred evaluation failed before rental; inspect its logs')
        if time.time()-status.get('updated_at_unix',0)>300:
            raise RuntimeError('Transferred evaluation owner state is stale; inspect rather than duplicate')
        return None
    assert status.get('evaluation_driver_completed') and status.get('local_log_backup',{}).get('verified'), 'Transferred eval did not complete cleanly'
    cost=float(status['estimated_gpu_and_storage_usd'])+float(status.get('judge_charged_or_reserved_usd',0))
    return dict(status_file=str(status_path),exit_code=0,adopted=True,cost_usd=cost)


def verify_models(training, plan, out):
    assert training['terminated'] and training['local_backup']['verified']
    assert sorted(training['completed_arms'])==[0,1]
    assert training['local_backup']['verified_completed_arms']==2
    archive=Path(training['local_backup']['archive'])
    assert archive.stat().st_size==training['local_backup']['bytes']
    if training['local_backup'].get('sha256'):
        assert file_sha256(archive)==training['local_backup']['sha256']
    wanted={a['data_repo']:a for a in plan['arms']}
    models={}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.name.endswith('/adapter/training_meta.json'):
                continue
            stamp=json.load(tar.extractfile(member)); ds=stamp['dataset']
            assert ds['repo'] in wanted and ds['revision']==wanted[ds['repo']]['data_revision']
            assert stamp['base_model']=='Qwen/Qwen3.6-27B' and stamp['base_model_revision']==plan['base_model_revision']
            assert stamp['thinking'] is True and stamp['train_config']['seed']==0
            assert stamp['train_config']['lora']['r']==64 and stamp['train_config']['train']['epochs']==1
            run=member.name.removesuffix('/adapter/training_meta.json')
            meta=json.load(tar.extractfile(run+'/run_meta.json'))
            assert meta['world_size']==2 and meta['n_examples']==9968
            history=meta['log_history']; totals=[x for x in history if 'train_loss' in x]
            assert totals and totals[-1]['step']==623 and math.isfinite(totals[-1]['train_loss'])
            assert meta['config']['train']['token_budget']==8000
            for item in history:
                for key in ('loss','grad_norm'):
                    if key in item:assert math.isfinite(float(item[key]))
            repo=hf_org()+'/'+stamp['organism']; info=hf_api().model_info(repo,files_metadata=True)
            assert not info.private
            remote_stamp=hf_download(repo,'training_meta.json',revision=info.sha)
            assert read(remote_stamp)==stamp
            # Verify published adapter bytes against the already retrieved local archive.
            siblings={f.rfilename:f for f in info.siblings}
            weight_members=[m for m in tar.getmembers() if m.name.startswith(run+'/adapter/') and m.name.endswith('.safetensors')]
            assert weight_members
            for m in weight_members:
                name=m.name.removeprefix(run+'/adapter/')
                local=hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()
                lfs=siblings[name].lfs
                remote_hash=lfs.sha256 if lfs else file_sha256(hf_download(repo,name,revision=info.sha))
                assert local==remote_hash,(repo,name)
            arm=wanted[ds['repo']]['arm']
            models[arm]=dict(repo=repo,revision=info.sha,public=True,train_loss=totals[-1]['train_loss'],
                 dataset=ds,base_revision=stamp['base_model_revision'],local_weights_verified=True)
            write_json(out/f'{arm}_training_meta.json',stamp)
    assert set(models)=={'low','high'}
    write_json(out/'models.json',models)
    return models


def continue_run(out):
    out=Path(out).resolve(); plan=read(out/'train_plan.json')
    lock=out/'continuation.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    state=dict(phase='waiting_for_training',pid=os.getpid(),evaluations={})
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    def save(**items):
        state.update(items);state['updated_epoch']=time.time();write_json(out/'continuation_status.json',state)
    try:
        save()
        while True:
            recovered=out/'recovered_training.json'
            training=read(recovered if recovered.exists() else out/'training/status.json')
            if recovered.exists():
                assert training.get('recovery_evidence') and training.get('full_training_archive_complete') is False
                save(recovery_evidence=training['recovery_evidence'],full_training_archive_complete=False)
            if training.get('terminated') and training.get('local_backup',{}).get('verified'):
                break
            if training.get('phase')=='failed' and (training.get('terminated') or not training.get('owned_pod')):
                raise RuntimeError('Training failed; no evaluation or replacement training launched')
            if training.get('phase') in ('artifact_recovery_required','recovery_deadline_reached'):
                raise RuntimeError('Training backup needs attention; no evaluation launched')
            save(training_phase=training['phase'])
            time.sleep(30)
        save(phase='verifying_adapters')
        models=verify_models(training,plan,out)
        initial=float(plan['project_exposure_before_training_usd'])
        spent=float(training['estimated_gpu_usd'])
        save(models=models,training_cost_usd=spent)
        frozen=out/'odcv_frozen.yaml'; frozen_hash=file_sha256(frozen)
        for index,arm in enumerate(('low','high')):
            if (out/(arm+'_eval_handoff.json')).exists():
                save(phase='awaiting_existing_'+arm+'_evaluation')
                while True:
                    adopted=adopted_eval_status(out,arm,models[arm],frozen_hash)
                    if adopted is not None:
                        break
                    time.sleep(30)
                    save()
                spent+=adopted['cost_usd']
                state['evaluations'][arm]=adopted
                save(project_exposure_usd=initial+spent)
                continue
            remaining=300-initial-spent
            allocation=min(20,(remaining-1)/(2-index))
            judge_cap=min(3.5,allocation-6)
            gpu_cap=min(15,allocation-judge_cap)
            assert judge_cap>=2.5 and gpu_cap>=6,'Insufficient remaining budget for the fixed evaluation; no rental'
            eval_out=out/('evaluation_'+arm)
            eval_plan=dict(target=models[arm]['repo'],target_revision=models[arm]['revision'],
                base_model='Qwen/Qwen3.6-27B',base_revision=plan['base_model_revision'],
                output_dir=str(eval_out),run_name='odcv-stakes-'+arm+'-20260910',
                eval_output_root='C:/nm-stakes',eval_config=str(frozen),eval_config_sha256=frozen_hash,
                expected_cells=80,passes=3,gpu_cap_usd=gpu_cap,backup_reserve_usd=2,
                judge_cap_usd=judge_cap,max_gpu_hourly_usd=4.5,storage_hourly_reserve_usd=0.1)
            path=out/(arm+'_eval_plan.yaml');OmegaConf.save(OmegaConf.create(eval_plan),path)
            from scratch.nonmoral.overnight_baseline import load_plan
            load_plan(path)
            save(phase='evaluating_'+arm,current_eval_plan=eval_plan,project_exposure_before_eval=initial+spent)
            with (out/(arm+'_eval_owner.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-u','scratch/nonmoral/overnight_baseline.py','--plan',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            status=read(eval_out/'broader_eval_status.json')
            spent+=float(status.get('estimated_gpu_and_storage_usd',gpu_cap))+float(status.get('judge_charged_or_reserved_usd',judge_cap))
            state['evaluations'][arm]=dict(status_file=str(eval_out/'broader_eval_status.json'),exit_code=result.returncode)
            save(project_exposure_usd=initial+spent)
            assert result.returncode==0 and status.get('evaluation_driver_completed') and status.get('termination_verified'),f'{arm} evaluation failed; preserve artifacts and review'
            assert status['local_log_backup']['verified']
        save(phase='evaluations_complete_pending_report',project_exposure_usd=initial+spent)
    except BaseException as exc:
        save(phase='needs_attention',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        os.close(fd);lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output_dir')
    continue_run(parser.parse_args().output_dir)
