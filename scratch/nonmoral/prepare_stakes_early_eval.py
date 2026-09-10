# ABOUTME: Verifies the completed low-stakes checkpoint and prepares its overlapping ODCV run.
# ABOUTME: Records a bounded budget and ownership handoff; the existing eval owner alone rents GPUs.
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tarfile

from omegaconf import OmegaConf
from scratch.nonmoral.stakes_experiment import read
from scratch.nonmoral.stakes import write_json,file_sha256
from scratch.nonmoral.train_pair import SshExec
from src.infra.huggingface import hf_api,hf_download,hf_org


def prepare(out):
    out=Path(out).resolve(); plan=read(out/'train_plan.json'); training=read(out/'training/status.json')
    assert training['completed_arms']==[0] and training['progress']['arms'][0]['exit']==0
    receipt=read(out/'checkpoint_copies/low_623/status.json')['receipt']
    archive=Path(receipt['archive'])
    assert receipt['verified'] and archive.stat().st_size==receipt['bytes']
    assert file_sha256(archive)==receipt['sha256']
    remote=SshExec(training['host'],port=8000,workdir='/root/work')
    # Derive the run from the verified checkpoint, not a newly invented organism name.
    with tarfile.open(archive) as tar:
        names=tar.getnames(); state_name=next(n for n in names if n.endswith('/trainer_state.json'))
        checkpoint=json.load(tar.extractfile(state_name)); assert checkpoint['global_step']==623
        run=state_name.rsplit('/checkpoint-',1)[0]
        meta=json.loads(remote._ssh('cat '+shlex.quote('/root/work/'+run+'/run_meta.json'),timeout=30))
        stamp=json.loads(remote._ssh('cat '+shlex.quote('/root/work/'+run+'/adapter/training_meta.json'),timeout=30))
        ds=stamp['dataset']; arm=plan['arms'][0]
        assert arm['arm']=='low' and ds['repo']==arm['data_repo'] and ds['revision']==arm['data_revision']
        assert meta['dataset']==ds and meta['world_size']==2 and meta['n_examples']==9968
        assert stamp['base_model']=='Qwen/Qwen3.6-27B' and stamp['base_model_revision']==plan['base_model_revision']
        cfg=stamp['train_config']; assert cfg['seed']==0 and stamp['thinking'] is True
        assert cfg['lora']['r']==64 and cfg['train']['epochs']==1 and cfg['train']['token_budget']==8000
        import math
        totals=[x for x in meta['log_history'] if 'train_loss' in x]
        assert totals[-1]['step']==623 and math.isfinite(totals[-1]['train_loss'])
        for row in meta['log_history']:
            assert all(math.isfinite(float(row[k])) for k in ('loss','grad_norm') if k in row)
        repo=hf_org()+'/'+stamp['organism']; info=hf_api().model_info(repo,files_metadata=True)
        assert not info.private and read(hf_download(repo,'training_meta.json',revision=info.sha))==stamp
        weights=[n for n in names if n.endswith('.safetensors')]; assert weights
        siblings={f.rfilename:f for f in info.siblings}; hashes={}
        for name in weights:
            filename=name.rsplit('/',1)[-1]
            digest=hashlib.file_digest(tar.extractfile(name),'sha256').hexdigest()
            assert siblings[filename].lfs.sha256==digest, 'Final checkpoint and public adapter differ'
            hashes[filename]=digest
    write_json(out/'low_early_training_meta.json',stamp);write_json(out/'low_early_run_meta.json',meta)
    # Existing fixed watchdog: 5.6h at the verified $9.28/h plus $2 latency margin.
    lifetime=min(int(58/10*3600),int((float(plan['gpu_budget_usd'])-2)/10*3600))
    training_max=lifetime/3600*float(training['budget_hourly_usd'])+2
    allocation=14; future_eval_reserve=14; margin=1
    exposure=float(plan['project_exposure_before_training_usd'])+training_max+allocation+future_eval_reserve+margin
    assert exposure<=float(plan['project_ceiling_usd'])
    frozen=out/'odcv_frozen.yaml'
    eval_plan=dict(target=repo,target_revision=info.sha,base_model=stamp['base_model'],
        base_revision=plan['base_model_revision'],output_dir=str(out/'evaluation_low'),
        run_name='odcv-stakes-low-20260910',eval_output_root='C:/nm-stakes',
        eval_config=str(frozen),eval_config_sha256=file_sha256(frozen),expected_cells=80,passes=3,
        gpu_cap_usd=10.5,backup_reserve_usd=2,judge_cap_usd=3.5,
        max_gpu_hourly_usd=4.5,storage_hourly_reserve_usd=0.1)
    path=out/'low_eval_plan.yaml'; assert not path.exists()
    OmegaConf.save(OmegaConf.create(eval_plan),path)
    from scratch.nonmoral.overnight_baseline import load_plan
    load_plan(path)
    claim=dict(model=dict(repo=repo,revision=info.sha),plan=str(path),plan_sha256=file_sha256(path),
        checkpoint_receipt=receipt,weight_sha256=hashes,training_max_usd=training_max,
        eval_allocation_usd=allocation,future_high_eval_reserve_usd=future_eval_reserve,
        worst_case_project_exposure_usd=exposure,
        authorization='User requested immediate low-stakes ODCV on a separate RunPod GPU; existing training unchanged.')
    write_json(out/'low_eval_handoff.json',claim)
    print(json.dumps({k:v for k,v in claim.items() if k!='checkpoint_receipt'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output_dir');prepare(parser.parse_args().output_dir)
