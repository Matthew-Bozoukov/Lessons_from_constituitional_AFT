# ABOUTME: Recovers verified final models after externally observed training-pod disappearance.
# ABOUTME: Preserves partial-transfer evidence and explicitly records missing final optimizer checkpoints.
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import time

from src.infra import runpod
from src.infra.huggingface import hf_api,hf_download,hf_org
from scratch.nonmoral.stakes import write_json,file_sha256
from scratch.nonmoral.stakes_experiment import read,verify_models
from scratch.nonmoral.result_backup import verify_archive


def recover(out):
    out=Path(out).resolve(); original=read(out/'training/status.json'); plan=read(out/'train_plan.json')
    assert original['completed_arms']==[0,1]
    assert not any(p['id']==original['owned_pod'] for p in runpod.active_pods()), 'Training pod still exists'
    dest=out/'recovered_training';dest.mkdir(exist_ok=False)
    partial=out/'training/training_outputs.tar.partial';size=partial.stat().st_size
    evidence=dict(pod=original['owned_pod'],observed_absent_epoch=time.time(),partial_bytes=size,
        partial_sha256=file_sha256(partial),cause='Unknown: both local watchdogs reported pod already gone; API returned404.',
        full_training_archive_complete=False,missing='High final optimizer/RNG/scheduler checkpoint and the untransferred full archive; older verified checkpoints retained.')
    write_json(dest/'incident.json',evidence)
    files={}
    # Read only complete ordinary members. The interrupted archive itself is never modified.
    with tarfile.open(partial) as tar:
        try:
            for member in tar:
                if member.offset_data+member.size>size:break
                path=PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                if '/adapter/' not in member.name:continue
                target=dest/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
                files[member.name]='complete member of interrupted remote archive'
        except tarfile.ReadError:pass
    high_stamp_path=next(dest.glob('output/train/*/adapter/training_meta.json'))
    stamps={'high':read(high_stamp_path),'low':read(out/'low_early_training_meta.json')}
    low_receipt=read(out/'checkpoint_copies/low_623/status.json')['receipt']
    assert low_receipt['verified'] and file_sha256(low_receipt['archive'])==low_receipt['sha256']
    low_run=None
    with tarfile.open(low_receipt['archive']) as tar:
        for member in tar:
            if not member.name.endswith('.safetensors'):continue
            low_run=member.name.rsplit('/checkpoint-',1)[0]
            name=low_run+'/adapter/'+PurePosixPath(member.name).name
            target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
            files[name]='hash-verified final low checkpoint623'
    assert low_run is not None
    model_pins={}
    for arm,stamp in stamps.items():
        repo=hf_org()+'/'+stamp['organism'];info=hf_api().model_info(repo,files_metadata=True)
        assert not info.private
        model_pins[arm]=dict(repo=repo,revision=info.sha)
        run=low_run if arm=='low' else high_stamp_path.parent.parent.relative_to(dest).as_posix()
        adapter=dest/run/'adapter';adapter.mkdir(exist_ok=True,parents=True)
        for sibling in info.siblings:
            name=sibling.rfilename
            if '/' in name or name=='.gitattributes':continue
            target=adapter/name
            if not target.exists():
                assert not name.endswith('.safetensors'), 'Required independent local final weights absent'
                shutil.copyfile(hf_download(repo,name,revision=info.sha),target)
                files[target.relative_to(dest).as_posix()]='pinned public HF artifact'
            if name.endswith('.safetensors'):
                assert file_sha256(target)==sibling.lfs.sha256
        assert read(adapter/'training_meta.json')==stamp
        local_meta=out/'training'/(PurePosixPath(run).name+'_run_meta.json')
        shutil.copyfile(local_meta,dest/run/'run_meta.json')
        files[run+'/run_meta.json']='training owner metadata fetched before source loss'
    archive=dest/'training_outputs.tar'
    with tarfile.open(archive,'w') as tar:
        for name in sorted(files):tar.add(dest/name,arcname=name,recursive=False)
    manifest=dict(bytes=archive.stat().st_size,sha256=file_sha256(archive),files=len(files))
    expected=[dict(a,base_model_revision=plan['base_model_revision']) for a in plan['arms']]
    receipt=verify_archive(archive,manifest,expected)
    receipt['scope']='Final adapter weights, tokenizer/configs, full training metrics; reconstructed from local evidence and pinned HF. Not the complete optimizer-checkpoint archive.'
    state=dict(original,terminated=True,termination_reason='Externally observed pod absence; no local teardown requested',
        estimated_gpu_usd=(evidence['observed_absent_epoch']-original['created_epoch'])/3600*original['budget_hourly_usd'],
        local_backup=receipt,recovery_evidence=str(dest/'incident.json'),full_training_archive_complete=False)
    verify_models(state,plan,out)
    write_json(dest/'file_sources.json',files);write_json(dest/'model_pins.json',model_pins)
    write_json(out/'recovered_training.json',state)
    print(json.dumps(dict(recovered_models=model_pins,bytes=receipt['bytes'],sha256=receipt['sha256'],full_archive_complete=False)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output_dir');recover(parser.parse_args().output_dir)
