# ABOUTME: Checks the real artifact gate with tiny local adapters and mocked HF metadata.
# ABOUTME: Rejects incomplete training, private publication and wrong dataset identity before eval rental.
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
import pytest
from scratch.nonmoral import stakes_experiment as experiment


@pytest.mark.parametrize('defect',[None,'private','steps','revision'])
def test_verified_pair_required_before_evaluation(tmp_path,monkeypatch,defect):
    base='b'*40;plan=dict(base_model_revision=base,arms=[])
    stamps={};path=tmp_path/'training.tar';weights=b'verified adapter'
    with tarfile.open(path,'w') as tar:
        for arm in ('low','high'):
            repo='test/'+arm;plan['arms'].append(dict(arm=arm,data_repo=repo,data_revision='a'*40))
            cfg=dict(seed=0,lora=dict(r=64),train=dict(epochs=1,token_budget=8000))
            ds=dict(repo=repo,revision='c'*40 if defect=='revision' else 'a'*40)
            stamp=dict(dataset=ds,base_model='Qwen/Qwen3.6-27B',base_model_revision=base,
                       thinking=True,train_config=cfg,organism=arm)
            stamps['dougalldeepmind/'+arm]=stamp
            meta=dict(world_size=2,n_examples=9968,config=cfg,log_history=[dict(train_loss=.5,step=622 if defect=='steps' else 623)])
            for name,value in {'run_meta.json':json.dumps(meta).encode(),
                   'adapter/training_meta.json':json.dumps(stamp).encode(),
                   'adapter/adapter_model.safetensors':weights}.items():
                info=tarfile.TarInfo('output/train/'+arm+'/'+name);info.size=len(value);tar.addfile(info,io.BytesIO(value))
    training=dict(terminated=True,completed_arms=[0,1],local_backup=dict(verified=True,
        verified_completed_arms=2,archive=str(path),bytes=path.stat().st_size))
    def download(repo,file,revision):
        p=tmp_path/(repo.split('/')[-1]+'.json');p.write_text(json.dumps(stamps[repo]));return str(p)
    monkeypatch.setattr(experiment,'hf_org',lambda:'dougalldeepmind')
    monkeypatch.setattr(experiment,'hf_download',download)
    monkeypatch.setattr(experiment,'hf_api',lambda:SimpleNamespace(model_info=lambda *a,**kw:SimpleNamespace(
        private=defect=='private',sha='d'*40,siblings=[SimpleNamespace(rfilename='adapter_model.safetensors',
        lfs=SimpleNamespace(sha256=hashlib.sha256(weights).hexdigest()))])))
    if defect:
        with pytest.raises(AssertionError):experiment.verify_models(training,plan,tmp_path)
    else:
        result=experiment.verify_models(training,plan,tmp_path)
        assert set(result)=={'low','high'} and all(r['local_weights_verified'] for r in result.values())


@pytest.mark.parametrize('state',['active','complete','failed','wrong_pin','recovered','recovered_wrong_pin'])
def test_adopt_existing_eval_without_relaunch(tmp_path,state):
    from omegaconf import OmegaConf
    import time
    model=dict(repo='dougalldeepmind/low',revision='a'*40)
    plan=dict(target=model['repo'],target_revision=model['revision'],eval_config_sha256='frozen',output_dir=str(tmp_path/'eval'))
    path=tmp_path/'plan.yaml';OmegaConf.save(OmegaConf.create(plan),path)
    (tmp_path/'eval').mkdir()
    claim=dict(model=model,plan=str(path),plan_sha256=experiment.file_sha256(path))
    (tmp_path/'low_eval_handoff.json').write_text(json.dumps(claim))
    status=dict(target=model['repo'],target_revision='b'*40 if state=='wrong_pin' else model['revision'],
        termination_verified=state in ('complete','failed','recovered','recovered_wrong_pin'),updated_at_unix=time.time(),
        evaluation_driver_completed=state=='complete',local_log_backup=dict(verified=True),
        estimated_gpu_and_storage_usd=4,judge_charged_or_reserved_usd=1)
    (tmp_path/'eval/broader_eval_status.json').write_text(json.dumps(status))
    if state.startswith('recovered'):
        recovery=dict(completed=True,public=True,existing_verdicts_unchanged=True,
            target=model['repo'],target_revision='b'*40 if state=='recovered_wrong_pin' else model['revision'],
            plan_sha256=claim['plan_sha256'],rollouts_rerun=0,judge_charged_or_reserved_usd=1)
        (tmp_path/'low_eval_completion_recovery.json').write_text(json.dumps(recovery))
    if state in ('failed','wrong_pin','recovered_wrong_pin'):
        with pytest.raises(AssertionError):experiment.adopted_eval_status(tmp_path,'low',model,'frozen')
    else:
        result=experiment.adopted_eval_status(tmp_path,'low',model,'frozen')
        assert result is None if state=='active' else result['cost_usd']==5
