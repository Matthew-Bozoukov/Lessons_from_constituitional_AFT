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
