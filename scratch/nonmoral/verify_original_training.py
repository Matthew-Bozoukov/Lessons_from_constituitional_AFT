# ABOUTME: Verify the completed exact-original nonmoral adapter and freeze its authorized eval plan.
# ABOUTME: Run with uv run --no-sync python -m scratch.nonmoral.verify_original_training; no GPU is rented.
import hashlib
import json
import math
import tarfile
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf
from src.infra import runpod
from src.infra.huggingface import hf_api
from scratch.nonmoral.overnight_baseline import load_plan, checked_spec
from src.eval.docker import docker_preflight

ROOT = Path('output/nonmoral_original_reuse')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    load_dotenv()
    state = read(ROOT/'train_attempt1/status.json')
    assert state['phase'] == 'trained' and state['terminated'] is True
    assert state['completed_arms'] == [0] and state['local_backup']['verified']
    assert not any(p['id'] == state['owned_pod'] for p in runpod.active_pods())
    meta = read(ROOT/'train_attempt1/2026-09-15_qwen36_0_nonmoral_original_7_run_meta.json')
    assert meta['world_size'] == 1 and meta['n_examples'] == 9968
    assert meta['log_history'][-1]['step'] == 623 and meta['log_history'][-1]['epoch'] == 1
    assert all(math.isfinite(float(e[k])) for e in meta['log_history'] for k in ('loss','grad_norm') if k in e)
    cfg = OmegaConf.load(ROOT/'eval_plan_template.yaml')
    assert meta['base_model_revision'] == cfg.base_revision
    assert meta['dataset']['revision'] == 'b35ead8eaf4d59091e0ef1d08b187c7822f05632'
    info = hf_api().model_info(cfg.target, files_metadata=True)
    prefix = 'output/train/2026-09-15_qwen36_0_nonmoral_original_7/adapter/'
    verified = []
    with tarfile.open(state['local_backup']['archive']) as tar:
        for remote in info.siblings:
            if remote.rfilename == '.gitattributes':
                continue
            member = tar.getmember(prefix+remote.rfilename)
            assert member.size == remote.size
            with tar.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if remote.lfs:
                assert digest == remote.lfs.sha256, remote.rfilename
            else:
                raw = tar.extractfile(member).read()
                assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == remote.blob_id, remote.rfilename
            verified.append({'file':remote.rfilename, 'bytes':member.size, 'sha256':digest})
    cfg.target_revision = info.sha
    assert state['estimated_gpu_usd'] <= 45
    assert state['estimated_gpu_usd'] + cfg.gpu_cap_usd + cfg.judge_cap_usd <= 60
    plan_path = ROOT/'eval_plan.yaml'
    OmegaConf.save(cfg, plan_path)
    plan = load_plan(plan_path)
    checked_spec(plan)
    docker_preflight()
    receipt = {'repo':cfg.target, 'revision':info.sha, 'training_complete':True,
               'steps':623, 'epoch':1, 'train_loss':meta['log_history'][-1]['train_loss'],
               'train_runtime_s':meta['log_history'][-1]['train_runtime'],
               'training_gpu_storage_estimate_usd':state['estimated_gpu_usd'],
               'training_pod_termination_verified':True, 'files':verified,
               'plan_preflight':'passed', 'docker_preflight':'passed'}
    (ROOT/'training_verified.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))

if __name__ == '__main__':
    main()
