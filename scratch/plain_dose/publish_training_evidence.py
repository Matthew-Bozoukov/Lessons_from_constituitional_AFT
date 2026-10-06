# ABOUTME: Verifies final adapter bytes on the owned training pod against Hugging Face and publishes small training evidence.
# ABOUTME: Retains full losses and logs without making inference downloads include optimizer archives; performs no teardown.
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from huggingface_hub import CommitOperationAdd
from src.infra.endpoints.vllm import SshExec
from src.infra.huggingface import hf_api
from src.naming import to_local
from scratch.plain_dose.preserve_evidence import collect


def publish(owner):
    owner = Path(owner).resolve()
    assert owner.is_relative_to(ROOT / 'output/plain_dose_campaign')
    receipt_path = owner / 'publication_verified.json'
    assert not receipt_path.exists(), 'Publication already verified; reuse its immutable revision'
    state = json.loads((owner / 'status.json').read_text())
    assert state.get('training_complete') and not state.get('terminated')
    arm = state['arm']
    repo = arm['organism']
    run_name = to_local(repo.split('/', 1)[1])
    collect(owner / 'status.json')
    local = owner / 'publication_evidence/remote/output/train' / run_name
    meta = json.loads((local / 'run_meta.json').read_text())
    expected_steps, expected_rows = {'da5': (123, 1960), 'da25': (154, 2458)}[arm['key']]
    assert meta['n_examples'] == expected_rows and meta['world_size'] == 1 and not meta['smoke']
    assert meta['dataset']['repo'] == arm['data_repo'] and meta['dataset']['revision'] == arm['data_revision']
    assert meta['base_model_revision'] == state['plan']['base_model_revision']
    assert meta['log_history'][-1]['step'] == expected_steps and meta['log_history'][-1]['epoch'] == 1
    assert all(math.isfinite(float(v)) for row in meta['log_history'] for k, v in row.items()
               if k in ('loss', 'grad_norm', 'train_loss'))
    final_state = local / f'checkpoint-{expected_steps}/trainer_state.json'
    assert json.loads(final_state.read_text())['global_step'] == expected_steps
    cfg = meta['config']
    assert cfg['seed'] == 0 and cfg['train']['loss_agg'] == 'token_mean'
    assert cfg['train']['batch_size'] * cfg['train']['grad_accum'] == 16
    assert cfg['lora'] == {'r': 64, 'alpha': 128, 'dropout': 0.05}
    remote = SshExec(state['host'], port=8000, workdir='/root/work')
    probe = '''import hashlib,json
from pathlib import Path
root=Path('/root/work/output/train')/RUN/'adapter'
rows={}
for p in root.rglob('*'):
 if not p.is_file():continue
 assert not p.is_symlink()
 size=p.stat().st_size;sha=hashlib.sha256();git=hashlib.sha1(f'blob {size}\\0'.encode())
 with p.open('rb') as f:
  while chunk:=f.read(4*1024*1024):sha.update(chunk);git.update(chunk)
 rows[p.relative_to(root).as_posix()]={'bytes':size,'sha256':sha.hexdigest(),'git_blob':git.hexdigest()}
print(json.dumps(rows))
'''.replace('RUN', repr(run_name))
    hashes = json.loads(remote._ssh('python3 -c ' + shlex.quote(probe), timeout=120))
    api = hf_api()
    initial = api.model_info(repo, files_metadata=True)
    assert initial.sha == state['adapter_revision'], 'Adapter head changed after training publication'
    for f in initial.siblings:
        if f.rfilename == '.gitattributes':
            continue
        row = hashes[f.rfilename]
        assert row['bytes'] == f.size
        assert (row['sha256'] == f.lfs.sha256 if f.lfs else row['git_blob'] == f.blob_id), f.rfilename
    assert 'adapter_model.safetensors' in hashes
    stage = owner / 'published_training_evidence'
    stage.mkdir(exist_ok=True)
    shutil.copy2(local / 'run_meta.json', stage / 'run_meta.json')
    shutil.copy2(final_state, stage / 'trainer_state.json')
    log_root = owner / 'publication_evidence/remote/output/da-supervision'
    for name in ('smoke.log', 'train.log'):
        shutil.copy2(log_root / name, stage / name)
    for source in (ROOT / 'output/plain_dose/independent_mixture_audit.json',
                   ROOT / 'output/plain_dose_campaign/matched_protocol.json'):
        shutil.copy2(source, stage / source.name)
    verification = {'verified': True, 'repo': repo, 'adapter_revision_before_evidence': initial.sha,
                    'remote_adapter_files': hashes, 'steps': expected_steps, 'rows': expected_rows,
                    'final_metrics': meta['log_history'][-1], 'dataset': meta['dataset'],
                    'base_model_revision': meta['base_model_revision'], 'training_commit': meta['git_sha']}
    (stage / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    names = ('run_meta.json', 'trainer_state.json', 'smoke.log', 'train.log',
             'independent_mixture_audit.json', 'matched_protocol.json', 'verification.json')
    files = [stage / name for name in names]
    assert all(p.is_file() and not p.is_symlink() and p.stat().st_size < 10_000_000 for p in files)
    commit = api.create_commit(repo_id=repo, repo_type='model', parent_commit=initial.sha,
        commit_message='Preserve verified training losses, final state, logs and matched protocol',
        operations=[CommitOperationAdd(path_in_repo='training_evidence/' + p.name, path_or_fileobj=p) for p in files])
    published = api.model_info(repo, revision=commit.oid, files_metadata=True)
    listing = {f.rfilename: f for f in published.siblings}
    for p in files:
        f = listing['training_evidence/' + p.name]
        data = p.read_bytes()
        assert len(data) == f.size
        digest = hashlib.sha256(data).hexdigest() if f.lfs else hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        assert digest == (f.lfs.sha256 if f.lfs else f.blob_id), p.name
    for name in (f.rfilename for f in initial.siblings if f.rfilename != '.gitattributes'):
        row = hashes[name]
        f = listing[name]
        assert f.size == row['bytes'] and (f.lfs.sha256 == row['sha256'] if f.lfs else f.blob_id == row['git_blob'])
    verification.update(revision=commit.oid, evidence_files=[p.name for p in files])
    receipt_path.write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'verified': True, 'repo': repo, 'revision': commit.oid, 'steps': expected_steps}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('owner')
    publish(parser.parse_args().owner)
