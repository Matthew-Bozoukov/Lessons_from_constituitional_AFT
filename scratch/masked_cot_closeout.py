# ABOUTME: Preserve and verify the completed October 5 masked-CoT adapter before teardown.
# ABOUTME: Uses the existing training backup helper and publishes exact validation evidence.
import hashlib
import json
import math
import tarfile
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd
from scratch.nonmoral.result_backup import fetch_training_outputs, verify_archive
from src.infra.endpoints.vllm import SshExec
from src.infra.huggingface import hf_api

out = Path('output/masked_cot_campaign')
state = json.loads((out / 'campaign.json').read_text())
remote = SshExec(state['host'], port=8000, workdir='/root/work')
assert remote._ssh('cat /root/work/output/masked_cot_campaign/training.exit', timeout=30).strip() == '0'
base_revision = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
arm = {'data_repo': state['dataset']['repo'], 'data_revision': state['dataset']['revision'],
       'base_model_revision': base_revision}
receipt_path = out / 'local_backup.json'
if receipt_path.exists():
    receipt = json.loads(receipt_path.read_text())
    verify_archive(receipt['archive'], receipt['remote_manifest'], expected_arms=[arm])
else:
    receipt = fetch_training_outputs(remote, out, expected_arms=[arm], timeout=1800,
                                     include_roots=('output/train',),
                                     archive_name='masked-cot-result-backup.tar',
                                     exclude_checkpoints=True)
    receipt_path.write_text(json.dumps(receipt, indent=2))
print(json.dumps({'phase': 'local_backup_verified', 'bytes': receipt['bytes']}), flush=True)
log = remote._ssh('cat /root/work/output/masked_cot_campaign/training.log', timeout=60)
(out / 'training.log').write_text(log, encoding='utf-8')
api = hf_api()
info = api.model_info(state['model_expected'], files_metadata=True)
required = {'adapter_model.safetensors', 'adapter_config.json', 'training_meta.json',
            'train_config.yaml', 'tokenizer.json', 'tokenizer_config.json', 'README.md'}
assert required <= {f.rfilename for f in info.siblings}
mask_audit_path = Path('output/mixture_da_answer_only/20261005_093442/mask_audit.json')
mask_audit = json.loads(mask_audit_path.read_text())
assert mask_audit['mixture_sha256'] == state['dataset']['sha256']
assert mask_audit['DA_rows_checked'] == mask_audit['independent_answer_only_decode_checks'] == 1283
assert mask_audit['replay_rows_identical_to_oct3_answeronly'] == 8434
assert mask_audit['full_forward_token_ids_equal_to_all_supervision']
assert mask_audit['da_supervised_tokens'] == 718887
arrow_audit = json.loads((out / 'arrow_render_audit.json').read_text())
assert arrow_audit['render_changed_rows'] == 17 and arrow_audit['supervised_delta'] == 64
assert arrow_audit['sources'] == ['apigen_function_calling']
assert '4,849,975/8,776,403 tokens supervised' in log
with tarfile.open(receipt['archive']) as tar:
    runs = [m for m in tar.getmembers() if m.name.endswith('/run_meta.json')]
    assert len(runs) == 1
    meta = json.load(tar.extractfile(runs[0]))
    prefix = str(PurePosixPath(runs[0].name).parent) + '/adapter/'
    stamp = json.load(tar.extractfile(prefix + 'training_meta.json'))
    assert meta['n_examples'] == 9717 and meta['world_size'] == 1 and not meta['smoke']
    assert meta['dataset']['repo'] == arm['data_repo']
    assert meta['dataset']['revision'] == arm['data_revision']
    assert meta['base_model_revision'] == base_revision
    assert meta['git_sha'].startswith(state['source_commit'])
    assert stamp['dataset'] == meta['dataset'] and stamp['base_model_revision'] == base_revision
    assert stamp['thinking'] and stamp['supervise_counts'] == {'all': 8434, 'answer': 1283}
    assert meta['config']['train']['loss_agg'] == 'token_mean'
    history = meta['log_history']
    assert history[-1]['step'] == 608 and history[-1]['epoch'] == 1
    assert all(math.isfinite(float(v)) for row in history for k, v in row.items()
               if k in ('loss', 'grad_norm', 'train_loss'))
    verified = []
    for file in info.siblings:
        if file.rfilename == '.gitattributes' or file.rfilename.startswith('training_evidence/'):
            continue
        member = tar.getmember(prefix + file.rfilename)
        assert member.size == file.size
        sha = hashlib.sha256()
        blob = hashlib.sha1(f'blob {member.size}\0'.encode())
        with tar.extractfile(member) as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                sha.update(chunk)
                blob.update(chunk)
        assert sha.hexdigest() == file.lfs.sha256 if file.lfs else blob.hexdigest() == file.blob_id
        verified.append({'file': file.rfilename, 'bytes': member.size, 'sha256': sha.hexdigest()})
    assert any(f['file'] == 'adapter_model.safetensors' for f in verified)
(out / 'run_meta.json').write_text(json.dumps(meta, indent=2))
validation = {'verified': True, 'adapter_revision': info.sha, 'dataset': meta['dataset'],
              'base_model_revision': base_revision, 'steps': 608, 'epochs': 1,
              'supervise_counts': stamp['supervise_counts'], 'metrics': history[-1],
              'finite_loss_and_gradients': True, 'files': verified,
              'ablation': 'DA reasoning remains in input; only DA answer tokens receive loss',
              'approved_realized_DA_supervised_token_percent': 14.82,
              'builder_supervised_tokens': 4849911, 'trainer_supervised_tokens': 4849975,
              'render_count_note': 'HF schema normalization changes rendering of 17 apigen replay rows; DA rows unaffected.'}
(out / 'validation.json').write_text(json.dumps(validation, indent=2))
files = {'run_meta.json': out / 'run_meta.json', 'validation.json': out / 'validation.json',
         'training.log': out / 'training.log', 'arrow_render_audit.json': out / 'arrow_render_audit.json',
         'mask_audit.json': mask_audit_path}
commit = api.create_commit(repo_id=info.id, repo_type='model', parent_commit=info.sha,
    commit_message='Preserve training metrics and independently verified masking evidence',
    operations=[CommitOperationAdd(path_in_repo='training_evidence/' + key, path_or_fileobj=value)
                for key, value in files.items()])
published = api.model_info(info.id, revision=commit.oid, files_metadata=True)
for key, path in files.items():
    file = next(f for f in published.siblings if f.rfilename == 'training_evidence/' + key)
    content = path.read_bytes()
    assert file.size == len(content)
    assert (hashlib.sha256(content).hexdigest() == file.lfs.sha256 if file.lfs else
            hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest() == file.blob_id)
state.update(status='published_verified', publication={'repo': info.id, 'revision': commit.oid,
                                                     'verified': True}, local_backup=receipt)
(out / 'campaign.json').write_text(json.dumps(state, indent=2))
print(json.dumps({'phase': 'published_verified', 'repo': info.id, 'revision': commit.oid,
                  'metrics': history[-1]}), flush=True)
