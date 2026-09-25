# ABOUTME: Fetches complete training outputs and logs before an owned pod is torn down.
# ABOUTME: Verifies transfer hashes and required adapter provenance; no credentials or model caches are archived.
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import time

from src.infra.endpoints.vllm import ssh_argv


def pack_script(root='/root/work', *, include_roots=None, archive_name='nonmoral-result-backup.tar', exclude_checkpoints=False):
    """Only the two owned output trees, including saved resume checkpoints."""
    roots=tuple(include_roots or ('output/train','output/nonmoral-paired-supervision'))
    for relative in roots:
        p=PurePosixPath(relative)
        if p.is_absolute() or '..' in p.parts or p.parts[:2] not in (
                ('output','train'),('output','nonmoral-paired-supervision')):
            raise ValueError('Backup root outside owned training output trees')
    if PurePosixPath(archive_name).name!=archive_name or not archive_name.endswith('.tar'):
        raise ValueError('Expected a plain archive filename')
    return f'''
import hashlib, json, tarfile
from pathlib import Path
root=Path({root!r})
archive=root/'output'/{archive_name!r}
files=[]
for relative in {roots!r}:
 directory=root/relative
 for p in sorted(directory.rglob('*')):
  if {exclude_checkpoints!r} and any(part.startswith('checkpoint-') for part in p.relative_to(root).parts):
   continue
  if p.is_symlink():
   raise RuntimeError('Refusing symlink in training outputs: '+str(p))
  if p.is_file():
   files.append(p)
if not files:
 raise RuntimeError('No training outputs or driver logs to preserve')
with tarfile.open(archive,'w') as tar:
 for p in files:
  before=p.stat()
  tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
  after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
   raise RuntimeError('Output changed while archiving: '+str(p))
with archive.open('rb') as f:
 digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(files))))
'''


def verify_archive(path, remote_manifest, expected_arms=()):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != remote_manifest['bytes'] or digest != remote_manifest['sha256']:
        raise ValueError('Local backup does not match remote archive size/hash')
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        names = {m.name for m in members}
        if len(names) != len(members) or len(members) != remote_manifest['files']:
            raise ValueError('Backup member count differs or contains duplicate paths')
        for m in members:
            p = PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.parts[:2] not in (
                    ('output', 'train'), ('output', 'nonmoral-paired-supervision')):
                raise ValueError('Unexpected path/type in result backup')
        datasets = set()
        for m in members:
            if m.name.endswith('/run_meta.json') and m.name.startswith('output/train/'):
                meta = json.load(tar.extractfile(m))
                run = str(PurePosixPath(m.name).parent)
                adapter = run + '/adapter/'
                required = ['adapter_config.json', 'training_meta.json', 'train_config.yaml',
                            'tokenizer_config.json']
                present = all(adapter + f in names and tar.getmember(adapter + f).size > 0 for f in required)
                weights = any(n.startswith(adapter) and n.endswith('.safetensors')
                              and tar.getmember(n).size > 0 for n in names)
                tokenizer = any(adapter + f in names for f in ('tokenizer.json', 'tokenizer.model'))
                if present and weights and tokenizer:
                    stamp = json.load(tar.extractfile(adapter + 'training_meta.json'))
                    if stamp['dataset'] != meta['dataset']:
                        raise ValueError('Adapter and run dataset provenance differ')
                    datasets.add((meta['dataset']['repo'], meta['dataset']['revision'],
                                  meta['base_model_revision']))
        for arm in expected_arms:
            identity = (arm['data_repo'], arm['data_revision'], arm['base_model_revision'])
            if identity not in datasets:
                raise ValueError('Completed arm missing local weights/tokenizer/config/provenance: '+arm['data_repo'])
    return dict(verified=True, archive=str(path.resolve()), sha256=digest,
                bytes=path.stat().st_size, files=len(members), verified_completed_arms=len(expected_arms))


def fetch_training_outputs(remote, out, expected_arms=(), timeout=600, *, include_roots=None,
                           archive_name='nonmoral-result-backup.tar', exclude_checkpoints=False):
    deadline = time.monotonic() + timeout
    # The base image's system Python may predate hashlib.file_digest (3.11).
    # Training has already installed the repository interpreter; reuse it.
    manifest = json.loads(remote._ssh('/root/work/.venv/bin/python -c ' + shlex.quote(
        pack_script(include_roots=include_roots,archive_name=archive_name,
                    exclude_checkpoints=exclude_checkpoints)), timeout=timeout))
    out = Path(out)
    if shutil.disk_usage(out).free < manifest['bytes'] + 1024**3:
        raise RuntimeError('Insufficient local disk space for training backup plus 1 GiB reserve')
    temporary = out / 'training_outputs.tar.partial'
    argv, target = ssh_argv(remote.host, remote.identity)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Training archive creation exhausted the recovery window')
    with temporary.open('wb') as stream:
        result = subprocess.run([*argv, target, 'cat ' + shlex.quote(manifest['path'])],
                                stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.PIPE, timeout=remaining)
    if result.returncode:
        raise RuntimeError('Training backup transfer failed; retain pod and partial download')
    receipt = verify_archive(temporary, manifest, expected_arms)
    final = out / 'training_outputs.tar'
    temporary.replace(final)
    receipt.update(archive=str(final.resolve()), remote_manifest=manifest)
    return receipt


def may_terminate_training(state):
    """A failed fetch must never fall through to ordinary teardown."""
    return (not state.get('training_started') or state.get('local_backup', {}).get('verified') is True
            or (state.get('publication', {}).get('verified') is True
                and state.get('hub_backup', {}).get('verified') is True))


def verify_publication(archive, expected_arms, *, steps, world_size, n_examples=10000):
    """Verify actual Hub payloads against preserved final adapters and training facts."""
    from src.infra.huggingface import hf_api, hf_org
    results = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.name.startswith('output/train/') or not member.name.endswith('/run_meta.json'):
                continue
            meta = json.load(tar.extractfile(member))
            identity = (meta['dataset']['repo'], meta['dataset']['revision'], meta['base_model_revision'])
            if identity not in [(a['data_repo'], a['data_revision'], a['base_model_revision']) for a in expected_arms]:
                raise ValueError('Unexpected completed training identity')
            assert meta['world_size'] == world_size and meta['n_examples'] == n_examples
            history = meta['log_history']
            assert history[-1]['step'] == steps and history[-1]['epoch'] == 1
            assert all(math.isfinite(float(v)) for h in history for k, v in h.items()
                       if k in ('loss', 'grad_norm', 'train_loss'))
            prefix = str(PurePosixPath(member.name).parent) + '/adapter/'
            stamp = json.load(tar.extractfile(prefix + 'training_meta.json'))
            assert stamp['dataset'] == meta['dataset'] and stamp['base_model_revision'] == meta['base_model_revision']
            assert stamp['thinking'] and stamp['supervise_counts'] == {'all': n_examples}
            info = hf_api().model_info(hf_org() + '/' + stamp['organism'], files_metadata=True)
            verified = []
            for remote in info.siblings:
                if remote.rfilename == '.gitattributes' or remote.rfilename.startswith('training_backup/'):
                    continue
                local = tar.getmember(prefix + remote.rfilename)
                assert local.size == remote.size
                sha = hashlib.sha256()
                blob = hashlib.sha1(f'blob {local.size}\0'.encode())
                with tar.extractfile(local) as stream:
                    while chunk := stream.read(4 * 1024 * 1024):
                        sha.update(chunk)
                        blob.update(chunk)
                assert (sha.hexdigest() == remote.lfs.sha256 if remote.lfs else blob.hexdigest() == remote.blob_id)
                verified.append({'file': remote.rfilename, 'bytes': local.size, 'sha256': sha.hexdigest()})
            assert any(f['file'] == 'adapter_model.safetensors' for f in verified)
            results.append({'repo': info.id, 'revision': info.sha, 'steps': steps, 'world_size': world_size,
                            'metrics': history[-1], 'dataset': meta['dataset'], 'files': verified})
    assert len(results) == len(expected_arms)
    return {'verified': True, 'arms': results}


if __name__=='__main__':
    import argparse
    import os
    from src.infra.endpoints.vllm import SshExec
    parser=argparse.ArgumentParser(description='Preserve one completed checkpoint while training continues')
    parser.add_argument('--host',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    checkpoint=PurePosixPath(args.checkpoint)
    if len(checkpoint.parts)!=4 or checkpoint.parts[:2]!=('output','train') or not checkpoint.name.startswith('checkpoint-') or not checkpoint.name.removeprefix('checkpoint-').isdigit():
        raise ValueError('Expected an exact owned training checkpoint directory')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    status=dict(phase='copying',pid=os.getpid(),checkpoint=str(checkpoint),host=args.host)
    def save():
        (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    save()
    try:
        remote=SshExec(args.host,port=8000,workdir='/root/work')
        archive_name='nonmoral-'+checkpoint.parts[2]+'-'+checkpoint.name+'-backup.tar'
        receipt=fetch_training_outputs(remote,out,timeout=1800,include_roots=[str(checkpoint)],archive_name=archive_name)
        status.update(phase='verified',receipt=receipt);save()
        remote._ssh('rm -f -- '+shlex.quote('/root/work/output/'+archive_name),timeout=30)
    except BaseException as exc:
        status.update(phase='failed',error=f'{type(exc).__name__}: {exc}');save();raise
