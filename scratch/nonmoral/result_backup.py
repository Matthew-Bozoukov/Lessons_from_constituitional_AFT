# ABOUTME: Fetches complete training outputs and logs before an owned pod is torn down.
# ABOUTME: Verifies transfer hashes and required adapter provenance; no credentials or model caches are archived.
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import time

from src.infra.endpoints.vllm import ssh_argv


def pack_script(root='/root/work'):
    """Only the two owned output trees, including saved resume checkpoints."""
    return f'''
import hashlib, json, tarfile
from pathlib import Path
root=Path({root!r})
archive=root/'output/nonmoral-result-backup.tar'
files=[]
for relative in ('output/train', 'output/nonmoral-paired-supervision'):
 directory=root/relative
 for p in sorted(directory.rglob('*')):
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


def fetch_training_outputs(remote, out, expected_arms=(), timeout=600):
    deadline = time.monotonic() + timeout
    # The base image's system Python may predate hashlib.file_digest (3.11).
    # Training has already installed the repository interpreter; reuse it.
    manifest = json.loads(remote._ssh('/root/work/.venv/bin/python -c ' + shlex.quote(pack_script()), timeout=timeout))
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
                                stdout=stream, stderr=subprocess.PIPE, timeout=remaining)
    if result.returncode:
        raise RuntimeError('Training backup transfer failed; retain pod and partial download')
    receipt = verify_archive(temporary, manifest, expected_arms)
    final = out / 'training_outputs.tar'
    temporary.replace(final)
    receipt.update(archive=str(final.resolve()), remote_manifest=manifest)
    return receipt


def may_terminate_training(state):
    """A failed fetch must never fall through to ordinary teardown."""
    return not state.get('training_started') or state.get('local_backup', {}).get('verified') is True
