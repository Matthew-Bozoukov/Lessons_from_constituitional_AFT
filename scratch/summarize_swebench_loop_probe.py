# ABOUTME: Summarize and publish immutable evidence from the single-GPU looping investigation.
# ABOUTME: Separates prefix-level generation recovery from whole-task correctness and full benchmark scores.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics

# Set before importing HF: parallel first-use symlink probes race on Windows.
if os.name == 'nt':
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')

from scratch.swebench_loop_probe import config, save


def summarize(root):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'responses').glob('*/*/*/summary.json'))]
    groups={}
    for row in rows:groups.setdefault((row['phase'],row['condition']),[]).append(row)
    result=[]
    for (phase,condition),group in groups.items():
        result.append(dict(phase=phase,condition=condition,n=len(group),loops=sum(x['loop'] for x in group),
            valid_bash=sum(x['valid_bash'] for x in group),length_capped=sum(x['finish_reason']=='length' for x in group),
            completion_tokens=sum(x['usage']['completion_tokens'] for x in group),
            median_completion_tokens=statistics.median(x['usage']['completion_tokens'] for x in group),
            median_seconds=statistics.median(x['seconds'] for x in group)))
    save(root/'results.json',{'groups':result,'cases':rows,'scope':'Diagnostic next-response replay at frozen historical failure prefixes; not full-task grading or an unbiased benchmark score.'})
    print(json.dumps(result,indent=2))


def publish(root, revision=None):
    from huggingface_hub import CommitOperationAdd
    from src.infra.huggingface import hf_api, hf_download, hf_repo_id
    assert (root/'report.md').exists()
    assert (root/'cleanup.json').exists()
    assert not set(json.loads((root/'cleanup.json').read_text())['owned_ids']) & set(json.loads((root/'cleanup.json').read_text())['remaining_ids'])
    repo=hf_repo_id('2026-09-24-swebench-lite-infrastructure')
    prefix='metadata/audits/2026-09-24-loop-probe'
    # Preserve full operational logs locally; do not publish shared-account balances,
    # unrelated renters' inventory, or connection details in the research evidence.
    private={'hf-verification.json','artifact-hashes.json','balance-before.json',
             'inventory-before.json','connection.json','cleanup.json','driver.log'}
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in private
           and '__pycache__' not in p.parts]
    assert not any(p.name.startswith('.env') or p.suffix in ('.pem','.key') for p in files)
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(root/'artifact-hashes.json',hashes);files.append(root/'artifact-hashes.json')
    api=hf_api();assert api.repo_exists(repo,repo_type='dataset')
    if revision is None:
        commit=api.create_commit(repo_id=repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=prefix+'/'+p.relative_to(root).as_posix(),path_or_fileobj=str(p)) for p in files],commit_message='Document controlled single-GPU replay of ten DA-5 generation loops')
        revision=commit.oid
    print(f'Verifying {len(files)} files at {revision}',flush=True)
    def verify(path):
        remote=Path(hf_download(repo,prefix+'/'+path.relative_to(root).as_posix(),repo_type='dataset',revision=revision))
        assert hashlib.sha256(remote.read_bytes()).digest()==hashlib.sha256(path.read_bytes()).digest(),path
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify,files))
    result=dict(repo=repo,revision=revision,prefix=prefix,files_hash_verified=len(files))
    save(root/'hf-verification.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true')
    parser.add_argument('--verify-revision',help='Resume readback of an existing commit without uploading again')
    args=parser.parse_args()
    _,root=config();summarize(root)
    if args.publish or args.verify_revision:publish(root,args.verify_revision)
