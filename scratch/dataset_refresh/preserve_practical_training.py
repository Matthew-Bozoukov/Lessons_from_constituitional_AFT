# ABOUTME: Preserve the verified completed training archive and receipts in the final model repository.
# ABOUTME: Run after the training owner finishes: uv run --no-sync python -m scratch.dataset_refresh.preserve_practical_training
import hashlib
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd

from src.infra.huggingface import hf_api


def main():
    load_dotenv('.env')
    root = Path('output/lowstakes_practical_training')
    state = json.loads((root/'run/status.json').read_text(encoding='utf-8'))
    assert state['phase'] == 'trained' and state['terminated'] is True
    assert state['publication']['verified'] and state['local_backup']['verified']
    arm = state['publication']['arms'][0]
    assert arm['steps'] == 625 and arm['world_size'] == 1
    archive = Path(state['local_backup']['archive'])
    with archive.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert sha == state['local_backup']['sha256']
    api = hf_api()
    info = api.model_info(arm['repo'], files_metadata=True)
    key = 'training_backup/training_outputs.tar'
    existing = next((f for f in info.siblings if f.rfilename == key), None)
    manifest = dict(bytes=archive.stat().st_size, sha256=sha, training_steps=625,
        dataset=arm['dataset'], adapter_revision_verified=arm['revision'],
        owned_pod=state['owned_pod'], terminated=True,
        estimated_gpu_usd=state['estimated_gpu_usd'])
    if existing:
        assert existing.size == manifest['bytes'] and existing.lfs.sha256 == sha
        revision = info.sha
    else:
        # Publish selected receipts only. Account balances, credentials and other pod inventory stay local.
        ops = [CommitOperationAdd(path_in_repo=key,path_or_fileobj=archive),
            CommitOperationAdd(path_in_repo='training_backup/manifest.json',
                path_or_fileobj=json.dumps(manifest,indent=2).encode())]
        for source, target in [(root/'run/publication.json','adapter_verification.json'),
            (root/'train_plan.json','launch_plan.json'),
            (root/'mixture_receipt.json','mixture_receipt.json')]:
            ops.append(CommitOperationAdd(path_in_repo='training_backup/'+target,path_or_fileobj=source))
        revision = api.create_commit(repo_id=arm['repo'],repo_type='model',parent_commit=info.sha,
            commit_message='Preserve verified training outputs, checkpoints and provenance',operations=ops).oid
    remote = api.model_info(arm['repo'],revision=revision,files_metadata=True)
    file = next(f for f in remote.siblings if f.rfilename == key)
    assert file.size == manifest['bytes'] and file.lfs.sha256 == sha
    receipt = dict(verified=True,repo=arm['repo'],revision=revision,archive=manifest)
    (root/'final_release_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    if '--wait' in sys.argv:
        deadline = time.monotonic() + 9 * 3600
        while True:
            state = json.loads(Path('output/lowstakes_practical_training/run/status.json').read_text(encoding='utf-8'))
            if state.get('terminated'):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Owner did not finish within its bounded training/recovery window')
            time.sleep(30)
    main()
