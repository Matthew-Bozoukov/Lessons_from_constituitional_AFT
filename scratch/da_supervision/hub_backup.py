# ABOUTME: Publishes an immutable full training-output archive directly from its owned pod.
# ABOUTME: Avoids slow laptop transfer and verifies Hub size and SHA256 before recording success.
import argparse,hashlib,json,time
from pathlib import Path
from huggingface_hub import CommitOperationAdd
from src.infra.huggingface import hf_api
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--archive',required=True);p.add_argument('--out',required=True);a=p.parse_args()
archive=Path(a.archive).resolve();assert archive==Path('/root/work/output/da-supervision-backup.tar') and archive.is_file()
out=Path(a.out);assert out.parent==Path('/root/work/output/da-supervision');assert not out.exists()
api=hf_api();info=api.model_info(a.repo);assert any(f.rfilename=='training_meta.json' for f in info.siblings)
with archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
manifest={'source_archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'adapter_revision_before_backup':info.sha,'created_epoch':time.time()}
key='training_backup/da-supervision-backup.tar'
print(json.dumps({'phase':'uploading','bytes':manifest['bytes'],'repo':a.repo}),flush=True)
commit=api.create_commit(repo_id=a.repo,repo_type='model',commit_message='Preserve full training outputs and resume checkpoints',parent_commit=info.sha,operations=[CommitOperationAdd(path_in_repo=key,path_or_fileobj=archive),CommitOperationAdd(path_in_repo='training_backup/manifest.json',path_or_fileobj=(json.dumps(manifest,indent=2)+'\n').encode())])
remote=api.model_info(a.repo,revision=commit.oid,files_metadata=True);file=next(f for f in remote.siblings if f.rfilename==key)
assert file.size==manifest['bytes'] and file.lfs.sha256==sha
receipt={'verified':True,'repo':a.repo,'revision':commit.oid,'path':key,**manifest}
out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
