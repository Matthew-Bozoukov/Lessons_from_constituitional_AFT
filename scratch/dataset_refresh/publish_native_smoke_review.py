# ABOUTME: Archives the completed native smoke and verified old-corpus comparison without model calls.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.publish_native_smoke_review --config scratch/dataset_refresh/publish_native_smoke_review.yaml
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from huggingface_hub import CommitOperationAdd, hf_hub_download
from omegaconf import OmegaConf
from src.infra.huggingface import hf_api, hf_token
from src.utils import git_sha, timestamp
from scratch.dataset_refresh.run import write_json, write_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args=parser.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    root=Path(cfg['run_root'])
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert not manifest['aborted'] and manifest['counts']['export_sft']==10
    repo=manifest['hf_repo']
    api=hf_api()
    pinned=hf_hub_download(cfg['old_repo'],'dataset.jsonl',repo_type='dataset',revision=cfg['old_revision'],token=hf_token())
    assert sha(pinned)==cfg['old_sha256']
    originals=[json.loads(s) for s in Path(pinned).read_text(encoding='utf-8').splitlines()]
    by_id={x['metadata']['scenario_id']:x for x in originals}
    cached=json.loads(Path(cfg['old_cache']).read_text(encoding='utf-8'))
    assert len(originals)==len(cached)==716
    assert all(x['messages']==by_id[x['id']]['messages'] for x in cached)
    sampled=[min([x for x in cached if x['trait']==f't{i}' and x['id'] not in cfg['targeted_ids']],
                 key=lambda x:hashlib.sha256((cfg['sample_seed']+x['id']).encode()).hexdigest())['id'] for i in range(1,10)]
    review=root/'review'
    review.mkdir(exist_ok=True)
    write_rows(review/'old_reference_examples.jsonl',[
        dict(scenario_id=sid,selection='targeted' if sid in cfg['targeted_ids'] else 'hash_selected',
             messages=by_id[sid]['messages'], metadata=by_id[sid]['metadata'])
        for sid in cfg['targeted_ids']+sampled])
    write_json(review/'old_source_verification.json',dict(repo=cfg['old_repo'],revision=cfg['old_revision'],
        sha256=sha(pinned),rows=716,messages_match_all_cached_rows=True,
        targeted_ids=cfg['targeted_ids'],sampled_ids=sampled,sample_seed=cfg['sample_seed'],
        limitation='14 complete reads, 5 targeted plus 9 selected one per principle; no population error rate'))
    for report in cfg['reports']:
        shutil.copy2(report,review/Path(report).name)
    write_json(root/'run_meta.json',dict(git_sha=git_sha(),completed_at=timestamp(),config=cfg,
        original_launch=json.loads((root/'launch_meta.json').read_text(encoding='utf-8')),
        status='completed_diagnostic_not_training',cost=json.loads((root/'cost_summary.json').read_text(encoding='utf-8'))))
    archive=root/'campaign_budget_snapshot.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root/'campaign_budget_snapshot').rglob('*')):
            if p.is_file() and p.suffix not in ['.lock','.tmp']:
                z.write(p,p.relative_to(root/'campaign_budget_snapshot').as_posix())
    files=[]
    for folder in ['review','frozen','interruption_archive']:
        files.extend((p,p.relative_to(root).as_posix()) for p in (root/folder).rglob('*') if p.is_file())
    for name in ['launch_meta.json','run_meta.json','cost_summary.json','campaign_budget_snapshot.zip']:
        files.append((root/name,'metadata/'+name))
    for p in root.glob('*.partial.jsonl'):
        files.append((p,'interruption_checkpoints/'+p.name))
    readme_path=hf_hub_download(repo,'README.md',repo_type='dataset',token=hf_token())
    readme=Path(readme_path).read_text(encoding='utf-8').split('\n## Diagnostic status:')[0]
    note='''

## Diagnostic status: completed, not approved for training

18 candidates produced 10 automatic exports. Independent full reads found stakes-filter
mistakes and material factual problems. Corpus PASS is not factual-quality approval.
The old corpus also contains comparable defects; no relative error-rate or MR-causation
claim is supported. See [smoke review](review/2026-09-21_native_lowstakes_smoke.md) and
[old-corpus comparison](review/2026-09-21_old_lowstakes_factual_check.md).

Full run exposure is $1.3276235 ($1.301356 settled plus $0.0262675 retained reservation);
cumulative campaign exposure $9.1143415 / $20. The root manifest records only resumed
segment usage. Full accounting and all campaign receipts are in metadata/.
'''
    readme=readme.replace('Fresh recipe, not yet live-validated.','Live smoke completed; diagnostic only, content validation failed.')
    readme=readme.replace('survivors of 716 requested candidates', 'survivors of 18 requested smoke candidates')
    ops=[CommitOperationAdd(path_in_repo=remote,path_or_fileobj=str(p)) for p,remote in files]
    ops.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=(readme+note).encode()))
    commit=api.create_commit(repo_id=repo,repo_type='dataset',operations=ops,
        commit_message='Archive native smoke accounting and same-standard old-corpus factual review')
    revision=commit.oid
    verified={}
    for name,local in [('dataset.jsonl',root/'dataset.jsonl'),
                       ('review/old_reference_examples.jsonl',review/'old_reference_examples.jsonl'),
                       ('metadata/campaign_budget_snapshot.zip',archive),
                       ('review/2026-09-21_native_lowstakes_smoke.md',review/'2026-09-21_native_lowstakes_smoke.md')]:
        remote=hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=hf_token())
        assert sha(remote)==sha(local)
        verified[name]=sha(local)
    write_json(root/'publication_receipt.json',dict(repo=repo,revision=revision,verified=verified))
    print(json.dumps(dict(repo=repo,revision=revision,verified_files=len(verified),old_sample=sampled)))


if __name__=='__main__':
    main()
