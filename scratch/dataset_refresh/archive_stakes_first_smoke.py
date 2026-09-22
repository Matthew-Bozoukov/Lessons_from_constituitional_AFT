# ABOUTME: Collects complete per-candidate smoke provenance and publishes its reviewed run archive.
# ABOUTME: Run: uv run --no-sync python -m scratch.dataset_refresh.archive_stakes_first_smoke --config scratch/dataset_refresh/archive_stakes_first_smoke.yaml [--publish]
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from omegaconf import OmegaConf


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    root = Path(cfg['run_root'])
    review = root / 'review'
    review.mkdir(exist_ok=True)
    records = {}
    for path in sorted(root.glob('stage_*.jsonl'), key=lambda p: int(p.name.split('_')[1])):
        if '.partial.' in path.name:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            sid = row.get('scenario_id', row.get('metadata', {}).get('scenario_id'))
            if sid:
                records.setdefault(sid, {})[path.stem] = row
    dest = review / 'complete_conversations.jsonl'
    dest.write_text(''.join(json.dumps({'scenario_id': sid, 'stages': stages}, ensure_ascii=False)+'\n'
                            for sid, stages in sorted(records.items())), encoding='utf-8')
    print(json.dumps({'candidates': len(records), 'file': str(dest)}))
    if not args.publish:
        return
    from huggingface_hub import CommitOperationAdd, hf_hub_download
    from src.infra.huggingface import hf_api, hf_token
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    assert not manifest['aborted'], 'Archive interrupted runs separately before marking completed'
    report = Path(cfg['report'])
    assert report.exists() and (review / 'summary.json').exists()
    (review / 'report.md').write_bytes(report.read_bytes())
    for reference in cfg.get('reference_reports', []):
        source = Path(reference)
        (review / source.name).write_bytes(source.read_bytes())
    ledger = root / 'campaign_budget_snapshot'
    entries = json.loads((ledger / 'spend.json').read_text(encoding='utf-8'))
    own = [e for e in entries if e.get('run_root') == str(root.resolve())]
    receipt_zip = root / 'smoke_receipts.zip'
    with zipfile.ZipFile(receipt_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('spend.json', json.dumps(entries, indent=2))
        for entry in own:
            p = ledger / 'raw_calls' / f"{entry['call_id']:06d}.json"
            assert p.exists(), f'Missing receipt {p}'
            z.write(p, f'raw_calls/{p.name}')
    api = hf_api()
    repo = manifest['hf_repo']
    readme = Path(hf_hub_download(repo, 'README.md', repo_type='dataset', token=hf_token())).read_text(encoding='utf-8')
    readme = readme.replace('Updated recipe is offline-validated only;',
                            'Live smoke completed; not cleared for full generation;')
    if cfg.get('current_recipe'):
        current = OmegaConf.to_container(OmegaConf.load(cfg['current_recipe']), resolve=True)
        old = manifest['config']['card']['experiment']
        readme = readme.replace(old, current['card']['experiment'])
        readme = readme.replace(old.replace('Updated recipe is offline-validated only;',
                               'Live smoke completed; not cleared for full generation;'),
                               current['card']['experiment'])
    prefix = f'runs/{root.name}'
    note = (f'\n\n## Reviewed smoke: {root.name}\n\n'
            f"{cfg.get('readiness_summary', 'Read the full report for readiness and limitations.')}\n\n"
            f'This is diagnostic smoke data. Read the [current report]({prefix}/review/report.md) '
            f'and [summary]({prefix}/review/summary.json) before using it. '
            f'The [complete run archive]({prefix}) is authoritative for this run; '
            'older root-level files may belong to preceding smoke revisions. '
            f"The preceding complete archive is pinned at `{cfg['previous_revision']}`.\n")
    files = [p for p in root.glob('stage_*.jsonl') if '.partial.' not in p.name]
    files += [p for folder in ['review', 'frozen'] for p in (root/folder).rglob('*') if p.is_file()]
    files += [root/n for n in ['dataset.jsonl','manifest.json','launch_meta.json','cost_summary.json','smoke_receipts.zip']]
    operations = [CommitOperationAdd(path_in_repo=f'{prefix}/{p.relative_to(root).as_posix()}', path_or_fileobj=str(p)) for p in files]
    operations += [CommitOperationAdd(path_in_repo=f'review/{p.name}', path_or_fileobj=str(p)) for p in review.iterdir() if p.is_file()]
    operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=(readme+note).encode()))
    revision = api.create_commit(repo_id=repo, repo_type='dataset', operations=operations,
                                 commit_message=f'Archive reviewed native smoke {root.name}').oid
    verified = {}
    verify_files = [root/'dataset.jsonl',review/'report.md',review/'summary.json',dest,receipt_zip]
    verify_files += [root / name for name in cfg.get('verify_additional', [])]
    for p in verify_files:
        remote = f'{prefix}/{p.relative_to(root).as_posix()}'
        fetched = Path(hf_hub_download(repo, remote, repo_type='dataset', revision=revision, token=hf_token()))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert hashlib.sha256(fetched.read_bytes()).hexdigest() == digest
        verified[remote] = digest
    dump(root/'publication_receipt.json', {'repo':repo,'revision':revision,'verified':verified})
    print(json.dumps({'repo':repo,'revision':revision,'verified_files':len(verified)}))


if __name__ == '__main__':
    main()
