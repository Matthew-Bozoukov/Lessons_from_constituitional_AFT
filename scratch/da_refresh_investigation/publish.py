# ABOUTME: Publishes this completed audit under one new auxiliary HF metadata prefix and verifies every byte.
# ABOUTME: Run: uv run --frozen python scratch/da_refresh_investigation/publish.py [--verify REVISION]
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from huggingface_hub import CommitOperationAdd
from omegaconf import OmegaConf
from src.infra.huggingface import hf_api, hf_download, hf_repo_id

CFG = OmegaConf.load(Path(__file__).with_name('audit.yaml'))
OUT = ROOT / CFG.output_root
PACKAGE = OUT / 'publication'
PREFIX = str(CFG.publication_prefix)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    """Explicit allowlist: no secrets, figures, unrelated outputs or source mutations."""
    paths = {}
    for path in OUT.glob('*'):
        if path.is_file() and path.suffix in {'.json', '.md'} and path.name not in {
            'figure_receipt.json', 'publication_receipt.json', 'run_meta.json'
        }:
            paths[path.name] = path
    for path in (OUT / 'provenance').rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.md', '.diff'}:
            paths[path.relative_to(OUT).as_posix()] = path
    for path in Path(__file__).parent.glob('*'):
        if path.is_file() and path.suffix in {'.py', '.yaml', '.md'}:
            paths['code/' + path.name] = path
    paths['stakes_facts_sample.jsonl'] = OUT / 'stakes_facts_sample.jsonl'
    paths['report.md'] = ROOT / 'docs/training/2026-09-29_da_refresh_investigation.md'
    paths['2026-09-29_da_corpus_refresh.md'] = ROOT / 'docs/training/2026-09-29_da_corpus_refresh.md'
    for name, path in paths.items():
        target = PACKAGE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    generated = []
    for arm, expected in [('old', 628), ('new', 617)]:
        rows = [json.loads(line) for line in (OUT / f'{arm}_selected.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == expected
        target = PACKAGE / 'selected_rows' / f'{arm}.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        compact = [dict(id=r['id'], index=r['index'], fingerprint=r['fingerprint'],
                        messages=r['messages'], metadata=r['source_row']['metadata']) for r in rows]
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in compact), encoding='utf-8')
        generated.append(target)
    raw_count = 0
    for folder in sorted((OUT / 'judgments').iterdir()):
        if not folder.is_dir():
            continue
        rows = [read(p) for p in sorted(folder.glob('*.json'))]
        raw_count += len(rows)
        target = PACKAGE / 'judgments' / (folder.name + '.jsonl')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        generated.append(target)
    summary = read(OUT / 'audit_summary.json')
    assert raw_count == summary['cost']['requests'] == 1378
    assert sum(summary['counts']['primary'][a]['n'] for a in ['old', 'new']) == 1245
    assert sum(summary['counts']['secondary'][a]['n'] for a in ['old', 'new']) == 108
    assert sum(summary['counts']['challenge'][a]['n'] for a in ['old', 'new']) == 19
    assert summary['cost']['retained_unknown_reservations_usd'] == 0
    assert summary['cost']['api_reported_usd'] <= 30
    manifest = {'schema': 1, 'date_generated': datetime.now(timezone.utc).isoformat(),
                'experiment': 'Pinned DA corpus investigation, auxiliary analysis; no new benchmark run',
                'source_git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'config': OmegaConf.to_container(CFG, resolve=True),
                'cost': summary['cost'], 'raw_judgment_receipts': raw_count,
                'limitations': summary['warnings'], 'files': {}}
    for path in sorted([PACKAGE / name for name in paths] + generated):
        assert path.suffix not in {'.png', '.svg', '.env', '.pem', '.key'}
        manifest['files'][path.relative_to(PACKAGE).as_posix()] = {
            'sha256': digest(path), 'bytes': path.stat().st_size}
    write(PACKAGE / 'run_meta.json', manifest)
    write(OUT / 'run_meta.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', help='Verify an already published immutable revision; never republish')
    args = parser.parse_args()
    repo = hf_repo_id(str(CFG.publication_repo))
    assert repo.startswith('dougalldeepmind/')
    assert PREFIX.startswith('metadata/analyses/') and '..' not in PREFIX
    if args.verify:
        manifest = read(PACKAGE / 'run_meta.json')
        revision = args.verify
    else:
        manifest = prepare()
        api = hf_api()
        parent = api.dataset_info(repo).sha
        remote = api.list_repo_files(repo, repo_type='dataset', revision=parent)
        assert not any(name.startswith(PREFIX + '/') for name in remote), 'Destination prefix exists; use --verify'
        files = list(manifest['files']) + ['run_meta.json']
        result = api.create_commit(repo, repo_type='dataset', parent_commit=parent,
            commit_message='Add pinned DA corpus investigation and raw audit evidence',
            operations=[CommitOperationAdd(path_in_repo=PREFIX + '/' + name,
                        path_or_fileobj=str(PACKAGE / name)) for name in files])
        revision = result.oid
        write(OUT / 'publication_receipt.json', {'repo': repo, 'revision': revision,
              'parent_revision': parent, 'prefix': PREFIX, 'verified': False})
        print(json.dumps({'published_revision': revision, 'files': len(files)}), flush=True)

    def verify(name):
        path = Path(hf_download(repo, PREFIX + '/' + name, repo_type='dataset', revision=revision))
        expected = digest(PACKAGE / name)
        assert digest(path) == expected, 'Hash mismatch: ' + name
        return name

    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(verify, list(manifest['files']) + ['run_meta.json']))
    receipt_path = OUT / 'publication_receipt.json'
    receipt = read(receipt_path) if receipt_path.exists() else {}
    receipt.update(repo=repo, revision=revision, prefix=PREFIX, verified=True,
                   files_verified=len(verified), cost=manifest['cost'])
    write(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
