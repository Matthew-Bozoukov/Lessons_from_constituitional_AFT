# ABOUTME: Read-only metadata inventory for the nonmoral-deliberation planning investigation.
# ABOUTME: Downloads small provenance files only; never inference, weights, rentals, or uploads.
import json
import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from src.infra.huggingface import hf_api, hf_token
from src.infra.huggingface import REQUIRED_FIELDS, gate_push
from huggingface_hub import HfApi

OUT = Path('output/nonmoral_investigation/20260908')
REPOS = [
 ('dataset', 'LASR-Callum/2026-09-04-odcv-qwen36-0-nonmoral-deliberation-7'),
 ('dataset', 'LASR-Callum/2026-09-04-odcv-qwen36-0-da-principle-scoped-7'),
 ('dataset', 'LASR-Callum/2026-09-02-craft-tensions-nonmoral-deliberation'),
 ('dataset', 'LASR-Callum/2026-09-02-table2-9284-nonmoral-deliberation-684-train-mixture'),
 ('model', 'LASR-Callum/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch'),
 ('model', 'LASR-Callum/qwen3.6-27b-lora-table2-only-9284-r64'),
 ('model', 'matboz/qwen3.6-27b-lora-9284-numina-control-716-r64'),
]

def audit_workstream(out):
    """Public, read-only audit of artifacts created by this workstream since September 8."""
    out = Path(out)
    api = HfApi(token=False)
    items = []
    for kind, list_repos in [('dataset', api.list_datasets), ('model', api.list_models)]:
        items += [(kind, r.id) for r in list_repos(author='dougalldeepmind', search='nonmoral')
                  if r.id.split('/')[-1][:10] >= '2026-09-08']
    migration = json.loads(Path('output/nonmoral_investigation/20260909/hf_artifact_audit/canonical_publications.json').read_text())
    items += [('dataset', r['new_repo']) for r in migration['publications']]
    def check(item):
        kind, repo = item
        info = api.repo_info(repo, repo_type=kind, files_metadata=True)
        assert not info.private, repo
        dest = out / repo.split('/')[-1]
        dest.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo, 'README.md', repo_type=kind, revision=info.sha, token=False)
        card = Path(path).read_text(encoding='utf-8')
        (dest / 'README.md').write_text(card, encoding='utf-8')
        fields = dict(re.findall(r'^\| `([^`]+)` \| (.*?) \|$', card, re.M))
        issues = [f'missing card field: {f}' for f in REQUIRED_FIELDS if not fields.get(f)]
        try:
            gate_push(repo, fields)
        except ValueError as exc:
            issues.append(str(exc))
        files = [s.rfilename for s in info.siblings]
        figures = [f for f in files if f.startswith('results/') and Path(f).suffix.lower() in ('.png', '.svg', '.pdf')]
        if figures:
            issues.append('published figures should remain local')
        tags = list(info.tags or [])
        if 'eval-run' in tags:
            for required in ('results/results.json', 'metadata/run_meta.json', 'metadata/odcv_config.yaml'):
                if required not in files:
                    issues.append('missing eval file: ' + required)
            for prefix in ('eval:', 'model:', 'mode:'):
                if not any(t.startswith(prefix) for t in tags):
                    issues.append('missing eval tag: ' + prefix)
        if kind == 'model':
            for required in ('adapter_model.safetensors', 'adapter_config.json', 'training_meta.json', 'train_config.yaml'):
                if required not in files:
                    issues.append('missing model file: ' + required)
        return dict(repo=info.id, kind=kind, revision=info.sha, public=True,
                    card_fields=fields, tags=tags, files=files, figures=figures, issues=issues)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, sorted(set(items))))
    (out / 'inventory.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(repositories=len(rows), issues={r['repo']: r['issues'] for r in rows if r['issues']}), indent=2))


def inspect(item):
    kind, repo = item
    try:
        info = hf_api().repo_info(repo, repo_type=kind, files_metadata=True)
        result = dict(repo=repo, kind=kind, revision=info.sha, private=info.private,
                      files=[dict(name=s.rfilename, size=s.size) for s in info.siblings])
        dest = OUT / repo.split('/')[-1]
        dest.mkdir(parents=True, exist_ok=True)
        for s in info.siblings:
            if s.rfilename in {'README.md', 'results.json', 'run_meta.json',
                               'training_meta.json', 'adapter_config.json', 'manifest.json',
                               'results/results.json', 'results/judging_run_meta.json',
                               'metadata/run_meta.json', 'metadata/odcv_config.yaml',
                               'metadata/progress/run_meta.json', 'results/progress_results.json'}:
                if (s.size or 0) > 500_000:
                    continue
                path = hf_hub_download(repo, s.rfilename, repo_type=kind,
                                       revision=info.sha, token=hf_token())
                (dest / s.rfilename).parent.mkdir(parents=True, exist_ok=True)
                (dest / s.rfilename).write_bytes(Path(path).read_bytes())
        return result
    except Exception as exc:
        return dict(repo=repo, kind=kind, error_type=type(exc).__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workstream-audit', type=Path)
    args = parser.parse_args()
    if args.workstream_audit:
        audit_workstream(args.workstream_audit)
        raise SystemExit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, REPOS))
    (OUT / 'inventory.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    # The exact historical training revision, not today's movable dataset HEAD.
    mix_path = hf_hub_download(REPOS[3][1], 't2_9284_nonmoral_684.jsonl',
                               repo_type='dataset',
                               revision='6364505df02b0020b030bf379bd42285a14de6a5',
                               token=hf_token())
    import hashlib
    from collections import Counter
    corpus_path = Path('output/nonmoral_deliberation/20260902_013651/dataset.jsonl')
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding='utf-8').splitlines()]
    mixture = [json.loads(line) for line in Path(mix_path).read_text(encoding='utf-8').splitlines()]
    def sid(row):
        return row.get('scenario_id') or (row.get('metadata') or {}).get('scenario_id')
    trained = {sid(row) for row in mixture if sid(row)}
    old_trained = {(row.get('metadata') or {}).get('scenario_id') for row in mixture}
    old_trained.discard(None)
    audit = dict(corpus_rows=len(corpus), mixture_rows=len(mixture),
                 trained_scenario_ids=len(trained), old_helper_trained_ids=len(old_trained),
                 actual_holdout=sum(sid(row) not in trained for row in corpus),
                 old_first30_training_overlap=sum(sid(row) in trained for row in corpus[:30]),
                 corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                 mixture_sha256=hashlib.sha256(Path(mix_path).read_bytes()).hexdigest(),
                 mixture_sources=dict(Counter(row.get('source', 'missing') for row in mixture)))
    (OUT / 'local_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Local audit:', json.dumps(audit))
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
        print('Small files:', [s['name'] for s in result.get('files', [])
                               if (s['size'] or 0) < 500_000][:25])
