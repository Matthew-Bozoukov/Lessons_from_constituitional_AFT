# ABOUTME: Read-only inventory of published model lineage and evaluation target metadata.
# ABOUTME: Writes a dated local receipt; never trains, samples, or changes remote artifacts.
import concurrent.futures as cf
import json
from pathlib import Path
import re
import urllib.request
from dotenv import dotenv_values

ROOT = Path('output/2026-09-28_gptoss120b_inventory')
cfg = dotenv_values(r'C:\Users\nikak\source\repos\LASR\teaching_claude_why_replication\.env')
HEADERS = {'Authorization': 'Bearer ' + cfg['HF_TOKEN']}
PAT = re.compile(r'gpt.?oss|gptoss|tinker', re.I)

def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=25) as r:
            return json.load(r)
    except Exception as e:
        return {'_error': str(e)}

def inspect(item):
    kind, row = item
    repo, sha = row['id'], row['sha']
    prefix = 'datasets/' if kind == 'datasets' else ''
    result = {'kind': kind, 'repo': repo, 'revision': sha, 'checked': {}}
    if kind == 'models':
        names = ['adapter_config.json']
    else:
        info = get('https://huggingface.co/api/datasets/' + repo + '/revision/' + sha)
        if '_error' in info:
            result['info_error'] = info
            return result
        names = [s['rfilename'] for s in info.get('siblings', [])
                 if s['rfilename'].endswith('run_meta.json')]
        result['metadata_files'] = names
    for name in names:
        meta = get(f'https://huggingface.co/{prefix}{repo}/resolve/{sha}/{name}')
        if '_error' in meta:
            result['checked'][name] = meta
        else:
            result['checked'][name] = {
                k: v for k, v in meta.items()
                if k in ('target', 'target_revision', 'base_model', 'base_model_name_or_path',
                         'base_model_revision', 'model', 'models', 'eval', 'mode')}
            if PAT.search(json.dumps(meta)):
                result.setdefault('matches', {})[name] = meta
    return result

jobs = []
for author in ('dougalldeepmind', 'matboz'):
    for kind in ('models', 'datasets'):
        rows = json.loads((ROOT / f'{author}_{kind}.json').read_text(encoding='utf-8'))
        for row in rows:
            if kind == 'models' or any(t.startswith('eval') for t in row.get('tags', [])):
                jobs.append((kind, row))
print('Inspecting', len(jobs), 'model/eval repositories', flush=True)
results = []
with cf.ThreadPoolExecutor(max_workers=12) as pool:
    for result in pool.map(inspect, jobs):
        results.append(result)
        if result.get('matches'):
            print('MATCH', result['repo'], json.dumps(result['matches'])[:2500], flush=True)
(ROOT / 'lineage_scan.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('DONE', json.dumps({'repos': len(results), 'matches': sum(bool(r.get('matches')) for r in results),
    'with_eval_metadata': sum(bool(r.get('metadata_files')) for r in results),
    'dataset_info_errors': sum('info_error' in r for r in results),
    'metadata_read_errors': sum('_error' in m for r in results for m in r['checked'].values())}), flush=True)
