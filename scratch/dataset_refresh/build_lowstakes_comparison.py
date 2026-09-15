# ABOUTME: Build a self-contained, quotation-checked comparison from frozen local corpus reads.
# ABOUTME: Run with uv run --no-sync python scratch/dataset_refresh/build_lowstakes_comparison.py.
import json
import statistics
from pathlib import Path

from huggingface_hub import hf_hub_download
from scipy.stats import t

ROOT = Path('output/2026-09-15_lowstakes_comparison')
DEST = Path('C:/Users/nikak/.codex/visualizations/2026/09/14/01a0a0e6-ea9c-7e81-a65c-35b6c7b3a70a/low-stakes-comparison.html')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    records = read(ROOT / 'reviewed_records.json')
    records = [{k: r[k] for k in ('arm','id','trait','domain','user','reasoning','final','stakes','selection')}
               | {'system': next((m['content'] for m in r['messages'] if m['role'] == 'system'), '')}
               for r in records]
    lookup = {r['id']: r for r in records}
    notes = read('scratch/dataset_refresh/lowstakes_comparison_notes.json')
    for p in notes['pairs']:
        for arm in ('old','new'):
            for q in p[f'{arm}_quotes']:
                assert q['text'] in lookup[p[arm]][q['field']], (p['title'], arm, q)
    old = read(hf_hub_download('dougalldeepmind/2026-09-05-odcv-qwen36-0-da-lowstakes-7',
        'results/results.json', repo_type='dataset', revision='8ccdc4476ce9d959cc1783d9d0a90726a8eb6b7e', local_files_only=True))
    new = read('C:/odcv-low/2026-09-15_odcv_refresh_low_20260915_205557/results/results.json')
    old_cells, new_cells = old['per_scenario_medians'], new['per_scenario_medians']
    scenarios = sorted(old_cells['mandated'])
    assert len(scenarios) == 40
    stats = {}
    for group in ('overall','mandated','incentivized'):
        variants = ('mandated','incentivized') if group == 'overall' else (group,)
        diffs = [100 * statistics.mean(statistics.mean(x >= 3 for x in new_cells[v][s]) - statistics.mean(x >= 3 for x in old_cells[v][s]) for v in variants) for s in scenarios]
        delta = statistics.mean(diffs)
        err = t.ppf(.975,39) * statistics.stdev(diffs) / (40 ** .5)
        stats[group] = {'delta_pp': delta, 'ci95': [delta-err,delta+err]}
        for arm, cells in (('old',old_cells),('new',new_cells)):
            values = [x for v in variants for s in scenarios for x in cells[v][s]]
            stats[group][arm] = {'flags': sum(x>=3 for x in values), 'n': len(values), 'mr':100*statistics.mean(x>=3 for x in values)}
    payload = {**notes, 'records':records, 'census':read(ROOT/'census.json')['census'], 'eval':stats}
    for arm in ('old','new'):
        payload['census'][arm].pop('literal_screen_ids')
    (ROOT/'comparison_evidence.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    template = Path('scratch/dataset_refresh/lowstakes_comparison_fragment.html').read_text(encoding='utf-8')
    assert template.count('<!-- COMPARISON_DATA -->') == 1
    output = template.replace('<!-- COMPARISON_DATA -->', json.dumps(payload, ensure_ascii=False).replace('</', '<\\/'))
    assert len(output.encode()) < 1_000_000
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(output, encoding='utf-8')
    print(json.dumps({'destination':str(DEST), 'bytes':len(output.encode()), 'reviewed':len(records), 'eval':stats}, indent=2))

if __name__ == '__main__':
    main()
