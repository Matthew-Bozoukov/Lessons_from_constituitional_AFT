# ABOUTME: Read-only census and deterministic sample for the old/new moral low-stakes comparison.
# ABOUTME: Run with uv run --no-sync python scratch/dataset_refresh/lowstakes_comparison_census.py.
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path('output/2026-09-15_lowstakes_comparison')
SOURCES = {
    'old': ('LASR-Callum/2026-08-26-difficult-advice-low-stakes-716', 'f268653539150af5a340164f994065f57cbef5ad'),
    'new': ('dougalldeepmind/2026-09-15-da-lowstakes-refresh-synth', 'ebafc3a60cda2a5390bde72336d660f34e580e6d'),
}
PATTERNS = {
    'ai_word': r'\bAI\b',
    'answer_label': r'\bthe (?:honest|legitimate) (?:alternative|option)\b',
    'review_word': r'\breview\b',
    'forty_eight_hours': r'\b48[ -]hours?\b',
}

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    corpus, source_receipts = {}, {}
    for arm, (repo, revision) in SOURCES.items():
        path = Path(hf_hub_download(repo, 'dataset.jsonl', repo_type='dataset', revision=revision, local_files_only=True))
        source_receipts[arm] = {'repo': repo, 'revision': revision, 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        corpus[arm] = []
        for index, row in enumerate(rows):
            meta = row['metadata']
            user = next(m['content'] for m in row['messages'] if m['role'] == 'user')
            assistant = next(m for m in row['messages'] if m['role'] == 'assistant')
            corpus[arm].append({'arm': arm, 'index': index, 'id': meta['scenario_id'], 'trait': meta['trait_id'],
                                'domain': meta.get('ls_domain', meta.get('domain')), 'metadata': meta,
                                'messages': row['messages'], 'user': user, 'reasoning': assistant['reasoning_content'], 'final': assistant['content']})
    census, selected = {}, []
    for arm, rows in corpus.items():
        assert len(rows) == 716
        scans = {key: [r['id'] for r in rows if re.search(rx, r['user'], re.I)]
                 for key, rx in PATTERNS.items()}
        census[arm] = {
            'rows': len(rows),
            'word_means': {part: statistics.mean(len(r[part].split()) for r in rows)
                           for part in ('user', 'reasoning', 'final')},
            'literal_screen_counts': {key: len(ids) for key, ids in scans.items()},
            'literal_screen_ids': scans,
            'ai_and_review': sum(bool(re.search(PATTERNS['ai_word'], r['user'], re.I))
                                 and bool(re.search(PATTERNS['review_word'], r['user'], re.I))
                                 for r in rows),
            'screens_by_trait': {f't{i}': {key: sum(r['id'] in ids for r in rows if r['trait'] == f't{i}')
                                          for key, ids in scans.items()} for i in range(1, 10)},
        }
        if arm == 'old':
            census[arm]['historical_stakes'] = dict(Counter(r['metadata']['stakes'] for r in rows))
        for i in range(1, 10):
            group = [r for r in rows if r['trait'] == f't{i}']
            chosen = min(group, key=lambda r: hashlib.sha256(
                f"lowstakes-review-20260915:{arm}:{r['id']}".encode()).hexdigest())
            selected.append(chosen)
    selected += [r for r in corpus['old'] if r['id'] in ('t7_b01_s003', 't9_b09_s007')]
    compact = [{k: r[k] for k in ('arm', 'index', 'id', 'trait', 'domain', 'messages', 'user', 'reasoning', 'final')}
               | {'stakes': r['metadata'].get('stakes'), 'selection': 'targeted historical grave label' if r['id'] in ('t7_b01_s003', 't9_b09_s007') else 'fixed hash sample, one per trait and arm'}
               for r in selected]
    result = {'sources': source_receipts, 'patterns': PATTERNS, 'census': census, 'sample_rule': 'Min SHA256 of lowstakes-review-20260915:{arm}:{id}, within each trait and arm; plus the two historically grave old rows.', 'reviewed_ids': [r['id'] for r in selected]}
    (ROOT / 'census.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'reviewed_records.json').write_text(json.dumps(compact, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({arm: {k: v for k, v in data.items() if k != 'literal_screen_ids'} for arm, data in census.items()}, indent=2))

if __name__ == '__main__':
    main()
