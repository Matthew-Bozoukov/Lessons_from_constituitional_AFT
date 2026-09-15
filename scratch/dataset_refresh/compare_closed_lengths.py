# ABOUTME: Compare closed incomplete advice pools with the pinned full new DA export using cached artifacts only.
# ABOUTME: Reuses retained-row token diagnostics and reports exposure differences without selecting or padding data.
from pathlib import Path
import statistics

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from scratch.dataset_refresh import run as runtime
from scratch.dataset_refresh.validate_mixtures import TOKENIZER, token_audit
from src.model_profile import model_profile


def stats(values):
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'total': sum(values)}


def main():
    q = Path('output/2026-09-15_dataset_refresh_quality_screen')
    source = Path(hf_hub_download('dougalldeepmind/2026-09-14-da-synth', 'dataset.jsonl', repo_type='dataset',
        revision='013886238fca238c4d54ace96530f444bb2b2f02', local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    profile = model_profile(TOKENIZER)
    results = {}
    for arm in ['new_da_full_export', 'low', 'nonmoral']:
        path = source if arm == 'new_da_full_export' else q / f'closed_{arm}/retained_conversations_for_audit.jsonl'
        rows = runtime.read_rows(path)
        if arm == 'new_da_full_export':
            diagnostics = [token_audit(r, tokenizer, profile, 8192) for r in rows]
        else:
            import json
            prior = json.loads((q / f'final_partial_{arm}/token_mask_audit.json').read_text(encoding='utf-8'))
            keep = {r['metadata']['scenario_id'] for r in rows}
            diagnostics = [r for r in prior['diagnostics'] if r['id'] in keep]
            assert len(diagnostics) == len(rows)
        characters = {'system': [], 'user': [], 'reasoning': [], 'response': []}
        for row in rows:
            by_role = {m['role']: m for m in row['messages']}
            for role in ['system', 'user']:
                characters[role].append(len(by_role[role]['content']))
            characters['reasoning'].append(len(by_role['assistant']['reasoning_content']))
            characters['response'].append(len(by_role['assistant']['content']))
        results[arm] = {'path': str(path), 'input_sha256': runtime.digest(path.read_bytes()), 'rows': len(rows),
                        'characters': {k: stats(v) for k, v in characters.items()},
                        'tokens': {k: stats([r[k] for r in diagnostics]) for k in ['training_tokens', 'supervised_tokens']}}
    report = {'scope': 'Descriptive comparison of full pinned DA752 against incomplete retained LOW706/NON631, not a matched716 analysis. No length matching, padding or selection performed.',
              'tokenizer': TOKENIZER, 'max_train_tokens': 8192, 'untruncated': True,
              'helper_sha256': runtime.digest(Path(__file__).read_bytes()), 'corpora': results}
    runtime.write_json(q / 'final_length_comparability.json', report)
    print({k: {'rows': v['rows'], 'median_supervised_tokens': v['tokens']['supervised_tokens']['median'],
               'mean_supervised_tokens': v['tokens']['supervised_tokens']['mean']} for k, v in results.items()})


if __name__ == '__main__':
    main()
