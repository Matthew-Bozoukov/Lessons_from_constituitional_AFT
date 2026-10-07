# ABOUTME: Rebuilds the charted original nonmoral arm on pinned plain replay while retaining every synthetic row.
# ABOUTME: Reuses trainer token masks, audits every row, and optionally publishes the mixture with complete provenance.
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.data.mixture.build_mixture import removal_order
from src.data.mixture.reasoning_backfill import base_reasoning_traces, inherited_block
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_files, training_data_tags
from src.model_profile import model_profile, render_chat
from src.naming import mix_name, model_name
from src.train.masking import build_labels
from src.train.mask_gate import expected_supervised_text, gate_generation_boundary


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load(spec):
    p = Path(hf_download(spec['repo'], spec['file'], repo_type='dataset', revision=spec['revision']))
    if spec.get('sha256'):
        assert sha(p.read_bytes()) == spec['sha256']
    return [json.loads(x) for x in p.read_text(encoding='utf-8').split('\n') if x.strip()], sha(p.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    out = ROOT / cfg['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'publication.json').exists(), 'Already published; use pinned receipt'
    parent, parent_sha = load(cfg['parent'])
    original, original_sha = load(cfg['original'])
    base, base_sha = load(cfg['base'])
    synthetic = [r for r in parent if r['source'] == cfg['parent']['source']]
    assert len(synthetic) == cfg['parent']['rows'] == 623
    original_by_messages = {canonical(r['messages']): r for r in original}
    assert all(canonical(r['messages']) in original_by_messages for r in synthetic)
    assert len({canonical(r['messages']) for r in synthetic}) == len(synthetic)
    assert all([m['role'] for m in r['messages']] == ['system', 'user', 'assistant'] for r in synthetic)
    assert all(r.get('supervise') == 'all' for r in synthetic)
    synthetic = [{**r, 'source': cfg['style'], 'supervise': 'full'} for r in synthetic]
    tok = AutoTokenizer.from_pretrained(cfg['tokenizer'], revision=cfg['tokenizer_revision'])
    profile = model_profile('qwen36')
    census, texts = [], []
    for i, r in enumerate(base + synthetic):
        text = render_chat(tok, r['messages'], r.get('tools'), render_kwargs=profile.render_kwargs)
        full = build_labels(text, tok, 100000, profile, supervise=r.get('supervise'), mask_spans=r.get('mask_spans'))
        assert len(full['input_ids']) <= cfg['max_seq_len'], f'Truncation at row {i}'
        labels = full['labels']
        assert tok.decode([x for x in labels if x != -100]) == expected_supervised_text(
            text, profile.prefill, profile.empty_think, supervise=r.get('supervise', 'full'), think_close=profile.think_close)
        count = sum(x != -100 for x in labels[1:])
        assert count > 0
        census.append({'index': i, 'source': r['source'], 'tokens': len(full['input_ids']), 'supervised_tokens': count})
        texts.append(text)
        if (i + 1) % 500 == 0:
            print(f'Audited {i+1} rows', flush=True)
    gate_generation_boundary(texts, tok, cfg['max_seq_len'], profile, True,
                             supervise=[r.get('supervise', 'full') for r in base + synthetic])
    base_tokens = sum(r['supervised_tokens'] for r in census[:len(base)])
    synthetic_tokens = sum(r['supervised_tokens'] for r in census[len(base):])
    assert base_tokens == 5000005
    order = removal_order(base, cfg['seed'])
    remaining = base_tokens
    options = [(abs(100 * synthetic_tokens / (synthetic_tokens + remaining) - cfg['target_pct']), 0)]
    for k, i in enumerate(order, 1):
        remaining -= census[i]['supervised_tokens']
        options.append((abs(100 * synthetic_tokens / (synthetic_tokens + remaining) - cfg['target_pct']), k))
    error, count = min(options)
    assert error <= cfg['tolerance_pp']
    removed = set(order[:count])
    indices = [i for i in range(len(base)) if i not in removed] + list(range(len(base), len(base) + len(synthetic)))
    random.Random(cfg['seed']).shuffle(indices)
    pool = base + synthetic
    rows = [pool[i] for i in indices]
    selected = [census[i] for i in indices]
    by_source = defaultdict(lambda: {'examples': 0, 'tokens': 0, 'supervised_tokens': 0})
    for r in selected:
        d = by_source[r['source']]
        d['examples'] += 1
        for key in ('tokens', 'supervised_tokens'):
            d[key] += r[key]
    total = sum(r['supervised_tokens'] for r in selected)
    pct = 100 * synthetic_tokens / total
    base_traces = base_reasoning_traces(cfg['base']['repo'], cfg['base']['revision'])
    traces = inherited_block(base_traces, cfg['base']['repo'], cfg['base']['revision']) if base_traces else None
    stats = {'total': {'examples': len(rows), 'tokens': sum(x['tokens'] for x in selected), 'supervised_tokens': total},
             'synthetic_examples': len(synthetic), 'synthetic_pct': pct, 'share_unit': 'supervised_tokens',
             'by_source': dict(by_source), 'reasoning_traces': traces}
    (out / 'mixture.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')
    date = datetime.now(timezone.utc).date().isoformat()
    name = mix_name(cfg['style'], cfg['target_pct'], date=date)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    audit = {'all_rows_decode_checked': len(census), 'truncated_rows': 0, 'unchanged_synthetic_messages': len(synthetic),
             'unchanged_replay_rows': len(rows) - len(synthetic), 'removed_base_rows': len(removed),
             'mixture_sha256': sha((out / 'mixture.jsonl').read_bytes()), 'input_sha256': {'parent': parent_sha, 'original': original_sha, 'base': base_sha},
             'longest_index': max(range(len(rows)), key=lambda i: selected[i]['tokens']),
             'da_index': next(i for i,r in enumerate(rows) if r['source'] == cfg['style']),
             'expected_optimizer_steps': math.ceil(len(rows) / 16), 'synthetic_pct': pct,
             'trait_counts': dict(Counter(original_by_messages[canonical(r['messages'])]['metadata']['trait_id'] for r in synthetic))}
    meta = {'git_sha': source, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
            'command': 'uv run python ' + ' '.join(sys.argv), 'new_generation_calls': 0, 'stats': stats}
    write(out / 'mixture_stats.json', stats)
    write(out / 'run_meta.json', meta)
    write(out / 'validation.json', audit)
    write(out / 'token_mask_census.json', selected)
    OmegaConf.save(OmegaConf.create(cfg), out / 'mixture_config.yaml')
    print(json.dumps({'name': name, 'stats': stats, 'audit': audit}), flush=True)
    if args.publish:
        fields = {'title': name, 'experiment': 'Exact 623 original nonmoral conversations from the latest charted Sep25 token-15 arm, with Oct5 plain replay and approximately 15% supervised synthetic tokens. All synthetic messages unchanged; replay alone is downsampled.',
                  'date_generated': date, 'constitution': 'none; historical craft preferences, no new ethical filtering',
                  'source_repo': 'teaching_claude_why_replication@' + source, 'models': cfg['tokenizer'] + '@' + cfg['tokenizer_revision'],
                  'generation_config': cfg, 'schema': 'mixture.jsonl: messages, source, supervise; native reasoning_content',
                  'provenance': {'parent': cfg['parent'], 'original': cfg['original'], 'base': cfg['base'], 'selection': cfg['selection']}}
        front = {'configs': [{'config_name': 'default', 'data_files': [{'split': 'train', 'path': 'mixture.jsonl'}]}],
                 'tags': training_data_tags('mixture', cfg['style'], 'none')}
        files = [out / n for n in ('mixture.jsonl', 'mixture_stats.json', 'run_meta.json', 'validation.json', 'token_mask_census.json', 'mixture_config.yaml')]
        push_files(files, hf_repo_id(name), fields, private=False, front_matter=front)
        info = hf_api().dataset_info(hf_repo_id(name))
        for p in files:
            downloaded = Path(hf_download(hf_repo_id(name), p.name, repo_type='dataset', revision=info.sha))
            assert downloaded.read_bytes() == p.read_bytes(), p.name
        write(out / 'publication.json', {'repo': hf_repo_id(name), 'revision': info.sha, 'files_verified': len(files),
                                         'organism': hf_repo_id(model_name('qwen36', 0, cfg['style'] + '-15', date=date)), 'audit': audit})


if __name__ == '__main__':
    main()
