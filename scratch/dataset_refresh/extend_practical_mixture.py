# ABOUTME: Append fresh practical low-stakes rows to an immutable mixture near a supervised-token target.
# ABOUTME: Run: uv run python -m scratch.dataset_refresh.extend_practical_mixture --config scratch/dataset_refresh/token15_mixture.yaml --audit-parent
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from scratch.dataset_refresh.validate_mixtures import canonical, digest, load_rows, token_audit
from scratch.dataset_refresh.run import write_json, write_rows
from src.data.mixture.sources.base import clean_messages
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_run_dir, training_data_tags
from src.model_profile import model_profile
from src.naming import mix_name
from src.utils import git_sha, origin_url


def fetch(spec):
    path = Path(hf_download(spec['repo'], spec.get('file', 'dataset.jsonl'),
                           repo_type='dataset', revision=spec['revision']))
    if spec.get('sha256'):
        assert digest(path.read_bytes()) == spec['sha256'], 'Pinned source hash differs'
    return path, load_rows(path)


def counter_rows(rows):
    return Counter(canonical(r) for r in rows)


def balanced_order(rows, domains, seed):
    """Fixed trait/domain round robin; no quality or answer-length ranking."""
    cells = defaultdict(list)
    for row in rows:
        meta = row['metadata']
        assert meta['assigned_domain'] in domains
        cells[(meta['trait_id'], meta['assigned_domain'])].append(row)
    for key, values in cells.items():
        cells[key] = deque(sorted(values, key=lambda r: digest(f"{seed}:{r['metadata']['scenario_id']}")))
    traits = [f't{i}' for i in range(1, 10)]
    queues = {}
    for i, trait in enumerate(traits):
        rotated = domains[i:] + domains[:i]
        queue = []
        while any(cells[(trait, domain)] for domain in domains):
            for domain in rotated:
                if cells[(trait, domain)]:
                    queue.append(cells[(trait, domain)].popleft())
        queues[trait] = deque(queue)
    ordered = []
    while any(queues.values()):
        for trait in traits:
            if queues[trait]:
                ordered.append(queues[trait].popleft())
    assert len(ordered) == len(rows)
    return ordered


def closest_prefix(counts, old_low, replay, target):
    shares = [(old_low / (old_low + replay), 0)]
    total = old_low
    for i, count in enumerate(counts, 1):
        assert count > 0
        total += count
        shares.append((total / (total + replay), i))
    return min(shares, key=lambda item: (abs(item[0] - target), item[1]))


def verify_training_loader(path, expected):
    """Exercise the trainer's actual Arrow loader, including JSON block boundaries."""
    from datasets import load_dataset
    dataset = load_dataset('json', data_files=str(path), split='train')
    assert len(dataset) == len(expected)
    for loaded, raw in zip(dataset, expected):
        assert loaded['messages'] == raw['messages']
        assert loaded['source'] == raw['source']
        assert (loaded.get('supervise') or 'all') == (raw.get('supervise') or 'all')
    return dict(rows=len(dataset), columns=dataset.column_names, verified=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--audit-parent', action='store_true')
    parser.add_argument('--synth-revision')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    root = Path(cfg['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    parent_path, parent = fetch(cfg['parent'])
    assert len(parent) == 10000
    old_low = [r for r in parent if r['source'] == cfg['style']]
    replay = [r for r in parent if r['source'] != cfg['style']]
    assert len(old_low) == 716 and len(replay) == 9284
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer']['repo'], revision=cfg['tokenizer']['revision'])
    profile = model_profile(cfg['tokenizer']['repo'])
    identity = {p: digest(Path(p).read_bytes()) for p in [
        'src/train/masking.py', 'src/model_profile.py', 'configs/models/qwen36.yaml']}
    census_path = root / 'parent_token_audit.json'
    if not census_path.exists():
        counts, by_source = [], defaultdict(Counter)
        for i, row in enumerate(parent):
            result = token_audit(row, tokenizer, profile, cfg['max_seq_len'])
            counts.append(result['supervised_tokens'])
            by_source[row['source']].update(result)
            by_source[row['source']]['rows'] += 1
            if (i + 1) % 1000 == 0:
                print('AUDITED', i + 1, flush=True)
        write_json(census_path, dict(parent=cfg['parent'], tokenizer=cfg['tokenizer'],
            code_hashes=identity, counts=counts, by_source=dict(by_source)))
    census = json.loads(census_path.read_text(encoding='utf-8'))
    assert census['code_hashes'] == identity and census['parent'] == cfg['parent']
    old_tokens = sum(n for r, n in zip(parent, census['counts']) if r['source'] == cfg['style'])
    replay_tokens = sum(census['counts']) - old_tokens
    assert (old_tokens, replay_tokens) == (670033, 4512849), 'Current mask differs; inspect before mixing'
    print('PARENT_TOKENS', old_tokens, replay_tokens, flush=True)
    if args.audit_parent:
        return
    assert args.synth_revision and len(args.synth_revision) == 40
    prior_path, prior = fetch(cfg['prior_synth'])
    extended_spec = dict(repo=cfg['extended_synth_repo'], revision=args.synth_revision)
    extended_path, extended = fetch(extended_spec)
    assert extended[:len(prior)] == prior, 'Native extension must carry all prior rows verbatim'
    prior_ids = {r['metadata']['scenario_id'] for r in prior}
    fresh = extended[len(prior):]
    assert fresh and all(r['metadata']['scenario_id'] not in prior_ids for r in fresh)
    assert len({r['metadata']['scenario_id'] for r in fresh}) == len(fresh)
    ordered = balanced_order(fresh, cfg['domains'], cfg['seed'])
    rows, counts = [], []
    prior_prompts = {canonical(r['messages'][:2]) for r in old_low}
    for raw in ordered:
        meta = raw['metadata']
        assert str(meta['prompt_stakes']) in ('0', '1') and meta['prompt_scope'] == 'text_advice'
        messages = clean_messages(raw['messages'])
        assert messages and [m['role'] for m in messages] == ['system', 'user', 'assistant']
        assert messages[-1].get('reasoning_content') and not raw.get('tools')
        prompt = canonical(messages[:2])
        assert prompt not in prior_prompts, 'Exact repeated prompt'
        prior_prompts.add(prompt)
        row = dict(messages=messages, supervise='all', source=cfg['style'])
        result = token_audit(row, tokenizer, profile, cfg['max_seq_len'])
        rows.append(row)
        counts.append(result['supervised_tokens'])
    share, take = closest_prefix(counts, old_tokens, replay_tokens, cfg['target_pct'] / 100)
    assert abs(share - cfg['target_pct'] / 100) <= cfg['tolerance_pp'] / 100, 'Not enough accepted tokens'
    combined = parent + rows[:take]
    assert counter_rows(combined) == counter_rows(parent) + counter_rows(rows[:take])
    assert combined[:10000] == parent
    # Preserve the existing row order and payloads. The trainer owns seeded shuffling.
    build = root / 'release'
    build.mkdir(exist_ok=False)
    write_rows(build/'mixture.jsonl', combined)
    assert load_rows(build/'mixture.jsonl')[:10000] == parent
    loader = verify_training_loader(build/'mixture.jsonl', combined)
    added = sum(counts[:take])
    selected = [dict(scenario_id=r['metadata']['scenario_id'], trait_id=r['metadata']['trait_id'],
                     assigned_domain=r['metadata']['assigned_domain'], supervised_tokens=n)
                for r, n in zip(ordered[:take], counts[:take])]
    report = dict(rows=len(combined), old_rows_preserved=10000, replay_rows=9284,
        low_stakes_rows=716+take, added_rows=take, new_accepted_candidates=len(fresh),
        low_stakes_supervised_tokens=old_tokens+added, replay_supervised_tokens=replay_tokens,
        total_supervised_tokens=old_tokens+added+replay_tokens, added_supervised_tokens=added,
        low_stakes_token_pct=100*share, low_stakes_row_pct=100*(716+take)/len(combined),
        new_trait_counts=dict(Counter(x['trait_id'] for x in selected)),
        new_domain_counts=dict(Counter(x['assigned_domain'] for x in selected)),
        selected=selected, parent=cfg['parent'], synthetic=extended_spec,
        mixture_sha256=digest((build/'mixture.jsonl').read_bytes()),
        mask_code_hashes=identity, truncated_rows=0, rewritten_rows=0,
        training_loader=loader)
    write_json(build/'mixture_validation.json', report)
    shutil.copy2(args.config, build/'mixture_config.yaml')
    shutil.copy2(census_path, build/'parent_token_audit.json')
    command = 'uv run python -m scratch.dataset_refresh.extend_practical_mixture --config '+args.config+' --synth-revision '+args.synth_revision+' --publish'
    meta = dict(git_sha=git_sha(), timestamp_utc=datetime.now(timezone.utc).isoformat(), command=command, config=cfg)
    write_json(build/'run_meta.json', meta)
    name = mix_name(cfg['style'], cfg['target_pct'])
    repo = hf_repo_id(name)
    print('MIXTURE', json.dumps({k:v for k,v in report.items() if k != 'selected'}), flush=True)
    if args.publish:
        assert not hf_api().repo_exists(repo, repo_type='dataset'), 'Do not overwrite an existing arm'
        fields = dict(title=name, date_generated=meta['timestamp_utc'][:10],
            experiment=f"Append-only practical low-stakes extension: {100*share:.5f}% of supervised tokens. All 10,000 prior rows retained, {take} fresh rows added.",
            constitution=cfg['constitution'], source_repo=origin_url()+' @ '+git_sha(),
            models=json.dumps(cfg['tokenizer'])+'; synthetic Sonnet provenance in '+cfg['extended_synth_repo']+'@'+args.synth_revision,
            generation_config=json.dumps(cfg), schema='mixture.jsonl: model-agnostic messages, source and optional supervise. Assistant-only masks; reasoning plus answer supervised. Token counts are in mixture_validation.json.',
            provenance=command,
            limitations='Inherited source quality limitations remain. The 15 in this arm name denotes supervised-token percentage, not row percentage. No base rows were replaced or edited.')
        push_run_dir(build, name, fields, front_matter=dict(configs=[dict(config_name='default', data_files='mixture.jsonl', default=True)],
            tags=training_data_tags('mixture',cfg['style'],cfg['constitution'])))
        revision = hf_api().dataset_info(repo).sha
        restored = Path(hf_download(repo, 'mixture.jsonl', repo_type='dataset', revision=revision, local_dir=root/'verification'))
        assert digest(restored.read_bytes()) == report['mixture_sha256']
        receipt = dict(repo=repo, revision=revision, verified=True, **{k:report[k] for k in ['rows','mixture_sha256','low_stakes_token_pct','added_rows']})
        write_json(root/'mixture_receipt.json', receipt)
        print('PUBLISHED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
