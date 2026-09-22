# ABOUTME: Deterministically select accepted origin records under explicit phase caps and trait quotas.
# ABOUTME: Preserves the eligible pool and selection policy; never edits records or overrides exclusions.
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random

from omegaconf import OmegaConf
from scratch.dataset_refresh import run as runtime


def balanced(rows, count, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda x: x['entry']['candidate_id']):
        groups[row['domain']].append(row)
    domains = sorted(groups)
    rng.shuffle(domains)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count and any(groups.values()):
        for domain in domains:
            if groups[domain] and len(selected) < count:
                selected.append(groups[domain].pop())
    return selected


def select(config):
    arm = config['arm']
    pools, blocked = [], []
    for phase in config['phases']:
        root = Path(phase['root']).resolve()
        runtime.validate_arm(root, arm)
        rows = []
        for path in sorted((root / arm / 'records').glob('*/result.json')):
            result = runtime.load_result(path)
            if result['status'] != 'accepted':
                blocked.append({'root': str(root), 'candidate_id': path.parent.name,
                                'status': result['status']})
                continue
            rows.append({'entry': {'root': str(root), 'arm': arm, 'candidate_id': path.parent.name,
                                   'result_sha256': runtime.digest(path.read_bytes())},
                         'trait_id': result['trait_id'], 'domain': result['record']['domain']})
        pools.append((phase, rows))
    picked, shortages = [], {}
    for trait, quota in runtime.quotas().items():
        remaining = quota
        for index, (phase, rows) in enumerate(pools):
            available = [r for r in rows if r['trait_id'] == trait]
            cap = phase.get('max_per_trait', quota)
            take = min(remaining, cap, len(available))
            picked.extend(balanced(available, take, int(config['seed']) + 100 * index + int(trait[1:])))
            remaining -= take
        if remaining:
            shortages[trait] = remaining
    report = {'config': config, 'eligible_pool': [r for _, rows in pools for r in rows],
              'blocked_statuses': dict(Counter(r['status'] for r in blocked)),
              'eligible_counts': {str(p['root']): dict(Counter(r['trait_id'] for r in rows)) for p, rows in pools},
              'selected_counts': dict(Counter(r['trait_id'] for r in picked)),
              'selected_phase_counts': dict(Counter(r['entry']['root'] for r in picked)),
              'shortages': shortages,
              'interpretation': 'Balances assigned domains, not empirically verified mechanisms. Independent holds must be applied as bound exclusions before selection.'}
    return {'arm': arm, 'entries': [r['entry'] for r in picked]}, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    selection, report = select(cfg)
    output = Path(args.output)
    runtime.write_json(output / 'selection_policy_and_pool.json', report)
    if report['shortages']:
        print('Shortages:', report['shortages'])
        raise SystemExit(2)
    runtime.write_json(output / 'selection.json', selection)
    print('Selected', len(selection['entries']), report['selected_phase_counts'])
