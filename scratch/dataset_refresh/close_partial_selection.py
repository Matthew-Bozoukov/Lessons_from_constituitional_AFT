# ABOUTME: Bind a final incomplete subset to an already audited pool after exact-hash exclusions.
# ABOUTME: Preserves retained conversation bytes and proves inherited token checks; never authorizes training or inference.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from contextlib import ExitStack

from filelock import FileLock
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as runtime, select_release


def identity(entry):
    return (entry['root'], entry['arm'], entry['candidate_id'], entry['result_sha256'])


def close(previous, config, output):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if output.exists() or output.is_relative_to(previous):
        raise ValueError('Use a new sibling analysis directory')
    for phase in cfg['phases']:
        if output.is_relative_to(Path(phase['root']).resolve()):
            raise ValueError('Output must be outside original run roots')
    prior_policy = json.loads((previous / 'selection_policy_and_pool.json').read_text(encoding='utf-8'))
    if prior_policy['config'] != cfg:
        raise ValueError('Subset proof cannot change selection policy')
    source = previous / 'conversations_for_audit.jsonl'
    original_sha = runtime.digest(source.read_bytes())
    audit = json.loads((previous / 'token_mask_audit.json').read_text(encoding='utf-8'))
    if audit['input_sha256'] != original_sha or audit['status'] != 'passed' or audit['failures']:
        raise ValueError('Require a passing mask audit of these exact original bytes')
    with ExitStack() as locks:
        for root in sorted({str(Path(p['root']).resolve()) for p in cfg['phases']}):
            locks.enter_context(FileLock(str(Path(root) / 'execution.lock'), timeout=1))
        selected, policy = select_release.select(cfg)
        wanted = {identity(e) for e in selected['entries']}
        old = json.loads((previous / 'pool_selection.json').read_text(encoding='utf-8'))
        prior = {identity(e) for e in old['entries']}
        if len(wanted) != len(selected['entries']) or not wanted <= prior:
            raise ValueError('New or changed rows require a fresh full audit')
        removed = []
        for entry in old['entries']:
            if identity(entry) not in wanted:
                path = Path(entry['root']) / entry['arm'] / 'records' / entry['candidate_id'] / 'result.json'
                result = runtime.load_result(path)
                if result['status'] == 'accepted' or runtime.digest(path.read_bytes()) != entry['result_sha256']:
                    raise ValueError('Removed row lacks an unchanged, documented exclusion')
                removed.append({'entry': entry, 'exclusion': result.get('independent_exclusion')})
        retained, ids = [], set()
        for line in source.read_bytes().splitlines(keepends=True):
            row = json.loads(line)
            if identity(row['metadata']['origin']) in wanted:
                retained.append(line)
                ids.add(row['metadata']['scenario_id'])
        if len(retained) != len(wanted) or len(ids) != len(wanted):
            raise ValueError('Incomplete or duplicate subset transport')
        output.mkdir(parents=True)
        target = output / 'retained_conversations_for_audit.jsonl'
        target.write_bytes(b''.join(retained))
        runtime.write_json(output / 'pool_selection.json', selected)
        runtime.write_json(output / 'selection_policy_and_pool.json', policy)
        for name in ['full_census.jsonl', 'semantic_pairs.jsonl', 'lexical_pairs.jsonl']:
            rows = runtime.read_rows(previous / name)
            keep = [r for r in rows if (r['scenario_id'] in ids if name == 'full_census.jsonl' else r['a'] in ids and r['b'] in ids)]
            runtime.write_rows(output / name, keep)
        result = {'train_ready': False, 'mixture_ready': False, 'rows': len(retained),
                  'shortages': policy['shortages'], 'phase_counts': policy['selected_phase_counts'],
                  'prior_input_sha256': original_sha, 'input_sha256': runtime.digest(target.read_bytes()),
                  'original_token_audit_sha256': runtime.digest((previous / 'token_mask_audit.json').read_bytes()),
                  'inherited_mask_check': 'Every retained row is byte-identical to a row that passed the original untruncated8192-token audit.',
                  'removed': removed, 'helper_sha256': runtime.digest(Path(__file__).read_bytes()),
                  'scope': 'Incomplete effective pool after recorded exclusions, not a release or independent certification of every answer. Pair lists retain original scores and are filtered by identity.'}
        runtime.write_json(output / 'closed_pool.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['previous', 'config', 'output']:
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    result = close(args.previous, args.config, args.output)
    print({k: v for k, v in result.items() if k != 'removed'})
