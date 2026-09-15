# ABOUTME: Resume only pre-dispatch reservation-ceiling failures, retaining every paid checkpoint and failed receipt.
# ABOUTME: No calls, budget changes or quality overrides; dry-run by default and root-locked on apply.
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import re

from filelock import FileLock

from scratch.dataset_refresh import run as runtime, per_row
from scratch.dataset_refresh.recover_scenario_json import within, byte_sha
from scratch.dataset_refresh.retry_technical_review import checked

ERROR = re.compile(r'Budget stop: \$(\d+\.\d{3}) exposed, \$(\d+\.\d{3}) next reservation, ceiling \$(\d+(?:\.\d+)?)')


def plan_row(root, terminal, entries):
    root, terminal = Path(root).resolve(), Path(terminal).resolve()
    row = within(terminal.parent, root)
    hashes = {}
    result = checked(terminal, hashes)
    arm, cid = row.parent.parent.name, row.name
    match = ERROR.fullmatch(result.get('error', ''))
    if (terminal.name != 'result.json' or row.parent.name != 'records'
            or result.get('candidate_id') != cid or result.get('status') != 'failed'
            or result.get('error_type') != 'BudgetStop' or not match):
        raise ValueError('Only the exact pre-dispatch reservation-ceiling BudgetStop qualifies')
    exposed, reserve, ceiling = map(float, match.groups())
    if not (0 <= exposed <= 250 and reserve >= 0 and 0 < ceiling < 250):
        raise ValueError('Only a lower batch ceiling inside the shared250 cap can be resumed')
    if (row / 'independent_exclusion.json').exists():
        raise ValueError('Never resume a substantively excluded terminal')
    cfg = runtime.validate_arm(root, arm)
    if cfg.get('per_row_regime') is not True:
        raise ValueError('Require the frozen per-row regime')
    per_row.assert_models(cfg)
    hashes[str(root / arm / 'config.json')] = byte_sha((root / arm / 'config.json').read_bytes())
    meta = checked(root / 'run_meta.json', hashes)
    for name, field in [('run.py', 'code_sha256'), ('per_row.py', 'code_per_row_sha256'),
                        ('reviewer_probe.py', 'critic_validator_sha256')]:
        if meta.get(field) != byte_sha(Path(__file__).with_name(name).read_bytes()):
            raise ValueError('Frozen implementation differs: ' + name)
    candidates = [c for c in runtime.read_rows(root / arm / 'candidates.jsonl') if c['candidate_id'] == cid]
    if len(candidates) != 1 or checked(row / 'identity.json', hashes) != {
            'candidate_sha256': runtime.digest(candidates[0]), 'config_sha256': runtime.digest(cfg)}:
        raise ValueError('Candidate/config identity differs from frozen inputs')
    if result.get('trait_id') != candidates[0]['trait_id']:
        raise ValueError('Failed terminal trait differs from candidate')
    calls = [e for e in entries if e.get('run_root') and Path(e['run_root']).resolve() == root
             and e.get('arm') == arm and e.get('candidate_id') == cid]
    allowed = {'scenario', 'preflight', *(s['name'] for s in cfg['response_stages'])}
    evidence = []
    budget = Path(meta['budget_root']).resolve()
    for entry in calls:
        cost = entry.get('api_reported_cost_usd')
        if (entry.get('status') != 'settled' or type(cost) not in (int, float)
                or not math.isfinite(cost) or cost < 0):
            raise ValueError('All prior candidate calls must be settled with known cost')
        if entry.get('model') not in {m['model'] for m in cfg['models'].values()}:
            raise ValueError('Unconfigured historical candidate model')
        stage = entry.get('stage', '')
        if stage not in allowed and not re.fullmatch(r'(grounding|review|repair)_\d+', stage):
            raise ValueError('Unsupported candidate stage; no external or unknown calls')
        checkpoint = within(row / (stage + '.json'), row)
        checked(checkpoint, hashes)
        raw_path = within(budget / 'raw_calls' / f'{entry["call_id"]:06d}.json', budget)
        raw = json.loads(raw_path.read_bytes())
        hashes[str(raw_path)] = byte_sha(raw_path.read_bytes())
        if (raw.get('accounting') != entry or runtime.digest(raw['request']) != entry['request_sha256']
                or raw.get('response', {}).get('finish_reason') != 'stop'):
            raise ValueError('Prior physical response/request/accounting is incomplete or changed')
        evidence.append({'call_id': entry['call_id'], 'stage': stage, 'request_sha256': entry['request_sha256'],
                         'raw_call_sha256': hashes[str(raw_path)], 'checkpoint_sha256': hashes[str(checkpoint)]})
    # Even archived formatting failures and their original receipts must remain intact.
    for path in row.rglob('*.json'):
        if path.name.endswith('.receipt.json'):
            if not path.with_name(path.name.removesuffix('.receipt.json') + '.json').exists():
                raise ValueError('Orphan receipt in candidate evidence')
        else:
            checked(within(path, row), hashes)
    number = 0
    while (row / f'budget_stop_resume_{number}.json').exists():
        number += 1
    archive = within(row / 'recovered_failures' / f'budget_stop_{number}', row)
    note = within(row / f'budget_stop_resume_{number}.json', row)
    if archive.exists() or note.with_suffix('.receipt.json').exists():
        raise ValueError('An incomplete recovery already exists')
    return {'root': str(root), 'row': str(row), 'arm': arm, 'candidate_id': cid,
            'archive': str(archive), 'note': str(note), 'original_error': result['error'],
            'previous_ceiling_usd': ceiling, 'reported_exposure_usd': exposed,
            'undispatched_reservation_usd': reserve, 'physical_calls': evidence,
            'candidate_calls_sha256': runtime.digest(calls), 'bound_file_sha256': hashes,
            'script_sha256': byte_sha(Path(__file__).read_bytes()),
            'action': 'Archive only the failed terminal; reuse all checkpoints and continue normal frozen checks under the same ledger. No API call or budget change.'}


def apply_plan(plan):
    root, row = Path(plan['root']).resolve(), within(plan['row'], plan['root'])
    archive, note = within(plan['archive'], row), within(plan['note'], row)
    if archive.exists() or note.exists():
        raise ValueError('Recovery already started')
    for path, expected in plan['bound_file_sha256'].items():
        if byte_sha(Path(path).read_bytes()) != expected:
            raise ValueError('Bound evidence changed after planning: ' + path)
    meta = runtime.load_checkpoint(root / 'run_meta.json')
    entries = json.loads((Path(meta['budget_root']) / 'spend.json').read_text(encoding='utf-8'))
    calls = [e for e in entries if e.get('run_root') and Path(e['run_root']).resolve() == root
             and e.get('arm') == plan['arm'] and e.get('candidate_id') == plan['candidate_id']]
    if runtime.digest(calls) != plan['candidate_calls_sha256']:
        raise ValueError('Candidate accounting changed after planning')
    archive.mkdir(parents=True)
    runtime.save_checkpoint(note, plan)
    terminal = row / 'result.json'
    for path in (terminal.with_suffix('.receipt.json'), terminal):
        path.rename(within(archive / path.name, row))
    runtime.load_checkpoint(archive / 'result.json')


def inspect(root, apply=False):
    root = Path(root).resolve()
    meta = runtime.load_checkpoint(root / 'run_meta.json')
    budget = Path(meta['budget_root']).resolve()
    with FileLock(str(budget / 'spend.lock'), timeout=1) if apply else nullcontext():
        entries = json.loads((budget / 'spend.json').read_text(encoding='utf-8'))
    report = {'root': str(root), 'mode': 'apply' if apply else 'dry_run', 'recoverable': [], 'refused': []}
    for path in sorted(root.glob('*/records/*/result.json')):
        result = runtime.load_checkpoint(path)
        if result.get('status') != 'failed' or result.get('error_type') != 'BudgetStop':
            continue
        try:
            with FileLock(str(path.parent / 'independent_repair.lock'), timeout=1) if apply else nullcontext():
                plan = plan_row(root, path, entries)
                if apply:
                    apply_plan(plan)
            report['recoverable'].append(plan)
        except (ValueError, KeyError, FileNotFoundError, runtime.BudgetStop) as exc:
            report['refused'].append({'path': str(path), 'reason': str(exc)})
    return report


def recover(root, apply=False):
    with FileLock(str(Path(root) / 'execution.lock'), timeout=1) if apply else nullcontext():
        return inspect(root, apply)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', help='Optional report outside immutable source root')
    args = parser.parse_args()
    if args.output and Path(args.output).resolve().is_relative_to(Path(args.root).resolve()):
        parser.error('--output must be outside the source root')
    report = recover(args.root, args.apply)
    if args.output:
        runtime.write_json(Path(args.output), report)
    print(json.dumps({'root': report['root'], 'mode': report['mode'],
                      'recoverable': len(report['recoverable']), 'refused': report['refused']}, indent=2))
