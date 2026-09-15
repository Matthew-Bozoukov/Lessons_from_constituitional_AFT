# ABOUTME: Plan one provenance-bound retry of a technically invalid grounding review without changing answers.
# ABOUTME: Never retries valid adverse verdicts, unknown physical outcomes, or an already retried stage.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from filelock import FileLock

from scratch.dataset_refresh import run as runtime
from scratch.dataset_refresh.recover_scenario_json import byte_sha, within
from scratch.dataset_refresh.reviewer_probe import validate_verdict


def checked(path, hashes):
    path = Path(path).resolve()
    value = runtime.load_checkpoint(path)
    for file in (path, path.with_suffix('.receipt.json')):
        hashes[str(file)] = byte_sha(file.read_bytes())
    return value


def plan_row(root, terminal, entries, allow_truncated=False):
    root, terminal = Path(root).resolve(), Path(terminal).resolve()
    row = within(terminal.parent, root)
    hashes = {}
    result = checked(terminal, hashes)
    arm, cid = row.parent.parent.name, row.name
    if row.parent.name != 'records' or result.get('candidate_id') != cid:
        raise ValueError('Unexpected row layout/identity')
    if result.get('status') != 'failed' or result.get('error_type') not in ('ValueError', 'JSONDecodeError'):
        raise ValueError('Only failed parser/schema terminals are eligible, never quality rejections')
    if (row / 'independent_exclusion.json').exists():
        raise ValueError('Independently excluded row cannot be retried')
    cfg_path = root / arm / 'config.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    hashes[str(cfg_path)] = byte_sha(cfg_path.read_bytes())
    identity = checked(row / 'identity.json', hashes)
    if identity.get('config_sha256') != runtime.digest(cfg):
        raise ValueError('Changed frozen config')
    meta = checked(root / 'run_meta.json', hashes)
    validator_path = Path(__file__).with_name('reviewer_probe.py')
    if meta.get('critic_validator_sha256') != runtime.digest(validator_path.read_bytes()):
        raise ValueError('Frozen critic validator no longer matches')
    hashes[str(validator_path.resolve())] = byte_sha(validator_path.read_bytes())
    all_calls = [e for e in entries if e.get('run_root') and Path(e['run_root']).resolve() == root
                 and e.get('arm') == arm and e.get('candidate_id') == cid]
    calls = [e for e in all_calls if re.fullmatch(r'grounding_\d+', e.get('stage', ''))]
    if not calls:
        raise ValueError('No physical grounding review found')
    attempt = max(int(e['stage'].split('_')[1]) for e in calls)
    stage = f'grounding_{attempt}'
    matches = [e for e in calls if e['stage'] == stage]
    if len(matches) != 1:
        raise ValueError('Exactly one original call at this stage required; no second technical retry')
    if attempt > int(cfg.get('max_response_repairs', 1)):
        raise ValueError('Grounding attempt exceeds frozen response repair limit')
    archive = within(row / 'recovered_failures' / 'technical_retry_0' / stage, row)
    note = row / f'technical_retry_{stage}.json'
    if archive.exists() or note.exists():
        raise ValueError('One technical retry for this stage already claimed')
    if any((row / name).exists() for name in (f'review_{attempt}.json', f'answer_{attempt}.json', f'repair_{attempt + 1}.json')):
        raise ValueError('Downstream stage exists; failure was not this grounding validation')
    entry = matches[0]
    if entry != max(all_calls, key=lambda e: e['call_id']):
        raise ValueError('Grounding was not the last physical call for this row')
    cost = entry.get('api_reported_cost_usd')
    if entry.get('status') != 'settled' or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Unknown outcome/billing is not eligible for automatic technical retry')
    budget = Path(meta.get('budget_root', root / 'budget')).resolve()
    raw_path = within(budget / 'raw_calls' / f'{entry["call_id"]:06d}.json', budget)
    raw = json.loads(raw_path.read_bytes())
    hashes[str(raw_path)] = byte_sha(raw_path.read_bytes())
    if raw.get('accounting') != entry or runtime.digest(raw['request']) != entry['request_sha256']:
        raise ValueError('Raw request/accounting does not match shared ledger')
    finish = raw.get('response', {}).get('finish_reason')
    if finish != 'stop' and not (allow_truncated and finish == 'length'):
        raise ValueError('Only complete stop responses qualify; known-cost length requires --allow-truncated')

    # Reconstruct the exact answer that frozen execution will reload on resume.
    scenario = checked(row / 'scenario.json', hashes)
    if not runtime.acceptance(checked(row / 'preflight.json', hashes), cfg['preflight']):
        raise ValueError('Scenario did not pass frozen eligibility')
    reconstructed = {k: scenario[k] for k in ('system', 'user')}
    for spec in cfg['response_stages']:
        reconstructed.update(checked(row / (spec['name'] + '.json'), hashes))
    for number in range(1, attempt + 1):
        # Every earlier valid verdict is retained, including adverse findings.
        earlier = {k: reconstructed[k] for k in ('system', 'user', 'reasoning', 'response')}
        earlier['final'] = earlier.pop('response')
        validate_verdict(checked(row / f'grounding_{number - 1}.json', hashes), earlier)
        checked(row / f'review_{number - 1}.json', hashes)
        previous_answer = checked(row / f'answer_{number - 1}.json', hashes)
        if previous_answer != {k: reconstructed[k] for k in ('system', 'user', 'reasoning', 'response')}:
            raise ValueError('Earlier saved answer does not match reconstruction')
        reconstructed.update(checked(row / f'repair_{number}.json', hashes))
    actual = {k: reconstructed[k] for k in ('system', 'user', 'reasoning', 'response')}
    if actual != {k: result['record'][k] for k in actual}:
        raise ValueError('Failed terminal answer differs from resumable saved answer')
    narrow = {**actual}
    narrow['final'] = narrow.pop('response')
    spec = cfg['grounding_review']
    expected = runtime.request_options(cfg['models'][spec['model']])
    if expected['model'] != 'anthropic/claude-sonnet-5':
        raise ValueError('Only the already frozen Sonnet reviewer may be retried')
    expected['messages'] = [{'role': role, 'content': runtime.render(spec['prompts'][role],
                             {'conversation_json': json.dumps(narrow, ensure_ascii=False)})}
                            for role in ('system', 'user')]
    if expected != raw['request']:
        raise ValueError('Original grounding request does not match exact frozen answer/config')
    checkpoint = row / (stage + '.json')
    parsed = None
    physical_length_error = (finish == 'length' and result['error_type'] == 'ValueError'
                             and result['error'] == 'Excluded incomplete/provider-filtered output: length')
    try:
        parsed = runtime._parse_json(raw['response']['content'])
        validate_verdict(parsed, narrow)
    except (ValueError, TypeError, KeyError) as exc:
        # Only the exact saved error proves this failure came from this validator.
        if not physical_length_error and (type(exc).__name__ != result['error_type'] or str(exc) != result['error']):
            raise ValueError('Exact original parser/validator error was not reproduced') from exc
        error = {'type': result['error_type'], 'message': result['error'],
                 'raw_verdict_validation_error': {'type': type(exc).__name__, 'message': str(exc)}}
    else:
        raise ValueError('Valid verdict must never be retried, whether favorable or adverse')
    if result['error_type'] == 'JSONDecodeError' or physical_length_error:
        if checkpoint.exists() or checkpoint.with_suffix('.receipt.json').exists():
            raise ValueError('Parser failure unexpectedly has a grounding checkpoint')
    elif checked(checkpoint, hashes) != parsed:
        raise ValueError('Malformed saved checkpoint differs from exact provider response')
    return {'root': str(root), 'arm': arm, 'candidate_id': cid, 'row': str(row), 'stage': stage,
            'call_id': entry['call_id'], 'request_sha256': entry['request_sha256'],
            'ledger_entry_sha256': runtime.digest(entry), 'raw_call_sha256': hashes[str(raw_path)],
            'conversation_sha256': runtime.digest(narrow), 'original_error': error,
            'finish_reason': finish, 'api_reported_cost_usd': cost,
            'has_grounding_checkpoint': checkpoint.exists(), 'archive': str(archive),
            'bound_file_sha256': hashes,
            'effect': 'One fresh grounding call on the unchanged answer and frozen prompt; valid adverse outcomes remain binding.'}


def apply_plan(plan):
    row = Path(plan['row']).resolve()
    archive = within(plan['archive'], row)
    note = within(row / f'technical_retry_{plan["stage"]}.json', row)
    if archive.exists() or note.exists():
        raise ValueError('Technical retry already claimed')
    for name, expected in plan['bound_file_sha256'].items():
        if byte_sha(Path(name).read_bytes()) != expected:
            raise ValueError('Bound evidence changed after planning: ' + name)
    # Claim durably before moving anything. A partial crash refuses further automatic mutation.
    runtime.save_checkpoint(note, {**plan, 'method': 'one_technical_grounding_retry'})
    archive.mkdir(parents=True)
    names = [plan['stage'] + '.json', plan['stage'] + '.receipt.json'] if plan['has_grounding_checkpoint'] else []
    names += ['result.receipt.json', 'result.json']  # Terminal removed last, after archive is complete.
    for name in names:
        source, dest = within(row / name, row), within(archive / name, row)
        source.rename(dest)
    runtime.load_checkpoint(archive / 'result.json')
    if plan['has_grounding_checkpoint']:
        runtime.load_checkpoint(archive / (plan['stage'] + '.json'))


def inspect(root, apply=False, allow_truncated=False):
    root = Path(root).resolve()
    meta = runtime.load_checkpoint(root / 'run_meta.json')
    entries = json.loads((Path(meta.get('budget_root', root / 'budget')) / 'spend.json').read_text(encoding='utf-8'))
    report = {'root': str(root), 'mode': 'apply' if apply else 'dry_run', 'retryable': [], 'refused': []}
    for terminal in sorted(root.glob('*/records/*/result.json')):
        saved = runtime.load_checkpoint(terminal)
        if saved.get('status') != 'failed' or saved.get('error_type') not in ('ValueError', 'JSONDecodeError'):
            continue
        try:
            plan = plan_row(root, terminal, entries, allow_truncated)
            if apply:
                apply_plan(plan)
            report['retryable'].append(plan)
        except (ValueError, KeyError, TypeError, FileNotFoundError, runtime.BudgetStop) as exc:
            report['refused'].append({'path': str(terminal), 'reason': str(exc)})
    return report


def recover(root, apply=False, allow_truncated=False):
    if not apply:
        return inspect(root, allow_truncated=allow_truncated)
    with FileLock(str(Path(root) / 'execution.lock'), timeout=1):
        return inspect(root, apply=True, allow_truncated=allow_truncated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--allow-truncated', action='store_true', help='Explicitly include known-cost length-finished malformed JSON, preserving raw evidence.')
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.apply, args.allow_truncated), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
