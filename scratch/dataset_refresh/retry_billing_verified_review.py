# ABOUTME: Prepare one retry of a provably empty grounding review after external billing verification.
# ABOUTME: Preserves failed calls, all answer bytes and review limits; never treats missing verdicts as passes.
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
from scratch.dataset_refresh.retry_technical_review import checked, apply_plan
from scratch.dataset_refresh.billing_evidence import validate_billing_call

def plan_row(root, terminal, entries, billing_cache=None):
    root, terminal = Path(root).resolve(), Path(terminal).resolve()
    row = within(terminal.parent, root)
    hashes = {}
    result = checked(terminal, hashes)
    arm, cid = row.parent.parent.name, row.name
    if row.parent.name != 'records' or result.get('candidate_id') != cid:
        raise ValueError('Unexpected row layout/identity')
    if result.get('status') != 'failed' or result.get('error_type') != 'EmptyCompletionError':
        raise ValueError('Only failed empty-completion terminals qualify, never quality rejections')
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
    if entry.get('status') != 'billing_verified_failure' or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Requires externally verified failed-call billing')
    budget = Path(meta['budget_root']).resolve()
    raw_path = within(budget / 'raw_calls' / f"{entry['call_id']:06d}.json", budget)
    raw = json.loads(raw_path.read_bytes())
    hashes[str(raw_path)] = byte_sha(raw_path.read_bytes())
    proof = validate_billing_call(budget, entry, raw, billing_cache)
    if proof is None:
        raise ValueError('Failed billing reconciliation proof is mandatory')
    for path in proof.rglob('*'):
        if path.is_file():
            hashes[str(path.resolve())] = byte_sha(path.read_bytes())
    original = raw['accounting']
    if original.get('exception_type') != result['error_type'] or original.get('error') != result['error']:
        raise ValueError('Terminal error differs from original physical failure')
    if 'response' in raw or (row / (stage + '.json')).exists() or (row / (stage + '.receipt.json')).exists():
        raise ValueError('Any returned verdict/checkpoint blocks empty-review recovery')
    diagnostic = raw['diagnostics']
    choices = diagnostic.get('choices')
    if diagnostic.get('provider_error') or not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('Provider error or ambiguous choice outcome')
    choice = choices[0]
    finish = choice.get('finish_reason')
    if (finish not in ('stop', 'length') or choice.get('dropped_content_chars') != 0
            or any(choice.get(key) for key in ('provider_error', 'refusal', 'refusal_details', 'content_filter_results'))):
        raise ValueError('Only proven empty unfiltered stop/length outcomes qualify')
    expected_error = f"Model {entry['model']} returned empty content (provider {diagnostic['provider']}): finish_reason={finish}"
    if result['error'] != expected_error:
        raise ValueError('Empty-completion error did not exactly reproduce')

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
    error = {'type': result['error_type'], 'message': result['error'],
             'technical_outcome': 'Verified empty critic completion; no verdict exists to accept or reject.'}
    return {'root': str(root), 'arm': arm, 'candidate_id': cid, 'row': str(row), 'stage': stage,
            'call_id': entry['call_id'], 'request_sha256': entry['request_sha256'],
            'ledger_entry_sha256': runtime.digest(entry), 'raw_call_sha256': hashes[str(raw_path)],
            'conversation_sha256': runtime.digest(narrow), 'original_error': error,
            'finish_reason': finish, 'api_reported_cost_usd': cost,
            'has_grounding_checkpoint': checkpoint.exists(), 'archive': str(archive),
            'bound_file_sha256': hashes,
            'billing_reconciliation_archive': str(proof), 'helper_sha256': byte_sha(Path(__file__).read_bytes()),
            'effect': 'One fresh ordinary grounding call on unchanged answer after verified empty completion; no substantive verdict retried.'}


def recover(root, apply=False):
    root = Path(root).resolve()
    def inspect():
        meta = runtime.load_checkpoint(root / 'run_meta.json')
        entries = json.loads((Path(meta['budget_root']) / 'spend.json').read_text(encoding='utf-8'))
        report = {'root': str(root), 'mode': 'apply' if apply else 'dry_run', 'retryable': [], 'refused': []}
        cache = {}
        for terminal in sorted(root.glob('*/records/*/result.json')):
            result = runtime.load_checkpoint(terminal)
            if result.get('status') != 'failed' or result.get('error_type') != 'EmptyCompletionError':
                continue
            try:
                proposal = plan_row(root, terminal, entries, cache)
                if apply:
                    apply_plan(proposal)
                report['retryable'].append(proposal)
            except (ValueError, KeyError, TypeError, FileNotFoundError, runtime.BudgetStop) as exc:
                report['refused'].append({'path': str(terminal), 'reason': str(exc)})
        return report
    if not apply:
        return inspect()
    with FileLock(str(root / 'execution.lock'), timeout=1):
        return inspect()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    result = recover(args.root, args.apply)
    runtime.write_json(Path(args.output), result)
    print(json.dumps({key: len(result[key]) for key in ('retryable', 'refused')}))
