# ABOUTME: Frozen20-case Luna/Terra critic comparison using unchanged blind production prompts.
# ABOUTME: One call per model/case, persistent combined$2 probe allowance, original shared$250 ledger.
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from filelock import FileLock

from scratch.dataset_refresh import run as runtime
from scratch.dataset_refresh.reviewer_probe import validate_verdict

MODELS = ('openai/gpt-5.6-luna', 'openai/gpt-5.6-terra')
ARM = 'independent_luna_terra_probe'
ALLOCATION = 'independent_luna_terra_probe_allocation.json'
LOW, NON = 'da-lowstakes-refresh', 'nonmoral-advice'
# Counterbalanced across labels; full conversation and independent findings are frozen separately.
SELECTION = [
    (LOW, 't1_006_v0', 'reject'), (LOW, 't3_000_v0', 'pass'),
    (LOW, 't1_007_v0', 'reject'), (LOW, 't4_000_v0', 'pass'),
    (NON, 't5_000_v0', 'reject'), (NON, 't1_002_v0', 'pass'),
    (NON, 't9_000_v0', 'reject'), (NON, 't3_000_v0', 'pass'),
    (NON, 't9_001_v0', 'reject'), (NON, 't5_001_v0', 'pass'),
    (NON, 't6_001_v0', 'reject'), (NON, 't7_001_v0', 'pass'),
    (NON, 't4_000_v0', 'reject'), (NON, 't2_000_v0', 'pass'),
    (NON, 't7_000_v0', 'reject'), (LOW, 't7_000_v0', 'pass'),
    (LOW, 't3_008_v0', 'reject'), (LOW, 't8_000_v0', 'pass'),
    (LOW, 't4_003_v0', 'reject'), (LOW, 't2_002_v0', 'pass'),
]


def messages(manifest, case):
    # None of expected decision, independent annotation, source identity, or target enters the request.
    values = {'conversation_json': json.dumps(case['conversation'], ensure_ascii=False)}
    return [{'role': role, 'content': runtime.render(manifest['prompts'][role], values)}
            for role in ('system', 'user')]


def prepare(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if (output / 'manifest.json').exists():
        raise ValueError('Already frozen; do not recreate or tune this probe')
    meta = runtime.load_checkpoint(source / 'run_meta.json')
    budget = Path(meta['budget_root']).resolve()
    if budget != Path('output/2026-09-14_dataset_refresh/budget').resolve():
        raise ValueError('Original shared budget required')
    reports, cfgs = {}, {}
    for arm in (LOW, NON):
        path = source / arm / 'independent_all.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        reports[arm] = {r['candidate_id']: r for r in data.get('rows', data.get('review_events', []))}
        cfgs[arm] = json.loads((source / arm / 'config.json').read_text(encoding='utf-8'))
    prompts = cfgs[LOW]['grounding_review']['prompts']
    if prompts != cfgs[NON]['grounding_review']['prompts']:
        raise ValueError('Comparison requires identical existing blind prompts across both arms')
    cases = []
    for arm, cid, expected in SELECTION:
        path = source / arm / 'records' / cid / 'result.json'
        row = runtime.load_checkpoint(path)
        note = reports[arm][cid]
        if note['result_sha256'] != runtime.digest(path.read_bytes()):
            raise ValueError('Independent assessment no longer matches terminal: ' + str(path))
        if note['independent_disposition'].lower() != expected:
            raise ValueError('Unexpected independent label: ' + str(path))
        conversation = {k: row['record']['response' if k == 'final' else k]
                        for k in ('system', 'user', 'reasoning', 'final')}
        if not all(isinstance(v, str) and v.strip() for v in conversation.values()):
            raise ValueError('Missing full conversation: ' + str(path))
        cases.append({'id': arm + '__' + cid, 'expected': expected, 'conversation': conversation,
                      'source_path': str(path), 'source_sha256': runtime.digest(path.read_bytes()),
                      'independent_annotation': note})
    manifest = {'schema': 1, 'models': list(MODELS), 'cases': cases, 'prompts': prompts,
                'source_root': str(source), 'budget_root': str(budget), 'probe_root': str(output),
                'max_calls': 40, 'allowance_usd': 2, 'max_tokens': 6000,
                'reasoning': 'provider_default_no_override', 'code_sha256': runtime.digest(Path(__file__).read_bytes()),
                'validator_sha256': runtime.digest(Path(__file__).with_name('reviewer_probe.py').read_bytes()),
                'runtime_sha256': runtime.digest(Path(runtime.__file__).read_bytes()),
                'scope': 'Purposive calibration on10 adjudicated negatives and10 defensible material-content positives. Not held-out accuracy; labels and probe selection may have ambiguities.'}
    with FileLock(str(budget / 'independent_luna_terra_probe.lock'), timeout=1):
        if (budget / ALLOCATION).exists():
            raise ValueError('Shared probe allocation already claimed; cannot reset it with another output')
        runtime.save_checkpoint(budget / ALLOCATION, {'probe_root': str(output), 'allowance_usd': 2,
            'manifest_sha256': runtime.digest(manifest), 'max_calls': 40})
        runtime.save_checkpoint(output / 'manifest.json', manifest)
    return manifest


def reserve_check(entries, model, request):
    probe = [e for e in entries if e.get('arm') == ARM]
    costs = [e['charged_or_reserved_usd'] for e in probe]
    if any(not math.isfinite(v) or v < 0 for v in costs):
        raise runtime.BudgetStop('Invalid probe exposure')
    price = runtime.provider_price(model)
    input_bound = len(json.dumps(request, ensure_ascii=False).encode()) + 2048
    reserve = (1.25 * input_bound * price['in'] + 6000 * price['out']) / 1e6
    if len(probe) >= 40 or sum(costs) + reserve > 2:
        raise runtime.BudgetStop('Combined probe call/$2 reservation ceiling reached')
    return reserve


def run_probe(output, send=None):
    output = Path(output).resolve()
    manifest = runtime.load_checkpoint(output / 'manifest.json')
    budget = Path(manifest['budget_root'])
    with FileLock(str(budget / 'independent_luna_terra_probe.lock'), timeout=1):
        allocation = runtime.load_checkpoint(budget / ALLOCATION)
        if allocation != {'probe_root': str(output), 'allowance_usd': 2,
                          'manifest_sha256': runtime.digest(manifest), 'max_calls': 40}:
            raise ValueError('Probe allocation differs from frozen manifest')
        if (manifest['models'] != list(MODELS) or len(manifest['cases']) != 20
                or sum(c['expected'] == 'reject' for c in manifest['cases']) != 10
                or len({c['id'] for c in manifest['cases']}) != 20):
            raise ValueError('Frozen20-case design changed')
        for key, path in [('code_sha256', Path(__file__)), ('runtime_sha256', Path(runtime.__file__)),
                          ('validator_sha256', Path(__file__).with_name('reviewer_probe.py'))]:
            if manifest[key] != runtime.digest(path.read_bytes()):
                raise ValueError('Frozen code changed: ' + key)
        client = runtime.BudgetClient(budget, 250, set(MODELS), send=send)
        client.local.arm, client.local.run_root, client.local.stage = ARM, str(output), 'blind_grounding_comparison'
        for case in manifest['cases']:
            for model in MODELS:
                path = output / model.replace('/', '__') / (case['id'] + '.json')
                if path.exists():
                    runtime.load_checkpoint(path)
                    continue  # Started/uncertain/schema-invalid calls never silently repeat.
                entries = client.entries()
                prior = [e for e in entries if e.get('arm') == ARM and e.get('model') == model
                         and e.get('candidate_id') == case['id']]
                if prior:
                    raise runtime.BudgetStop('Physical call already exists without result checkpoint')
                request = messages(manifest, case)
                reserve_check(entries, model, request)
                client.local.candidate_id = case['id']
                result = {'id': case['id'], 'model': model, 'expected': case['expected'], 'status': 'started',
                          'request_sha256': runtime.digest(request)}
                runtime.save_checkpoint(path, result)
                try:
                    response = client.chat(model=model, messages=request, temperature=0, max_tokens=6000)
                    result['response'] = asdict(response)
                    verdict = validate_verdict(runtime._parse_json(response.content), case['conversation'])
                    result.update(status='complete', verdict=verdict,
                                  matches_expected=verdict['accepted'] == (case['expected'] == 'pass'))
                except BaseException as exc:
                    result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1500])
                    runtime.save_checkpoint(path, result)
                    if isinstance(exc, (runtime.BudgetStop, KeyboardInterrupt, SystemExit)):
                        raise
                runtime.save_checkpoint(path, result)
                print(json.dumps({k: result[k] for k in ('id', 'model', 'status')}), flush=True)
        return summarize(output)


def summarize(output):
    output = Path(output)
    manifest = runtime.load_checkpoint(output / 'manifest.json')
    entries = json.loads((Path(manifest['budget_root']) / 'spend.json').read_text(encoding='utf-8'))
    probe = [e for e in entries if e.get('arm') == ARM]
    summary = {'scope': manifest['scope'], 'models': {}, 'physical_calls': len(probe),
               'reported_cost_usd': sum(e.get('api_reported_cost_usd') or 0 for e in probe),
               'charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in probe)}
    for model in MODELS:
        results = [runtime.load_checkpoint(p) for p in (output / model.replace('/', '__')).glob('*.json')
                   if not p.name.endswith('.receipt.json')]
        summary['models'][model] = {'attempted_checkpoints': len(results),
            'complete': sum(r['status'] == 'complete' for r in results),
            'caught_negatives': [r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'reject' and not r['verdict']['accepted']],
            'missed_negatives': [r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'reject' and r['verdict']['accepted']],
            'false_rejects': [r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'pass' and not r['verdict']['accepted']],
            'passed_positives': [r['id'] for r in results if r['status'] == 'complete' and r['expected'] == 'pass' and r['verdict']['accepted']],
            'incomplete': [{'id': r['id'], 'error': r.get('error')} for r in results if r['status'] != 'complete']}
    runtime.save_checkpoint(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'summary'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default='output/2026-09-15_dataset_refresh_sonnet_qualified')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(args.output, args.source)
        print(json.dumps({'frozen_cases': len(manifest['cases']), 'models': manifest['models']}))
    elif args.command == 'summary':
        print(json.dumps(summarize(args.output), indent=2))
    else:
        if not args.execute:
            parser.error('Authorized paid dispatch requires --execute')
        print(json.dumps(run_probe(args.output), indent=2))


if __name__ == '__main__':
    main()
