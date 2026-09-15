# ABOUTME: Freeze stratified unaudited cases for one-shot blind Terra grounding flags.
# ABOUTME: Persist a shared $6 suballowance within the original $250 ledger; never exclude terminals.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import threading

from dotenv import load_dotenv
from filelock import FileLock

from scratch.dataset_refresh import run as runtime
from scratch.dataset_refresh.reviewer_probe import validate_verdict
from src.infra.endpoints import openrouter

MODEL = 'openai/gpt-5.6-terra'
ARM = 'independent_terra_flags_sample05'
ALLOWANCE = 'independent_terra_flags_sample05_allocation.json'
LIMIT, MAX_TOKENS, SEED = 6.0, 6000, 2026091505
LOW, NON = 'da-lowstakes-refresh', 'nonmoral-advice'
BUDGET = Path('output/2026-09-14_dataset_refresh/budget')
FIELDS = ('system', 'user', 'reasoning', 'final')


def prior_reviewed(root):
    """Actual annotations and previously selected IDs, never unread pool inventories."""
    ids, provenance = set(), []
    for path in sorted(root.glob('independent*.json')):
        if any(s in path.name for s in ('_snapshot', '_coverage', '.receipt')):
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if '_selection' in path.name:
            entries = data.get('selected', data.get('rows', []))
        else:
            entries = data.get('review_events', data.get('rows', data.get('reviews', [])))
            entries = entries + data.get('extra_rows', [])
        found = {e.get('candidate_id', e.get('id')) for e in entries if isinstance(e, dict)} - {None}
        ids.update(found)
        provenance.append({'path': str(path.resolve()), 'sha256': runtime.digest(path.read_bytes()),
                           'excluded_ids': sorted(found)})
    editing = Path('output/2026-09-15_dataset_refresh_quality_screen/editing_leak_adjudication.json')
    if root.name == NON and editing.exists():
        data = json.loads(editing.read_text(encoding='utf-8'))
        found = {e['candidate_id'] for e in data['rows'] if e['arm'] == NON}
        ids.update(found)
        provenance.append({'path': str(editing.resolve()), 'sha256': runtime.digest(editing.read_bytes()),
                           'excluded_ids': sorted(found)})
    return ids, provenance


def select_pool(pool, excluded, seed=SEED):
    rng, selected, counts = random.Random(seed), [], {}
    for arm in (LOW, NON):
        for t in range(1, 10):
            trait = f't{t}'
            candidates = sorted((p for p in pool if p['arm'] == arm and p['trait_id'] == trait
                                 and p['candidate_id'] not in excluded[arm]), key=lambda p: p['candidate_id'])
            counts[f'{arm}/{trait}'] = {'available': len(candidates), 'selected': min(9, len(candidates)),
                                      'shortage': max(0, 9-len(candidates))}
            selected.extend(rng.sample(candidates, min(9, len(candidates))))
    return selected, counts


def code_files():
    return {'helper.py': Path(__file__), 'runtime.py': Path(runtime.__file__),
            'validator.py': Path(__file__).with_name('reviewer_probe.py'),
            'openrouter.py': Path(openrouter.__file__), 'providers.yaml': openrouter.PROVIDER_PINS_PATH}


def prepare(output, low_source, non_source):
    output = Path(output).resolve()
    budget = BUDGET.resolve()
    with FileLock(str(budget / (ARM + '.lock')), timeout=30):
        if (budget / ALLOWANCE).exists() or (output / 'selection.json').exists():
            raise ValueError('Allocation/selection already frozen; cannot reset sample or allowance')
        roots = {LOW: Path(low_source).resolve() / LOW, NON: Path(non_source).resolve() / NON}
        pool, excluded, reports, configs = [], {}, {}, {}
        for arm, root in roots.items():
            meta = runtime.load_checkpoint(root.parent / 'run_meta.json')
            if Path(meta['budget_root']).resolve() != budget:
                raise ValueError('Original shared budget required')
            excluded[arm], reports[arm] = prior_reviewed(root)
            configs[arm] = json.loads((root / 'config.json').read_text(encoding='utf-8'))
            for path in sorted((root / 'records').glob('*/result.json')):
                raw = path.read_bytes()
                data = json.loads(raw)
                if data.get('status') != 'accepted':
                    continue
                record = data.get('record', {})
                if not all(isinstance(record.get(k), str) and record[k].strip()
                           for k in ('system', 'user', 'reasoning', 'response')):
                    continue
                pool.append({'arm': arm, 'trait_id': data['trait_id'], 'candidate_id': path.parent.name,
                             'source_path': str(path.resolve()), 'source_sha256': runtime.digest(raw)})
        selected, counts = select_pool(pool, excluded)
        selection = {'ABOUTME': ['Frozen pre-content sample and accepted source-pool snapshot.',
                                'Nine per trait per arm when available, without later substitution.'],
                     'created_at': runtime.timestamp(), 'seed': SEED, 'rows': selected,
                     'counts': counts, 'accepted_pool': pool, 'prior_review_reports': reports,
                     'excluded_ids': {a: sorted(v) for a, v in excluded.items()}}
        # This durable selection precedes construction or display of selected conversations.
        runtime.save_checkpoint(output / 'selection.json', selection)
        cases = []
        for row in selected:
            raw = Path(row['source_path']).read_bytes()
            if runtime.digest(raw) != row['source_sha256']:
                raise ValueError('Selected terminal changed before content freeze; no substitutions')
            record = json.loads(raw)['record']
            cases.append({**row, 'id': row['arm'] + '__' + row['candidate_id'],
                          'conversation': {k: record['response' if k == 'final' else k] for k in FIELDS}})
        qualified = json.loads((Path(non_source) / LOW / 'config.json').read_text(encoding='utf-8'))
        prompts = configs[NON]['grounding_review']['prompts']
        if prompts != qualified['grounding_review']['prompts']:
            raise ValueError('Qualified arms have different grounding prompts')
        pin = runtime.provider_pin(MODEL)
        if [p.lower() for p in pin.get('order', [])] != ['openai'] or pin.get('allow_fallbacks') is not False:
            raise ValueError('Strict OpenAI provider pin required')
        hashes = {}
        for name, path in code_files().items():
            dest = output / 'frozen_code' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            hashes[name] = runtime.digest(dest.read_bytes())
        for arm, config in configs.items():
            runtime.save_checkpoint(output / 'source_configs' / (arm + '.json'), config)
        probe = Path('output/2026-09-15_independent_luna_terra_probe')
        for name in ('manifest.json', 'manifest.receipt.json', 'summary.json', 'summary.receipt.json'):
            if (probe / name).exists():
                dest = output / 'prior_probe_receipt' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(probe / name, dest)
        manifest = {'ABOUTME': ['Supplementary Terra flags only, with independent negative adjudication.',
                               'No acceptance guarantee and no mutation of generation terminals.'],
                    'model': MODEL, 'provider_pin': pin, 'price': runtime.provider_price(MODEL),
                    'reasoning': 'provider_default_medium_no_override', 'temperature': 0,
                    'max_tokens': MAX_TOKENS, 'allowance_usd': LIMIT, 'combined_ceiling_usd': 250,
                    'budget_root': str(budget), 'output_root': str(output), 'max_calls': len(cases),
                    'selection_sha256': runtime.digest((output / 'selection.json').read_bytes()),
                    'prompts': prompts, 'cases': cases, 'frozen_code_sha256': hashes,
                    'source_roots': {a: str(p) for a, p in roots.items()}}
        runtime.save_checkpoint(output / 'manifest.json', manifest)
        runtime.save_checkpoint(budget / ALLOWANCE, {'output_root': str(output),
            'manifest_sha256': runtime.digest(manifest), 'allowance_usd': LIMIT, 'entries': []})
        return {'selected': len(cases), 'counts': counts}


def request_messages(manifest, case):
    values = {'conversation_json': json.dumps(case['conversation'], ensure_ascii=False)}
    return [{'role': role, 'content': runtime.render(manifest['prompts'][role], values)}
            for role in ('system', 'user')]


def reserve_amount(messages, price):
    bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 2048
    return (1.25 * bound * price['in'] + MAX_TOKENS * price['out']) / 1e6


def admit(allocation, cid, reserve, max_calls):
    entries = allocation['entries']
    if any(e['id'] == cid for e in entries):
        raise runtime.BudgetStop('Already started; no retry')
    costs = [e['charged_or_reserved_usd'] for e in entries]
    if not math.isfinite(reserve) or reserve <= 0 or any(not math.isfinite(c) or c < 0 for c in costs):
        raise runtime.BudgetStop('Invalid local reservation/accounting')
    if len(entries) >= max_calls or sum(costs) + reserve > LIMIT:
        raise runtime.BudgetStop('Supplementary $6/call ceiling reached')
    entries.append({'id': cid, 'status': 'started', 'charged_or_reserved_usd': reserve})


def run_audit(output, workers=12, send=None):
    output = Path(output).resolve()
    manifest = runtime.load_checkpoint(output / 'manifest.json')
    budget = Path(manifest['budget_root'])
    if budget != BUDGET.resolve() or manifest['model'] != MODEL or manifest['max_tokens'] != MAX_TOKENS:
        raise ValueError('Frozen model/budget changed')
    with FileLock(str(output / 'execution.lock'), timeout=1):
        for name, path in code_files().items():
            if runtime.digest(path.read_bytes()) != manifest['frozen_code_sha256'][name]:
                raise ValueError('Frozen implementation changed: ' + name)
        if runtime.digest((output / 'selection.json').read_bytes()) != manifest['selection_sha256']:
            raise ValueError('Frozen selection changed')
        allocation = runtime.load_checkpoint(budget / ALLOWANCE)
        if (allocation['manifest_sha256'] != runtime.digest(manifest) or allocation['output_root'] != str(output)
                or allocation['allowance_usd'] != LIMIT):
            raise ValueError('Allocation mismatch')
        client = runtime.BudgetClient(budget, 250, {MODEL}, send=send)
        stop = threading.Event()

        def work(case):
            path = output / 'results' / (case['id'] + '.json')
            if path.exists() or stop.is_set():
                return
            messages = request_messages(manifest, case)
            result = {'id': case['id'], 'source_sha256': case['source_sha256'], 'status': 'started',
                      'request_sha256': runtime.digest(messages)}
            with FileLock(str(budget / (ARM + '.lock')), timeout=120):
                allocation = runtime.load_checkpoint(budget / ALLOWANCE)
                if any(e['id'] == case['id'] for e in allocation['entries']):
                    return  # Existing started/unknown/invalid attempts are never repeated.
                with client.lock:
                    if any(e.get('arm') == ARM and e.get('candidate_id') == case['id'] for e in client.entries()):
                        raise runtime.BudgetStop('Existing physical call without local receipt')
                admit(allocation, case['id'], reserve_amount(messages, manifest['price']), manifest['max_calls'])
                runtime.save_checkpoint(budget / ALLOWANCE, allocation)
                runtime.save_checkpoint(path, result)
            client.local.arm, client.local.run_root = ARM, str(output)
            client.local.candidate_id, client.local.stage = case['id'], 'supplementary_blind_grounding'
            try:
                response = client.chat(model=MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS)
                result['response'] = asdict(response)
                result['verdict'] = validate_verdict(runtime._parse_json(response.content), case['conversation'])
                result['status'] = 'complete'
            except BaseException as exc:
                result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1500])
                if isinstance(exc, (runtime.BudgetStop, KeyboardInterrupt, SystemExit)):
                    stop.set()
            finally:
                with client.lock:
                    entries = [e for e in client.entries() if e.get('arm') == ARM and e.get('candidate_id') == case['id']]
                with FileLock(str(budget / (ARM + '.lock')), timeout=120):
                    allocation = runtime.load_checkpoint(budget / ALLOWANCE)
                    own = next(e for e in allocation['entries'] if e['id'] == case['id'])
                    if len(entries) == 1:
                        own['charged_or_reserved_usd'] = entries[0]['charged_or_reserved_usd']
                        own['call_id'] = entries[0]['call_id']
                        result['call_id'] = entries[0]['call_id']
                    elif len(entries) > 1:
                        stop.set()
                        raise runtime.BudgetStop('Multiple physical calls for a sampled case')
                    own['status'] = result['status']
                    runtime.save_checkpoint(budget / ALLOWANCE, allocation)
                runtime.save_checkpoint(path, result)
            print(json.dumps({'id': case['id'], 'status': result['status'],
                              'negative_flag': result.get('verdict', {}).get('accepted') is False}), flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, c) for c in manifest['cases']]
            for future in as_completed(futures):
                try:
                    future.result()
                except runtime.BudgetStop:
                    stop.set()
        return summarize(output)


def summarize(output):
    output = Path(output).resolve()
    manifest = runtime.load_checkpoint(output / 'manifest.json')
    with FileLock(str(Path(manifest['budget_root']) / 'spend.lock'), timeout=120):
        all_entries = json.loads((Path(manifest['budget_root']) / 'spend.json').read_text(encoding='utf-8'))
    entries = [e for e in all_entries if e.get('arm') == ARM]
    results = [runtime.load_checkpoint(p) for p in (output / 'results').glob('*.json')
               if not p.name.endswith('.receipt.json')]
    summary = {'sampled': len(manifest['cases']), 'physical_calls': len(entries),
               'complete': sum(r['status'] == 'complete' for r in results),
               'negative_flags': [r['id'] for r in results if r.get('verdict', {}).get('accepted') is False],
               'incomplete': [{'id': r['id'], 'status': r['status'], 'error': r.get('error')} for r in results if r['status'] != 'complete'],
               'not_started': len(manifest['cases'])-len(results),
               'reported_cost_usd': sum(e.get('api_reported_cost_usd') or 0 for e in entries),
               'charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in entries),
               'combined_charged_or_reserved_usd': sum(e['charged_or_reserved_usd'] for e in all_entries),
               'interpretation': 'Flags only. Negative findings require independent adjudication; passes are not correctness guarantees.'}
    audit = output / 'audit'
    hashes = {}
    for entry in entries:
        name = f"{entry['call_id']:06d}.json"
        source = Path(manifest['budget_root']) / 'raw_calls' / name
        raw = source.read_bytes()
        dest = audit / 'raw_calls' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        hashes[name] = runtime.digest(raw)
    runtime.save_checkpoint(audit / 'ledger_slice.json', {'entries': entries,
        'authoritative_ledger': str(Path(manifest['budget_root']) / 'spend.json'),
        'snapshot_at': runtime.timestamp(), 'raw_call_sha256': hashes})
    runtime.save_checkpoint(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'summary'])
    parser.add_argument('--output', default='output/2026-09-15_supplementary_terra_flags_sample05')
    parser.add_argument('--low-source', default='output/2026-09-15_dataset_refresh_diverse')
    parser.add_argument('--non-source', default='output/2026-09-15_dataset_refresh_sonnet_qualified')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.output, args.low_source, args.non_source), indent=2))
    elif args.command == 'summary':
        print(json.dumps(summarize(args.output), indent=2))
    else:
        if not args.execute:
            parser.error('Paid dispatch requires --execute')
        load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
        print(json.dumps(run_audit(args.output, args.workers), indent=2))


if __name__ == '__main__':
    main()
