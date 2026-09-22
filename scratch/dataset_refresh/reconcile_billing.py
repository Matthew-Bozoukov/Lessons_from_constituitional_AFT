# ABOUTME: Reconcile only independently verified failed-call billing, without changing raw outputs or accepting content.
# ABOUTME: Default dry run; apply archives the old ledger and exact API evidence under the shared spend lock.
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from filelock import FileLock
from scratch.dataset_refresh import run as runtime


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def proof_files(proof):
    proof = Path(proof).absolute()
    if any(p.is_symlink() or p.is_junction() for p in (proof, *proof.parents)):
        raise ValueError('Proof must not traverse links')
    expected = read(proof / 'full_file_manifest.json')
    for name, sha in expected.items():
        path = proof / name
        if Path(name).name != name or path.is_symlink() or runtime.digest(path.read_bytes()) != sha:
            raise ValueError('Changed or unsafe billing evidence')
    for name in ('inputs.json', 'models_catalogue.response.json'):
        if name not in expected:
            raise ValueError('Required billing evidence is not hash-bound')
    return {name: (proof / name).read_bytes() for name in expected} | {'full_file_manifest.json': (proof / 'full_file_manifest.json').read_bytes()}


def plan(budget, proof):
    budget, proof = Path(budget), Path(proof)
    evidence = proof_files(proof)
    inputs = json.loads(evidence['inputs.json'])['entries']
    catalogue = {m['id']: m.get('canonical_slug') for m in json.loads(evidence['models_catalogue.response.json'])['data']}
    entries = read(budget / 'spend.json')
    changes, seen = [], set()
    for item in inputs:
        cid = item['call_id']
        if type(cid) is not int or cid < 0 or cid >= len(entries) or cid in seen:
            raise ValueError('Invalid or duplicate call ID')
        seen.add(cid)
        entry = entries[cid]
        if entry['call_id'] != cid or entry['status'] != 'uncertain_failure' or runtime.digest(entry) != item['ledger_entry_sha256']:
            raise ValueError('Ledger entry changed or is not an uncertain failure')
        raw_bytes = (budget / 'raw_calls' / f'{cid:06d}.json').read_bytes()
        raw = json.loads(raw_bytes)
        if (runtime.digest(raw_bytes) != item['raw_sha256'] or raw['accounting'] != entry
                or runtime.digest(raw['request']) != entry['request_sha256']):
            raise ValueError('Raw request/accounting is not bound to this ledger entry')
        diag = raw['diagnostics']
        if diag != item['diagnostics'] or diag['generation_id'] != item['generation_id']:
            raise ValueError('Diagnostic identity changed')
        name = f'{cid:06d}.response.json'
        verification = json.loads(evidence[f'{cid:06d}.verification.json'])
        if name not in evidence or verification['http_status'] != 200 or runtime.digest(evidence[name]) != verification['response_sha256']:
            raise ValueError('Billing response is missing, changed or unsuccessful')
        data = json.loads(evidence[name])['data']
        usage = diag['usage']
        if (data['id'] != diag['generation_id'] or catalogue.get(entry['model']) != data['model']
                or data['provider_name'] != diag['provider']
                or data['native_tokens_prompt'] != usage['prompt_tokens']
                or data['native_tokens_completion'] != usage['completion_tokens']):
            raise ValueError('Billing identity/provider/native usage mismatch')
        values = [data.get('total_cost'), data.get('upstream_inference_cost'), usage.get('cost'),
                  (usage.get('cost_details') or {}).get('upstream_inference_cost')]
        if not finite(values[0]) or any(v is not None and not finite(v) for v in values):
            raise ValueError('Invalid or missing customer cost')
        charge = max(v for v in values if v is not None)
        if not finite(entry['charged_or_reserved_usd']) or charge > entry['charged_or_reserved_usd']:
            raise ValueError('Billing exceeds the existing bound; retain and investigate')
        changes.append({'call_id': cid, 'before': entry, 'raw_sha256': item['raw_sha256'],
                        'generation_id': data['id'], 'customer_cost_usd': data['total_cost'],
                        'conservative_charge_usd': charge, 'release_usd': entry['charged_or_reserved_usd'] - charge})
    return {'changes': changes, 'release_usd': sum(c['release_usd'] for c in changes),
            'proof_sha256': runtime.digest(evidence['full_file_manifest.json']),
            'interpretation': 'Verified billing only. Failed content remains failed and raw accounting remains original.'}


def reconcile(budget, proof, *, apply=False):
    budget = Path(budget).resolve()
    if not apply:
        return plan(budget, proof)
    with FileLock(str(budget / 'spend.lock'), timeout=120):
        proposal = plan(budget, proof)
        evidence = proof_files(proof)
        if runtime.digest(evidence['full_file_manifest.json']) != proposal['proof_sha256']:
            raise ValueError('Proof changed during validation')
        archive = budget / 'billing_reconciliations' / proposal['proof_sha256']
        if archive.exists():
            raise ValueError('Reconciliation already attempted; inspect preserved archive')
        archive.mkdir(parents=True)
        ledger_bytes = (budget / 'spend.json').read_bytes()
        (archive / 'spend_before.json').write_bytes(ledger_bytes)
        for name, content in evidence.items():
            (archive / name).write_bytes(content)
        runtime.write_json(archive / 'plan.json', proposal)
        (archive / 'raw_calls').mkdir()
        for c in proposal['changes']:
            raw_bytes = (budget / 'raw_calls' / f"{c['call_id']:06d}.json").read_bytes()
            if runtime.digest(raw_bytes) != c['raw_sha256']:
                raise ValueError('Raw evidence changed during reconciliation')
            (archive / 'raw_calls' / f"{c['call_id']:06d}.json").write_bytes(raw_bytes)
        entries = json.loads(ledger_bytes)
        for c in proposal['changes']:
            entry = entries[c['call_id']]
            if entry != c['before']:
                raise ValueError('Ledger changed under lock')
            entry.update(status='billing_verified_failure', charged_or_reserved_usd=c['conservative_charge_usd'],
                         api_reported_cost_usd=c['customer_cost_usd'], billing_reconciliation_sha256=proposal['proof_sha256'],
                         billing_generation_id=c['generation_id'])
        runtime.write_json(budget / 'spend.json', entries)
        runtime.save_checkpoint(archive / 'applied.json', {'plan_sha256': runtime.digest((archive / 'plan.json').read_bytes()),
            'ledger_before_sha256': runtime.digest(ledger_bytes), 'ledger_after_sha256': runtime.digest((budget / 'spend.json').read_bytes()),
            'helper_sha256': runtime.digest(Path(__file__).read_bytes()), 'calls': len(proposal['changes']), 'release_usd': proposal['release_usd']})
        return proposal


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget', required=True)
    p.add_argument('--proof', required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = reconcile(args.budget, args.proof, apply=args.apply)
    runtime.write_json(Path(args.output), result)
    print(json.dumps({'calls': len(result['changes']), 'release_usd': result['release_usd'], 'applied': args.apply}))
