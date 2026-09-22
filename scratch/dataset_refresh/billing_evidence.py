# ABOUTME: Verify corrected failed-call charges against immutable original accounting and exact API evidence.
from pathlib import Path
import json
import re
from scratch.dataset_refresh import run as runtime

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_billing_call(budget, call, raw, cache=None):
    """A billing correction never changes the original failed output or its accounting."""
    if call.get('status') != 'billing_verified_failure':
        if raw['accounting'] != call:
            raise ValueError('Scoped raw call differs from ledger')
        return None
    from scratch.dataset_refresh.reconcile_billing import proof_files, finite
    cache = {} if cache is None else cache
    key = call.get('billing_reconciliation_sha256', '')
    if not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('Missing billing reconciliation binding')
    archive = Path(budget) / 'billing_reconciliations' / key
    if key not in cache:
        proof = proof_files(archive)
        applied = runtime.load_checkpoint(archive / 'applied.json')
        plan = read_json(archive / 'plan.json')
        if (runtime.digest(proof['full_file_manifest.json']) != key
                or applied['plan_sha256'] != runtime.digest((archive / 'plan.json').read_bytes())
                or applied['ledger_before_sha256'] != runtime.digest((archive / 'spend_before.json').read_bytes())
                or plan['proof_sha256'] != key):
            raise ValueError('Billing reconciliation archive changed')
        cache[key] = (proof, plan, read_json(archive / 'spend_before.json'))
    proof, plan, before = cache[key]
    changes = [c for c in plan['changes'] if c['call_id'] == call['call_id']]
    if len(changes) != 1:
        raise ValueError('Billing call not uniquely reconciled')
    change = changes[0]
    original = change['before']
    expected = dict(original, status='billing_verified_failure', charged_or_reserved_usd=change['conservative_charge_usd'],
        api_reported_cost_usd=change['customer_cost_usd'], billing_reconciliation_sha256=key,
        billing_generation_id=change['generation_id'])
    cid = call['call_id']
    original_bytes = (archive / 'raw_calls' / f'{cid:06d}.json').read_bytes()
    if (call != expected or original.get('status') != 'uncertain_failure' or before[cid] != original
            or raw['accounting'] != original or json.loads(original_bytes) != raw
            or original_bytes != (Path(budget) / 'raw_calls' / f'{cid:06d}.json').read_bytes()
            or runtime.digest(original_bytes) != change['raw_sha256']):
        raise ValueError('Reconciled call differs from preserved failed accounting')
    data = json.loads(proof[f'{cid:06d}.response.json'])['data']
    verification = json.loads(proof[f'{cid:06d}.verification.json'])
    models = {m['id']: m.get('canonical_slug') for m in json.loads(proof['models_catalogue.response.json'])['data']}
    diag, usage = raw['diagnostics'], raw['diagnostics']['usage']
    costs = [data.get('total_cost'), data.get('upstream_inference_cost'), usage.get('cost'),
             (usage.get('cost_details') or {}).get('upstream_inference_cost')]
    if (verification['http_status'] != 200 or data['id'] != diag['generation_id'] or data['id'] != change['generation_id']
            or models.get(call['model']) != data['model'] or data['provider_name'] != diag['provider']
            or data['native_tokens_prompt'] != usage['prompt_tokens'] or data['native_tokens_completion'] != usage['completion_tokens']
            or not finite(costs[0]) or any(c is not None and not finite(c) for c in costs)
            or max(c for c in costs if c is not None) != call['charged_or_reserved_usd']
            or data['total_cost'] != call['api_reported_cost_usd']):
        raise ValueError('Billing proof does not substantiate corrected cost')
    return archive
