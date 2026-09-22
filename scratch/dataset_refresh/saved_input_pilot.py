# ABOUTME: Runs only the four approved saved-input Sonnet author requests against the original shared ledger.
# ABOUTME: Preparation is offline; explicit disabled-by-default dispatch preserves all origins and requires independent adoption.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from filelock import FileLock
from omegaconf import OmegaConf

from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.per_row import conversation, verify_accepted


REPO = Path(__file__).resolve().parents[2]
PROPOSAL_SHA = '878f793db54c356850c099e2b0dd29ae5f8a2292a1d711ec6b6f4efb89e7f9fe'
IDS = ('t1_004_v0', 't3_001_v0', 't5_007_v0', 't6_012_v0')
MAX_RESERVATION = 0.6722025
MODEL = 'anthropic/claude-sonnet-5'
ORIGINAL_BUDGET_ROOT = REPO/'output/2026-09-14_dataset_refresh/budget'


def sha(path):
    return base.digest(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_file(path, expected):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or sha(path) != expected:
        raise ValueError('Bound file changed or symlink: ' + str(path))
    return path


def code_files():
    return {str(p.resolve()): sha(p) for p in (
        Path(__file__), Path(base.__file__), REPO/'scratch/dataset_refresh/per_row.py',
        REPO/'src/infra/endpoints/openrouter.py', REPO/'configs/endpoints/providers.yaml')}


def verify_packet(config_path):
    config_path = Path(config_path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if config.get('execution_enabled') is not False or config.get('dispatch_authorized') is not False:
        raise ValueError('Original proposal config must remain disabled')
    proposal_path = check_file(config['proposal_manifest'], PROPOSAL_SHA)
    if config['proposal_manifest_sha256'] != PROPOSAL_SHA:
        raise ValueError('Unapproved proposal')
    proposal = read(proposal_path)
    if tuple(x['candidate_id'] for x in proposal['rows']) != IDS:
        raise ValueError('Exactly the four frozen candidate IDs are required')
    if proposal['sum_fixed_request_reservations_usd'] != MAX_RESERVATION:
        raise ValueError('Reservation total changed')
    registry = check_file(REPO/'configs/endpoints/providers.yaml', proposal['provider_registry_sha256'])
    review_path = check_file(proposal_path.parent/'independent_review_contract.json', proposal['review_contract_sha256'])
    review = read(review_path)
    payloads = []
    for entry in proposal['rows']:
        root = Path(entry['root']).resolve()
        cfg = base.validate_arm(root, entry['arm'])
        check_file(root/entry['arm']/'config.json', proposal['arm_config_sha256'])
        row = root/entry['arm']/'records'/entry['candidate_id']
        for key, name in [('source_result_sha256', 'result.json'), ('source_identity_sha256', 'identity.json'),
                          ('source_exclusion_sha256', 'independent_exclusion.json'),
                          ('source_scenario_sha256', 'scenario.json'), ('source_preflight_sha256', 'preflight.json')]:
            check_file(row/name, entry[key])
        original = base.load_checkpoint(row/'result.json')
        exclusion = base.load_checkpoint(row/'independent_exclusion.json')
        if original['status'] != 'accepted' or exclusion['result_sha256'] != entry['source_result_sha256']:
            raise ValueError('Original accepted answer must remain bound excluded')
        verify_accepted(row/'result.json', original, cfg)
        if not base.acceptance(base.load_checkpoint(row/'preflight.json'), cfg['preflight']):
            raise ValueError('Source scenario failed eligibility')
        if list(row.glob('independent_rewrite_*.json')):
            raise ValueError('An independent rewrite already exists')
        if (review['full_constitution_text'] != cfg['review_constitution_text'] or
                review['constitution_sha256'] != cfg['constitution_sha256']):
            raise ValueError('Full constitution compatibility review changed')
        path = proposal_path.parent/entry['proposal_file']
        if path.resolve().parent != proposal_path.parent.resolve():
            raise ValueError('Payload outside proposal')
        payload = read(check_file(path, entry['proposal_sha256']))
        req = payload['request']
        if base.digest(req) != entry['request_sha256'] or payload['request_sha256'] != entry['request_sha256']:
            raise ValueError('Exact request changed')
        if (payload['actual_conversation'] != conversation(original['record']) or
                payload['full_working_preference'] != original['record']['trait_text']):
            raise ValueError('Saved input changed')
        if {k: v for k, v in req.items() if k != 'messages'} != {
                'model': MODEL, 'temperature': 0.7, 'max_tokens': 12288}:
            raise ValueError('Author settings changed')
        price = base.provider_price(MODEL)
        bound = len(json.dumps(req['messages'], ensure_ascii=False).encode()) + 2048
        reserve = (1.25*bound*price['in'] + req['max_tokens']*price['out'])/1e6
        if (price != proposal['observed_config_price_usd_per_million_tokens'] or
                base.provider_pin(MODEL) != proposal['provider_pin'] or
                abs(reserve-entry['reservation_usd']) > 1e-12):
            raise ValueError('Provider or request reservation changed')
        payloads.append(payload)
    return config, proposal, payloads, proposal_path


def prepare(config_path, output):
    _, proposal, payloads, proposal_path = verify_packet(config_path)
    output = Path(output).resolve()
    roots = {Path(x['root']).resolve() for x in proposal['rows']}
    if output.exists() or any(output.is_relative_to(r) for r in roots) or output.is_relative_to(proposal_path.parent):
        raise ValueError('Output must be a new directory outside origins and proposal')
    meta = read(next(iter(roots))/'run_meta.json')
    budget = Path(meta['budget_root']).resolve()
    if budget != ORIGINAL_BUDGET_ROOT.resolve():
        raise ValueError('Must use the original shared budget directory')
    # All other source roots, if ever present, must name this same original ledger.
    if any(Path(read(r/'run_meta.json')['budget_root']).resolve() != budget for r in roots):
        raise ValueError('Sources do not share the original ledger')
    with FileLock(str(budget/'spend.lock'), timeout=1):
        ledger = check_file(budget/'spend.json', proposal['budget_ledger_sha256'])
        entries = read(ledger)
        if len(entries) != proposal['closed_call_count'] or any(e['status'] not in ('settled', 'billing_verified_failure') for e in entries):
            raise ValueError('Baseline ledger is not the closed approved ledger')
        if sum(e['charged_or_reserved_usd'] for e in entries) + MAX_RESERVATION > 250:
            raise ValueError('Insufficient shared headroom for all four request bounds')
        output.mkdir(parents=True)
        shutil.copyfile(ledger, output/'baseline_ledger.json')
        manifest = {'config_path': str(Path(config_path).resolve()), 'config_sha256': sha(config_path),
                    'source_git_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                    'proposal_path': str(proposal_path), 'proposal_sha256': PROPOSAL_SHA,
                    'code_files': code_files(), 'budget_root': str(budget),
                    'baseline_ledger_sha256': sha(ledger), 'baseline_entries_digest': base.digest(entries),
                    'baseline_count': len(entries), 'max_reservation_usd': MAX_RESERVATION,
                    'review_contract_sha256': proposal['review_contract_sha256'],
                    'candidate_ids': list(IDS), 'roots': [str(r) for r in sorted(roots)],
                    'status': 'prepared_disabled', 'maximum_physical_calls': 4}
        base.save_checkpoint(output/'manifest.json', manifest)
        for payload in payloads:
            base.save_checkpoint(output/(payload['candidate_id']+'.input.json'), payload)
        shutil.copyfile(proposal_path.parent/'independent_review_contract.json', output/'independent_review_contract.json')
        dispatch = {'execution_enabled': False, 'dispatch_authorized': False,
                    'manifest_sha256': sha(output/'manifest.json'), 'shared_cap_usd': 250,
                    'maximum_physical_calls': 4, 'workers': 1}
        OmegaConf.save(OmegaConf.create(dispatch), output/'dispatch.yaml')
        return manifest


def parsed_fields(content):
    fields, spans = {}, []
    for name in ('reasoning', 'response'):
        if content.count('<'+name+'>') != 1 or content.count('</'+name+'>') != 1:
            raise ValueError('Missing or ambiguous training tag: '+name)
        found = re.search('<'+name+'>(.*?)</'+name+'>', content, flags=re.S)
        if not found or not found.group(1).strip():
            raise ValueError('Incomplete or empty training tag: '+name)
        fields[name] = found.group(1).strip()
        spans.append(found.span())
    if spans[0][1] > spans[1][0]:
        raise ValueError('Training tags overlap or occur in wrong order')
    changes = re.findall(r'<changes>(.*?)</changes>', content, flags=re.S)
    return fields, {'changes': changes[0].strip() if len(changes) == 1 else None,
                    'changes_format_omission': len(changes) != 1,
                    'policy': 'Audit-only; no invented explanation and no paid formatting retry.'}


def verify_ledger(client, manifest, output, complete):
    entries = client.entries()
    n = manifest['baseline_count']
    if len(entries) != n + len(complete) or base.digest(entries[:n]) != manifest['baseline_entries_digest']:
        raise ValueError('Baseline ledger or exclusive pilot tail changed')
    if not complete and sha(client.path) != manifest['baseline_ledger_sha256']:
        raise ValueError('Baseline ledger bytes changed')
    for entry, receipt in zip(entries[n:], complete):
        if (entry['status'] != 'settled' or entry['run_root'] != str(output) or
                entry['arm'] != 'nonmoral-saved-input-pilot' or
                entry['stage'] != 'single_saved_revision' or
                entry['call_id'] != receipt['call_id'] or
                base.digest(entry) != receipt['ledger_entry_sha256']):
            raise ValueError('Prior pilot call is unsettled or its receipt changed')
    return entries


def execute(output, send=None):
    output = Path(output).resolve()
    with FileLock(str(output/'execution.lock'), timeout=1):
        manifest = base.load_checkpoint(output/'manifest.json')
        dispatch = OmegaConf.to_container(OmegaConf.load(output/'dispatch.yaml'), resolve=True)
        expected = {'execution_enabled': True, 'dispatch_authorized': True,
                    'manifest_sha256': sha(output/'manifest.json'), 'shared_cap_usd': 250,
                    'maximum_physical_calls': 4, 'workers': 1}
        if (dispatch != expected or dispatch.get('execution_enabled') is not True or
                dispatch.get('dispatch_authorized') is not True):
            raise ValueError('Explicit root dispatch remains disabled or changed')
        for path, digest in manifest['code_files'].items():
            check_file(path, digest)
        check_file(manifest['config_path'], manifest['config_sha256'])
        check_file(output/'baseline_ledger.json', manifest['baseline_ledger_sha256'])
        check_file(output/'independent_review_contract.json', manifest['review_contract_sha256'])
        if Path(manifest['budget_root']).resolve() != ORIGINAL_BUDGET_ROOT.resolve():
            raise ValueError('Must use the original shared budget directory')
        if (output/'summary.json').exists():
            return base.load_checkpoint(output/'summary.json')
        if (output/'execution_started.json').exists():
            raise ValueError('Pilot already started without final summary; inspect evidence, never retry')
        _, proposal, payloads, _ = verify_packet(manifest['config_path'])
        for p in payloads:
            if base.load_checkpoint(output/(p['candidate_id']+'.input.json')) != p:
                raise ValueError('Prepared payload changed')
        client = base.BudgetClient(manifest['budget_root'], 250, {MODEL}, send=send)
        client.local.arm = 'nonmoral-saved-input-pilot'
        client.local.stage = 'single_saved_revision'
        client.local.run_root = str(output)
        completed, results = [], []
        with client.lock.acquire(timeout=1):
            verify_ledger(client, manifest, output, completed)
            base.save_checkpoint(output/'execution_started.json', {'dispatch_sha256': sha(output/'dispatch.yaml')})
            for index, payload in enumerate(payloads):
                cid = payload['candidate_id']
                item = {'candidate_id': cid, 'status': 'failed', 'automatic_acceptance': False}
                try:
                    # Recheck origins, review text, code, exact requests and budget before EVERY call.
                    verify_packet(manifest['config_path'])
                    for path, digest in manifest['code_files'].items():
                        check_file(path, digest)
                    check_file(output/'independent_review_contract.json', manifest['review_contract_sha256'])
                    entries = verify_ledger(client, manifest, output, completed)
                    remaining = sum(x['reservation_usd'] for x in payloads[index:])
                    if sum(e['charged_or_reserved_usd'] for e in entries)+remaining > 250:
                        raise ValueError('Insufficient shared headroom for remaining frozen requests')
                    client.local.candidate_id = cid
                    call_id = len(entries)
                    base.save_checkpoint(output/(cid+'.started.json'), {
                        'call_id': call_id, 'request_sha256': payload['request_sha256']})
                    result = client.chat(**payload['request'])
                    raw = client.root/'raw_calls'/f'{call_id:06d}.json'
                    shutil.copyfile(raw, output/(cid+'.raw.json'))
                    entry = client.entries()[call_id]
                    if entry['request_sha256'] != payload['request_sha256'] or result.finish_reason != 'stop':
                        raise ValueError('Unexpected request binding or finish reason')
                    if result.response_model != MODEL:
                        raise ValueError('Response model differs from the pinned Sonnet author')
                    fields, audit = parsed_fields(result.content)
                    repaired = {**payload['actual_conversation'], **fields}
                    issues = base.simple_checks(repaired)
                    if issues:
                        raise ValueError('Frozen local checks: '+', '.join(issues))
                    receipt = {'call_id': call_id, 'ledger_entry_sha256': base.digest(entry),
                               'raw_sha256': sha(raw), 'request_sha256': payload['request_sha256']}
                    item.update(status='awaiting_independent_full_review', conversation=repaired,
                                audit=audit, physical_receipt=receipt)
                    completed.append(receipt)
                except BaseException as exc:
                    item.update(error_type=type(exc).__name__, error=str(exc)[:2000])
                    # The shared client preserves raw/unknown charges. Copy whatever exists as evidence.
                    marker = output/(cid+'.started.json')
                    if marker.exists():
                        ident = base.load_checkpoint(marker)['call_id']
                        raw = client.root/'raw_calls'/f'{ident:06d}.json'
                        if raw.exists():
                            shutil.copyfile(raw, output/(cid+'.raw.json'))
                    base.save_checkpoint(output/(cid+'.result.json'), item)
                    results.append(item)
                    break
                base.save_checkpoint(output/(cid+'.result.json'), item)
                results.append(item)
            summary = {'status': 'awaiting_independent_full_review' if len(completed) == 4 else 'stopped_no_retry',
                       'results': results, 'automatic_accepted_rows': 0,
                       'physical_calls_added': len(client.entries())-manifest['baseline_count'],
                       'shared_exposure_usd': sum(e['charged_or_reserved_usd'] for e in client.entries()),
                       'policy': 'No adoption/publication. Every changed answer requires exact-byte independent full review.'}
            base.save_checkpoint(output/'summary.json', summary)
            return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--config', required=True)
    prep.add_argument('--output', required=True)
    run = sub.add_parser('execute')
    run.add_argument('--output', required=True)
    args = parser.parse_args()
    result = prepare(args.config, args.output) if args.command == 'prepare' else execute(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
