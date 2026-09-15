# ABOUTME: Independently bind final nonmoral selection, physical authors and shared closed accounting.
# ABOUTME: Read-only validators and new evidence only; no model calls, origin changes or release approval.
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from omegaconf import OmegaConf
from scratch.dataset_refresh import run as rt, offline_acceptance as oa, publish_offline_composite as pub
from scratch.dataset_refresh.billing_evidence import validate_billing_call
from scratch.dataset_refresh.saved_input_pilot import parsed_fields
from scratch.dataset_refresh.recover_scenario_json import repair_two_string_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    preview, output = Path(cfg['preview']), Path(cfg['output'])
    if output.exists(): raise FileExistsError(output)
    manifest = oa.read(preview/'preview_manifest.json')
    assert manifest['dataset_sha256'] == oa.sha(preview/'dataset.jsonl') == cfg['dataset_sha256']
    assert manifest['selection_sha256'] == oa.sha(preview/'selection.json')
    rows, phases, dossiers = pub.validate_selection(preview/'selection.json')
    actual = rt.read_rows(preview/'dataset.jsonl')
    assert actual == rows
    ledger_path = oa.ORIGINAL_BUDGET_ROOT/'spend.json'; ledger = oa.read(ledger_path)
    ledger_sha = oa.sha(ledger_path)
    totals = pub.accounting(ledger, cfg['ledger_end_exclusive'])
    charged = sum(Decimal(str(x['charged_or_reserved_usd'])) for x in ledger)
    assert charged == Decimal(str(cfg['charged_usd']))
    by_source = defaultdict(list)
    for call in ledger:
        by_source[(call.get('run_root'), call['arm'], call['candidate_id'])].append(call)
    raw_cache = {}
    def raw(call):
        cid = call['call_id']
        if cid not in raw_cache:
            path = oa.ORIGINAL_BUDGET_ROOT/'raw_calls'/f'{cid:06d}.json'
            value = oa.read(path)
            assert value['accounting'] == call and rt.digest(value['request']) == call['request_sha256']
            raw_cache[cid] = (path, value)
        return raw_cache[cid]
    author_rows, source_texts = [], []
    base_selection = oa.read(oa.read(preview/'selection.json')['base_selection']['path'])
    for phase in phases:
        source_texts.extend(rt.read_rows(Path(phase['root'])/pub.ARM/'source.jsonl'))
    for entry in base_selection['entries']:
        path = Path(entry['root'])/pub.ARM/'records'/entry['candidate_id']/'result.json'
        result = rt.load_result(path); record = result['record']
        calls = by_source[(entry['root'], pub.ARM, entry['candidate_id'])]
        matches, scenarios = [], []
        for call in calls:
            if call['status'] != 'settled': continue
            if call['stage'] == 'scenario':
                rp, value = raw(call)
                try: parsed = rt._parse_json(value['response']['content'])
                except ValueError: parsed, _ = repair_two_string_json(value['response']['content'])
                if all(parsed[k] == record[k] for k in ('system', 'user')):
                    assert value['request']['model'] == value['response']['response_model'] == oa.MODEL
                    scenarios.append({'call_id': call['call_id'], 'raw_sha256': oa.sha(rp)})
            if not any(word in call['stage'] for word in ('revise', 'rewrite', 'repair')): continue
            rp, value = raw(call)
            try: fields, _ = parsed_fields(value['response']['content'])
            except (ValueError, KeyError, AssertionError): continue
            if all(fields[k] == record[k] for k in ('reasoning', 'response')):
                assert value['request']['model'] == value['response']['response_model'] == oa.MODEL
                assert value['response']['finish_reason'] == 'stop'
                matches.append({'call_id': call['call_id'], 'stage': call['stage'], 'raw_sha256': oa.sha(rp)})
        assert matches and scenarios, (entry['candidate_id'], matches, scenarios)
        author_rows.append({'candidate_id': entry['candidate_id'], 'route': 'base650', 'root': entry['root'],
                            'result_sha256': entry['result_sha256'], 'scenario_physical_matches': scenarios,
                            'answer_physical_matches': matches})
    offline_rows = []; routes = Counter(); selected_second_calls = set()
    for audit in dossiers:
        d, frozen = oa.validate_dossier(Path(audit['path'])/'dossier.json')
        call = ledger[d['physical_receipt']['call_id']]
        inp = oa.read(frozen['author_input']) if 'author_input' in frozen else {}
        route = ('unchanged' if d['author_kind'] == 'untouched_saved_final' else
                 'second' if inp.get('prior_physical_receipt') else
                 'pilot' if call['arm'] == 'nonmoral-saved-input-pilot' else 'first')
        routes[route] += 1
        scopes = pub.author_execution_scopes(Path(audit['path'])/'dossier.json', ledger, oa.ORIGINAL_BUDGET_ROOT)
        if route == 'second': selected_second_calls.add(call['call_id'])
        offline_rows.append({'candidate_id': d['source_ref']['candidate_id'], 'root': d['source_ref']['root'],
            'route': route, 'dossier_path': str(Path(audit['path'])/'dossier.json'), 'dossier_sha256': oa.sha(Path(audit['path'])/'dossier.json'),
            'acceptance_sha256': audit['acceptance_sha256'], 'call_id': call['call_id'],
            'physical_receipt': d['physical_receipt'], 'verified_execution_scopes': sorted(scopes),
            'constitution_sha256': d['constitution_sha256'], 'review_contract_sha256': d['review_contract_sha256']})
    assert dict(routes) == cfg['offline_counts'], routes
    second_calls = [x for x in ledger if x['arm'] == 'nonmoral-saved-input-second-pass']
    omitted = [x for x in second_calls if x['call_id'] not in selected_second_calls]
    assert len(second_calls) == 12 and len(selected_second_calls) == 11
    assert len(omitted) == 1 and omitted[0]['candidate_id'].endswith('::t8_006_v0')
    assert not any(r['metadata'].get('original_scenario_id') == 't8_006_v0' for r in actual)
    historical_pairs = {(s.get('reasoning'), s.get('response')) for s in source_texts}
    historical_finals = {s.get('response') for s in source_texts if s.get('response')}
    historical_users = {s.get('user') for s in source_texts if s.get('user')}
    overlaps = {'reasoning_and_final': [], 'final_only': [], 'user_exact': []}
    for row in actual:
        msg = row['messages'][-1]; sid = row['metadata']['scenario_id']
        if (msg['reasoning_content'], msg['content']) in historical_pairs: overlaps['reasoning_and_final'].append(sid)
        if msg['content'] in historical_finals: overlaps['final_only'].append(sid)
        if row['messages'][1]['content'] in historical_users: overlaps['user_exact'].append(sid)
    assert not any(overlaps.values()), overlaps
    billing_cache = {}; reconciled = []
    for call in ledger:
        if call['status'] == 'billing_verified_failure':
            path = oa.ORIGINAL_BUDGET_ROOT/'raw_calls'/f"{call['call_id']:06d}.json"
            validate_billing_call(oa.ORIGINAL_BUDGET_ROOT, call, oa.read(path), billing_cache)
            reconciled.append(call['call_id'])
    assert oa.sha(ledger_path) == ledger_sha
    counts = Counter(row['metadata']['trait_id'] for row in actual)
    assert counts == rt.quotas() and len(actual) == 716
    report = {'status': 'checked_no_provenance_or_accounting_blocker_in_scope', 'release_approved': False,
        'reviewer_provenance': {'kind': 'independent_codex_agent', 'task': '/root/audit_nonmoral', 'human_review': False},
        'dataset_sha256': cfg['dataset_sha256'], 'selection_sha256': manifest['selection_sha256'],
        'preview_manifest_sha256': oa.sha(preview/'preview_manifest.json'), 'rows': 716, 'quotas': dict(counts),
        'base_rows': 650, 'offline_rows': 66, 'offline_route_counts': dict(routes),
        'base_author_rows': author_rows, 'offline_author_rows': offline_rows,
        'actual_author_model': oa.MODEL, 'historical_source': phases[0]['config']['source'],
        'historical_source_rows_checked': len(source_texts), 'historical_exact_overlap': overlaps,
        'historical_interpretation': 'The original craft dataset is scenario/mechanism inspiration, not reused trained answer text. Exact comparison covers the frozen source snapshot; it is not a semantic originality guarantee.',
        'second_physical_calls': len(second_calls), 'second_selected_calls': sorted(selected_second_calls),
        'second_unselected_calls': [{'call_id': x['call_id'], 'candidate_id': x['candidate_id']} for x in omitted],
        'budget': {**totals, 'charged_decimal_usd': str(charged), 'ledger_sha256': ledger_sha,
                   'billing_verified_failure_proofs_revalidated': len(reconciled)},
        'validators': {p.name: oa.sha(p) for p in (Path(pub.__file__), Path(oa.__file__), Path(__file__))},
        'config_sha256': oa.sha(args.config),
        'limitations': ['No new full content review of all716 is claimed. Content, literal and semantic audits are separate.',
            'Re-execution of frozen validators is an integrity check, not an independent reimplementation of every gate.',
            'Physical model IDs and provider records do not establish immutable model weights.',
            'Budget is cumulative for both arms, pilots and review probes; it is not the marginal cost of selected rows.',
            'No API calls, publication, adoption, training or evaluation occurred in this audit. Root release approval is separate.']}
    rt.save_checkpoint(output, report); print(str(output), oa.sha(output), dict(routes), str(charged))


if __name__ == '__main__': main()
