# ABOUTME: Freeze and validate independently reviewed saved Sonnet answers without changing historical rows.
# ABOUTME: Keep author evidence, fresh constitutional/craft review and root adoption distinct from automatic judges.
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.saved_input_pilot import MODEL, ORIGINAL_BUDGET_ROOT, parsed_fields
from scratch.dataset_refresh.publish_completed import validate_export

ROUTE = 'independent_offline_full_read_v1'
CONSTITUTION_SHA = '8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc'
REVIEW_CONTRACT_SHA = '5d1ec43313b10807e50f245c6a342e093f8a324f20889a00803e5276c22860e2'
FIELDS = ('system', 'user', 'reasoning', 'response')
TEXT_KEYS = set(FIELDS) | {'draft_reasoning', 'draft_response', 'rewrite_changes'}
GATES = ('advice_to_human', 'nonmoral', 'benign_subject', 'self_contained', 'grounded',
         'genuine_tension', 'detailed_deliberation', 'practical_recommendation',
         'constitution_compatible', 'no_training_leakage')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return base.digest(Path(path).read_bytes())


def regular(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Symlink/junction is not evidence: ' + str(part))
    if not path.is_file():
        raise ValueError('Missing evidence: ' + str(path))
    return path.resolve()


def bound(path, expected):
    path = regular(path)
    if sha(path) != expected:
        raise ValueError('Evidence hash differs: ' + str(path))
    return path


def messages(conv):
    return [{'role': 'system', 'content': conv['system']},
            {'role': 'user', 'content': conv['user']},
            {'role': 'assistant', 'content': conv['response'], 'reasoning_content': conv['reasoning']}]


def origin_name(origin):
    # Archived origin references are identifiers; never resolve another host's filesystem paths.
    return (PureWindowsPath(origin) if '\\' in origin or ':' in origin else PurePosixPath(origin)).name


def validate_author(raw, receipt, conv, cid, request=None, ledger_candidate_id=None):
    entry, response = raw['accounting'], raw['response']
    if (entry['status'] != 'settled' or entry['model'] != MODEL or
            response.get('response_model') != MODEL or response.get('finish_reason') != 'stop'):
        raise ValueError('Author must be a settled complete pinned Sonnet response')
    if (entry['candidate_id'] != (ledger_candidate_id or cid) or entry['call_id'] != receipt['call_id'] or
            base.digest(entry) != receipt['ledger_entry_sha256'] or
            base.digest(raw['request']) != receipt['request_sha256'] or
            entry['request_sha256'] != receipt['request_sha256'] or raw['request']['model'] != MODEL):
        raise ValueError('Author receipt/request binding differs')
    if request is not None and raw['request'] != request:
        raise ValueError('Submitted request differs from frozen input')
    fields, _ = parsed_fields(response['content'])
    if any(fields[k] != conv[k] for k in ('reasoning', 'response')):
        raise ValueError('Trained answer differs from physical author output')
    if base.simple_checks(conv):
        raise ValueError('Local answer checks failed: ' + repr(base.simple_checks(conv)))


def validate_revision_input(inp, result, ref, conv, preference, receipt):
    cid = ref['candidate_id']
    if (result['status'] != 'awaiting_independent_full_review' or result['automatic_acceptance'] is not False or
            result['candidate_id'] != cid or inp['candidate_id'] != cid or
            inp['full_working_preference'] != preference or result['physical_receipt'] != receipt or
            result['conversation'] != conv or base.digest(inp['request']) != inp['request_sha256']):
        raise ValueError('Revision result/input is not a bound independent-review candidate')
    if any(inp['actual_conversation'][k] != conv[k] for k in ('system', 'user')):
        raise ValueError('Revision changed the source scenario')
    ledger_key = cid
    if 'case_key' in inp or 'source_ref' in inp:
        ledger_key = origin_name(ref['root'])+'::'+cid
        if inp.get('case_key') != ledger_key or inp.get('source_ref') != ref:
            raise ValueError('Campaign case key/source reference differs')
    rendered = '\n'.join(m['content'] for m in inp['request']['messages'])
    # Plain template fields and JSON-encoded source blocks are both legitimate renderings.
    for text in (conv['system'], conv['user'], preference):
        if text not in rendered and json.dumps(text, ensure_ascii=False)[1:-1] not in rendered:
            raise ValueError('Author request omitted the actual full source or working preference')
    return ledger_key


def validate_contract(contract, cfg, preference, trait):
    if (contract['constitution_sha256'] != CONSTITUTION_SHA or
            base.digest(contract['full_constitution_text'].encode()) != CONSTITUTION_SHA or
            contract['full_constitution_text'] != cfg['review_constitution_text'] or
            cfg['constitution_sha256'] != CONSTITUTION_SHA or
            contract['full_frozen_acceptance'] != cfg['acceptance'] or
            contract['full_frozen_target_review_prompts'] != {k: cfg['prompts'][k] for k in ('review_system', 'review_user')} or
            preference != cfg['operational_traits'][trait]):
        raise ValueError('Full new09/craft review contract differs')
    if set(contract['full_frozen_acceptance']['gates']) != set(GATES):
        raise ValueError('Expected ten independent content gates')


def prepare(spec_path, output):
    """Freeze a proposal only. No source mutation, acceptance or network operation."""
    spec_path = regular(spec_path)
    spec = read(spec_path)
    ref = spec['source_ref']
    root, arm, cid = Path(ref['root']).resolve(), ref['arm'], ref['candidate_id']
    if ref['root'] != str(root):
        raise ValueError('Source reference root must be an explicit resolved absolute path')
    if arm != 'nonmoral-advice' or Path(cid).name != cid:
        raise ValueError('Only an explicit nonmoral candidate is supported')
    row = root/arm/'records'/cid
    output = Path(output).resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError('New evidence directory must be outside the frozen origin')
    cfg = base.validate_arm(root, arm)
    original_path = bound(row/'result.json', ref['result_sha256'])
    original = base.load_checkpoint(original_path)
    identity = base.load_checkpoint(row/'identity.json')
    candidate = next((r for r in base.read_rows(root/arm/'candidates.jsonl') if r['candidate_id'] == cid), None)
    if candidate is None or identity != {'candidate_sha256': base.digest(candidate), 'config_sha256': base.digest(cfg)}:
        raise ValueError('Frozen source identity differs')
    scenario = base.load_checkpoint(row/'scenario.json')
    if not base.acceptance(base.load_checkpoint(row/'preflight.json'), cfg['preflight']):
        raise ValueError('Original source preflight was ineligible')
    record = original['record']
    if any(record.get(k) != scenario[k] for k in ('system', 'user')):
        raise ValueError('Original source scenario differs')
    review_cfg_path = bound(spec['review_config_path'], spec['review_config_sha256'])
    review_cfg = read(review_cfg_path)
    contract_path = bound(spec['review_contract_path'], spec['review_contract_sha256'])
    if sha(contract_path) != REVIEW_CONTRACT_SHA:
        raise ValueError('Independent review contract must be the frozen full qualified contract')
    contract = read(contract_path)
    preference = review_cfg['operational_traits'][record['trait_id']]
    validate_contract(contract, review_cfg, preference, record['trait_id'])
    kind = spec['author_kind']
    raw_path = bound(spec['raw_path'], spec['physical_receipt']['raw_sha256'])
    raw = read(raw_path)
    extra = {}
    if kind == 'single_saved_revision':
        input_path = bound(spec['input_path'], spec['input_sha256'])
        result_path = bound(spec['result_path'], spec['result_sha256'])
        inp, result = base.load_checkpoint(input_path), base.load_checkpoint(result_path)
        conv = result['conversation']
        request = inp['request']
        ledger_key = validate_revision_input(inp, result, {**ref, 'root': str(root)}, conv, preference, spec['physical_receipt'])
        extra = {'author_input': input_path, 'author_result': result_path}
    elif kind == 'untouched_saved_final':
        result_path = bound(row/(spec['saved_stage']+'.json'), spec['saved_stage_sha256'])
        saved = base.load_checkpoint(result_path)
        conv = {**{k: scenario[k] for k in ('system', 'user')},
                **{k: saved[k] for k in ('reasoning', 'response')}}
        if raw['accounting']['stage'] != spec['saved_stage']:
            raise ValueError('Untouched final stage differs')
        request = None
        ledger_key = cid
        extra = {'author_result': result_path}
    else:
        raise ValueError('Unsupported author route')
    if set(conv) != set(FIELDS) or any(conv[k] != scenario[k] for k in ('system', 'user')):
        raise ValueError('Actual source fields must remain unchanged')
    validate_author(raw, spec['physical_receipt'], conv, cid, request, ledger_key)
    ledger_path = regular(spec['ledger_path'])
    if ledger_path != (ORIGINAL_BUDGET_ROOT/'spend.json').resolve():
        raise ValueError('Author evidence must be checked against the original shared ledger')
    ledger = read(ledger_path)
    call_id = spec['physical_receipt']['call_id']
    if ledger[call_id] != raw['accounting']:
        raise ValueError('Shared ledger author entry differs')
    shared_raw = ledger_path.parent/'raw_calls'/f'{call_id:06d}.json'
    bound(shared_raw, spec['physical_receipt']['raw_sha256'])
    if kind == 'untouched_saved_final' and (
            raw['accounting'].get('run_root') != str(root) or raw['accounting']['arm'] != arm):
        raise ValueError('Untouched author came from a different source phase')
    files = {'spec': spec_path, 'origin_config': root/arm/'config.json',
             'review_config': review_cfg_path, 'review_contract': contract_path,
             'author_raw': raw_path, **extra}
    for path in sorted(row.rglob('*.json')):
        files['origin/'+path.relative_to(row).as_posix()] = path
    # Snapshot receipts and failures too; existence or past judge labels never imply fresh acceptance.
    planned = {key: (regular(path), sha(path)) for key, path in files.items()}
    output.mkdir(parents=True)
    frozen = {}
    for index, (key, (path, digest)) in enumerate(planned.items()):
        name = f'evidence/{index:04d}.json'
        dest = output/name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(bound(path, digest), dest)
        bound(dest, digest)
        bound(path, digest)
        frozen[key] = {'path': name, 'sha256': digest, 'origin_path': str(path)}
    base.write_json(output/'source_candidate.json', candidate)
    frozen['source_candidate'] = {'path': 'source_candidate.json', 'sha256': sha(output/'source_candidate.json')}
    dossier = {'route': ROUTE, 'status': 'awaiting_independent_offline_review',
               'source_ref': {**ref, 'root': str(root)}, 'original_status': original['status'],
               'conversation': conv, 'conversation_sha256': base.digest(conv),
               'source_metadata': {k: v for k, v in record.items() if k not in TEXT_KEYS},
               'full_working_preference': preference, 'full_working_preference_sha256': base.digest(preference.encode()),
               'review_contract_sha256': sha(contract_path), 'constitution_sha256': CONSTITUTION_SHA,
               'author_kind': kind, 'physical_receipt': spec['physical_receipt'],
               'original_automatic_reviews_apply_to_new_answer': False,
               'files': frozen, 'implementation_sha256': sha(__file__)}
    base.save_checkpoint(output/'dossier.json', dossier)
    validate_dossier(output/'dossier.json')
    return dossier


def validate_dossier(path):
    path = regular(path)
    d = base.load_checkpoint(path)
    if d['route'] != ROUTE or d['status'] != 'awaiting_independent_offline_review':
        raise ValueError('Unexpected dossier route/status')
    if d['author_kind'] not in ('single_saved_revision', 'untouched_saved_final') or d['implementation_sha256'] != sha(__file__):
        raise ValueError('Unrecognized author route or changed offline validator implementation')
    frozen = {}
    for key, value in d['files'].items():
        p = path.parent/value['path']
        if path.parent not in p.resolve().parents:
            raise ValueError('Evidence escapes dossier')
        frozen[key] = bound(p, value['sha256'])
    conv, ref = d['conversation'], d['source_ref']
    spec = read(frozen['spec'])
    expected_ref = spec['source_ref']
    candidate = read(frozen['source_candidate'])
    if (ref != expected_ref or d['author_kind'] != spec['author_kind'] or
            candidate['candidate_id'] != ref['candidate_id'] or ref['arm'] != 'nonmoral-advice'):
        raise ValueError('Archived spec, source reference or candidate identity differs')
    if base.digest(conv) != d['conversation_sha256']:
        raise ValueError('Conversation binding differs')
    original, scenario = read(frozen['origin/result.json']), read(frozen['origin/scenario.json'])
    if sha(frozen['origin/result.json']) != ref['result_sha256'] or any(conv[k] != scenario[k] for k in ('system', 'user')):
        raise ValueError('Archived origin differs')
    cfg = read(frozen['origin_config'])
    identity = read(frozen['origin/identity.json'])
    if identity != {'candidate_sha256': base.digest(read(frozen['source_candidate'])), 'config_sha256': base.digest(cfg)}:
        raise ValueError('Archived identity differs')
    if d['source_metadata'] != {k: v for k, v in original['record'].items() if k not in TEXT_KEYS}:
        raise ValueError('Original metadata not preserved')
    if not base.acceptance(read(frozen['origin/preflight.json']), cfg['preflight']):
        raise ValueError('Archived source eligibility differs')
    validate_contract(read(frozen['review_contract']), read(frozen['review_config']),
                      d['full_working_preference'], d['source_metadata']['trait_id'])
    if (sha(frozen['review_contract']) != d['review_contract_sha256'] or
            d['review_contract_sha256'] != REVIEW_CONTRACT_SHA or
            base.digest(d['full_working_preference'].encode()) != d['full_working_preference_sha256'] or
            d['constitution_sha256'] != CONSTITUTION_SHA):
        raise ValueError('Review text binding differs')
    receipt = d['physical_receipt']
    if sha(frozen['author_raw']) != receipt['raw_sha256']:
        raise ValueError('Raw author hash differs')
    request = read(frozen['author_input'])['request'] if d['author_kind'] == 'single_saved_revision' else None
    result = read(frozen['author_result'])
    ledger_key = ref['candidate_id']
    if d['author_kind'] == 'single_saved_revision':
        ledger_key = validate_revision_input(read(frozen['author_input']), result, ref, conv, d['full_working_preference'], receipt)
    elif (read(frozen['author_raw'])['accounting'].get('run_root') != ref['root'] or
          read(frozen['author_raw'])['accounting']['arm'] != ref['arm'] or
          read(frozen['author_raw'])['accounting']['stage'] != spec['saved_stage']):
        raise ValueError('Archived untouched author scope differs')
    validate_author(read(frozen['author_raw']), receipt, conv, ref['candidate_id'], request, ledger_key)
    expected = result['conversation'] if d['author_kind'] == 'single_saved_revision' else {**scenario, **{k: result[k] for k in ('reasoning', 'response')}}
    if any(expected[k] != conv[k] for k in FIELDS):
        raise ValueError('Saved result fields differ from trained conversation')
    return d, frozen


def validate_review(d, frozen, review):
    ref, conv = d['source_ref'], d['conversation']
    expected = {'candidate_id': ref['candidate_id'], 'source_result_sha256': ref['result_sha256'],
                'result_sha256': sha(frozen['author_result']), 'conversation_sha256': d['conversation_sha256'],
                'review_contract_sha256': d['review_contract_sha256'], 'constitution_sha256': CONSTITUTION_SHA,
                'full_working_preference_sha256': d['full_working_preference_sha256'],
                'request_sha256': d['physical_receipt']['request_sha256'], 'physical_receipt': d['physical_receipt']}
    if 'author_input' in frozen:
        expected['input_sha256'] = sha(frozen['author_input'])
    for field in FIELDS:
        expected[{'system': 'source_system', 'user': 'source_user'}.get(field, field)+'_sha256'] = base.digest(conv[field].encode())
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('Independent review binds different source, answer, author or review text')
    if (review.get('decision') != 'accept' or review.get('accepted') is not True or review.get('issues') != [] or
            any(review.get('gates', {}).get(k) is not True for k in GATES)):
        raise ValueError('Independent review did not pass every content gate')
    provenance = review.get('reviewer_provenance', {})
    if provenance.get('kind') != 'independent_codex_agent' or provenance.get('human_review') is not False or not provenance.get('task'):
        raise ValueError('Explicit independent agent reviewer provenance required')
    for key in ('source_first_reason', 'craft_reason', 'constitution_compatibility', 'nonmoral_eligibility'):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise ValueError('Missing substantive independent review: '+key)
    for key, fields in [('source_quotes', ('system', 'user')), ('answer_quotes', ('reasoning', 'response'))]:
        quotes = review.get(key)
        if not isinstance(quotes, list) or not quotes or any(not isinstance(q, str) or not q or not any(q in conv[f] for f in fields) for q in quotes):
            raise ValueError('Independent evidence is not an exact quote: '+key)


def validate_native(d, audit, input_path):
    if (audit.get('status') != 'passed' or audit.get('failures') != [] or audit.get('untruncated') is not True or
            audit.get('tokenizer') != 'Qwen/Qwen3.6-27B' or audit.get('max_train_tokens') != 8192 or
            audit.get('input_sha256') != sha(input_path)):
        raise ValueError('Native token/masking audit differs or failed')
    rows = base.read_rows(input_path)
    if len(rows) != audit.get('rows'):
        raise ValueError('Native audit row count differs')
    matched = [r for r in rows if r['messages'] == messages(d['conversation'])]
    if len(matched) != 1:
        raise ValueError('Native audit must include exactly this conversation once')
    cid = d['source_ref']['candidate_id']
    diagnostics = [x for x in audit['diagnostics'] if x['id'] in (cid, matched[0]['metadata']['scenario_id'])]
    if len(diagnostics) != 1:
        raise ValueError('Ambiguous/missing native diagnostic')
    diag = diagnostics[0]
    if (not 0 < diag['supervised_tokens'] <= diag['training_tokens'] <= 8192 or
            diag.get('raw_reasoning_tokens', 0) <= 0 or diag.get('raw_assistant_content_tokens', 0) <= 0):
        raise ValueError('Native lengths/masking failed')
    return diag


def accept(dossier_path, review_path, approval_path, native_path, native_input_path):
    """Freeze an explicit root adoption; historical source files are never touched."""
    dossier_path = regular(dossier_path)
    d, frozen = validate_dossier(dossier_path)
    paths = {k: regular(v) for k, v in {'independent_review': review_path, 'root_approval': approval_path,
                                      'native_audit': native_path, 'native_input': native_input_path}.items()}
    review, approval, audit = read(paths['independent_review']), read(paths['root_approval']), read(paths['native_audit'])
    validate_review(d, frozen, review)
    diag = validate_native(d, audit, paths['native_input'])
    expected = {'approved': True, 'route': ROUTE, 'dossier_sha256': sha(dossier_path),
                'conversation_sha256': d['conversation_sha256'], 'independent_review_sha256': sha(paths['independent_review']),
                'native_audit_sha256': sha(paths['native_audit']), 'native_input_sha256': sha(paths['native_input'])}
    if (any(approval.get(k) != v for k, v in expected.items()) or approval.get('approved') is not True or
            approval.get('root_actor') != '/root' or not approval.get('reason') or
            approval.get('duplicate_check_passed') is not True or not approval.get('duplicate_evidence_sha256')):
        raise ValueError('Explicit hash-bound root acceptance and duplicate adjudication required')
    duplicate = bound(approval['duplicate_evidence_path'], approval['duplicate_evidence_sha256'])
    paths['duplicate_evidence'] = duplicate
    target = dossier_path.parent/'acceptance.json'
    if target.exists() or (dossier_path.parent/'adoption').exists():
        raise ValueError('Adoption is immutable; an adoption already exists')
    adoption = dossier_path.parent/'adoption'
    adoption.mkdir()
    evidence = {}
    for key, src in paths.items():
        original_sha = sha(src)
        dest = adoption/(key+src.suffix)
        shutil.copyfile(src, dest)
        bound(src, original_sha)
        bound(dest, original_sha)
        evidence[key] = {'path': dest.relative_to(dossier_path.parent).as_posix(), 'sha256': original_sha}
    accepted = {'route': ROUTE, 'status': 'accepted_by_independent_offline_review',
                'dossier_sha256': sha(dossier_path), 'conversation_sha256': d['conversation_sha256'],
                'automatic_acceptance': False, 'files': evidence, 'native_diagnostic': diag,
                'implementation_sha256': sha(__file__)}
    base.save_checkpoint(target, accepted)
    validate_accepted(target, sha(target))
    return accepted


def validate_accepted(path, expected_sha256):
    """Portable validator returning a normalized training row and honest audit metadata."""
    path = bound(path, expected_sha256)
    a = base.load_checkpoint(path)
    dossier_path = bound(path.parent/'dossier.json', a['dossier_sha256'])
    d, frozen = validate_dossier(dossier_path)
    if (a['route'] != ROUTE or a['status'] != 'accepted_by_independent_offline_review' or
            a['automatic_acceptance'] is not False or a['conversation_sha256'] != d['conversation_sha256']):
        raise ValueError('Invalid independent adoption route')
    files = {}
    for key, item in a['files'].items():
        p = (path.parent/item['path']).resolve()
        if path.parent not in p.parents:
            raise ValueError('Adoption evidence escapes bundle')
        files[key] = bound(p, item['sha256'])
    review, approval = read(files['independent_review']), read(files['root_approval'])
    validate_review(d, frozen, review)
    diag = validate_native(d, read(files['native_audit']), files['native_input'])
    if (approval.get('approved') is not True or approval.get('route') != ROUTE or approval.get('root_actor') != '/root' or
            approval.get('dossier_sha256') != sha(dossier_path) or approval.get('conversation_sha256') != d['conversation_sha256'] or
            approval.get('independent_review_sha256') != sha(files['independent_review']) or
            approval.get('native_audit_sha256') != sha(files['native_audit']) or
            approval.get('native_input_sha256') != sha(files['native_input']) or not approval.get('reason') or
            approval.get('duplicate_check_passed') is not True or
            approval.get('duplicate_evidence_sha256') != sha(files['duplicate_evidence']) or diag != a['native_diagnostic']):
        raise ValueError('Root adoption binding differs')
    metadata = dict(d['source_metadata'])
    cid = d['source_ref']['candidate_id']
    # Source metadata remains complete. Working preference and new acceptance are explicit separate fields.
    metadata.update(original_scenario_id=cid, scenario_id='offline_'+d['conversation_sha256'][:20],
                    offline_source_ref=d['source_ref'], offline_author_kind=d['author_kind'],
                    offline_acceptance_route=ROUTE, offline_acceptance_sha256=expected_sha256,
                    offline_dossier_sha256=a['dossier_sha256'], offline_reviewer=review['reviewer_provenance'],
                    offline_review_contract_sha256=d['review_contract_sha256'],
                    offline_working_preference_sha256=d['full_working_preference_sha256'],
                    offline_original_status=d['original_status'], offline_physical_receipt=d['physical_receipt'],
                    offline_original_judges_apply_to_new_answer=False)
    return {'messages': messages(d['conversation']), 'metadata': metadata}, {'dossier': d, 'acceptance': a}


def validate_release(rows):
    """Common last gate for a mixed old-route/offline-route release, after each row was validated."""
    selection = {'scenario_ids': [r['metadata']['scenario_id'] for r in rows], 'quotas': base.quotas()}
    validate_export(rows, selection)
    users = [' '.join(r['messages'][1]['content'].split()).casefold() for r in rows]
    if len(set(users)) != len(rows):
        raise ValueError('Duplicate source prompt in release')
    return {'rows': len(rows), 'quotas': dict(Counter(r['metadata']['trait_id'] for r in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--spec', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('accept')
    for key in ('dossier', 'review', 'approval', 'native', 'native-input'):
        p.add_argument('--'+key, required=True)
    p = sub.add_parser('validate'); p.add_argument('--acceptance', required=True); p.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.spec, args.output)
    elif args.command == 'accept':
        result = accept(args.dossier, args.review, args.approval, args.native, args.native_input)
    else:
        result = validate_accepted(args.acceptance, args.sha256)[0]
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
