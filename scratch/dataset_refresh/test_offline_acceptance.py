# ABOUTME: Check portable offline answer adoption against tampering, failed reviews and incomplete releases.
# ABOUTME: All fixtures and calls are local; no model request or historical source mutation is permitted.
import copy
import json
from pathlib import Path

import pytest

from scratch.dataset_refresh import offline_acceptance as o


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    o.base.write_json(path, value)
    return path


@pytest.fixture
def packet(tmp_path, monkeypatch):
    root = tmp_path/'origin'
    arm, cid = 'nonmoral-advice', 't1_000_v0'
    row = root/arm/'records'/cid
    conv = {'system': 'You advise makers.', 'user': 'Should my hobby guide use a caption?',
            'reasoning': 'A caption can clarify order, but takes space.',
            'response': 'Try a short caption, then test whether readers follow the order.'}
    constitution, preference = 'Full nine principles.', 'Cut it or keep it. Consider attention and missing information.'
    monkeypatch.setattr(o, 'CONSTITUTION_SHA', o.base.digest(constitution.encode()))
    cfg = {'review_constitution_text': constitution, 'constitution_sha256': o.CONSTITUTION_SHA,
           'acceptance': {'required_true': ['accepted'], 'gates': list(o.GATES)},
           'preflight': {'acceptance': {'required_true': ['accepted']}},
           'operational_traits': {'t1': preference}, 'prompts': {'review_system': 'Full review', 'review_user': '{conversation}'}}
    monkeypatch.setattr(o.base, 'validate_arm', lambda root, arm: cfg)
    write(root/arm/'config.json', cfg)
    candidate = {'candidate_id': cid, 'source': {'text': 'original'}, 'trait_id': 't1'}
    (root/arm/'candidates.jsonl').write_text(json.dumps(candidate)+'\n', encoding='utf-8')
    record = {**conv, 'trait_id': 't1', 'trait_text': preference, 'scenario_id': cid,
              'source_record_sha256': 'preserve-me', 'rewrite_changes': 'Audit-only untrained text'}
    for name, value in {'result': {'status': 'failed', 'record': record, 'error': 'Old failed critic'},
                        'identity': {'candidate_sha256': o.base.digest(candidate), 'config_sha256': o.base.digest(cfg)},
                        'scenario': {k: conv[k] for k in ('system', 'user')}, 'preflight': {'accepted': True},
                        'independent_exclusion': {'reason': 'Old bad answer'}}.items():
        o.base.save_checkpoint(row/(name+'.json'), value)
    source_sha = o.sha(row/'result.json')
    request = {'model': o.MODEL, 'max_tokens': 12288, 'temperature': .7,
               'messages': [{'role': 'user', 'content': conv['system']+'\n'+conv['user']+'\n'+preference}]}
    entry = {'call_id': 0, 'status': 'settled', 'model': o.MODEL, 'candidate_id': cid,
             'run_root': str(tmp_path/'pilot'), 'arm': 'nonmoral-saved-input-pilot',
             'stage': 'single_saved_revision', 'request_sha256': o.base.digest(request), 'charged_or_reserved_usd': .05}
    raw = {'request': request, 'accounting': entry,
           'response': {'response_model': o.MODEL, 'finish_reason': 'stop',
                        'content': '<reasoning>'+conv['reasoning']+'</reasoning><response>'+conv['response']+'</response>'}}
    budget = tmp_path/'budget'
    monkeypatch.setattr(o, 'ORIGINAL_BUDGET_ROOT', budget)
    ledger = write(budget/'spend.json', [entry])
    raw_path = write(budget/'raw_calls/000000.json', raw)
    receipt = {'call_id': 0, 'request_sha256': o.base.digest(request), 'raw_sha256': o.sha(raw_path),
               'ledger_entry_sha256': o.base.digest(entry)}
    inp = {'candidate_id': cid, 'full_working_preference': preference, 'actual_conversation': conv,
           'request': request, 'request_sha256': o.base.digest(request)}
    result = {'candidate_id': cid, 'status': 'awaiting_independent_full_review', 'automatic_acceptance': False,
              'conversation': conv, 'physical_receipt': receipt}
    input_path, result_path = tmp_path/'pilot/input.json', tmp_path/'pilot/result.json'
    input_path.parent.mkdir()
    o.base.save_checkpoint(input_path, inp)
    o.base.save_checkpoint(result_path, result)
    contract = {'full_constitution_text': constitution, 'constitution_sha256': o.CONSTITUTION_SHA,
                'full_frozen_target_review_prompts': cfg['prompts'], 'full_frozen_acceptance': cfg['acceptance']}
    contract_path = write(tmp_path/'contract.json', contract)
    monkeypatch.setattr(o, 'REVIEW_CONTRACT_SHA', o.sha(contract_path))
    spec = {'source_ref': {'root': str(root), 'arm': arm, 'candidate_id': cid, 'result_sha256': source_sha},
            'review_config_path': str(root/arm/'config.json'), 'review_config_sha256': o.sha(root/arm/'config.json'),
            'review_contract_path': str(contract_path), 'review_contract_sha256': o.sha(contract_path),
            'author_kind': 'single_saved_revision', 'input_path': str(input_path), 'input_sha256': o.sha(input_path),
            'result_path': str(result_path), 'result_sha256': o.sha(result_path), 'raw_path': str(raw_path),
            'physical_receipt': receipt, 'ledger_path': str(ledger)}
    spec_path = write(tmp_path/'spec.json', spec)
    return {'spec': spec, 'spec_path': spec_path, 'row': row, 'out': tmp_path/'dossier', 'conv': conv,
            'raw': raw, 'raw_path': raw_path, 'cfg': cfg, 'ledger': ledger, 'root': root, 'tmp': tmp_path}


def prepare(packet):
    o.prepare(packet['spec_path'], packet['out'])
    return packet['out']/'dossier.json'


def review_packet(packet):
    dossier_path = prepare(packet)
    d, frozen = o.validate_dossier(dossier_path)
    conv = d['conversation']
    review = {'candidate_id': d['source_ref']['candidate_id'], 'source_result_sha256': d['source_ref']['result_sha256'],
              'result_sha256': o.sha(frozen['author_result']), 'conversation_sha256': d['conversation_sha256'],
              'review_contract_sha256': d['review_contract_sha256'], 'constitution_sha256': o.CONSTITUTION_SHA,
              'full_working_preference_sha256': d['full_working_preference_sha256'],
              'request_sha256': d['physical_receipt']['request_sha256'], 'physical_receipt': d['physical_receipt'],
              'decision': 'accept', 'accepted': True, 'issues': [], 'gates': {k: True for k in o.GATES},
              'reviewer_provenance': {'kind': 'independent_codex_agent', 'task': '/root/independent', 'human_review': False},
              'source_first_reason': 'The source is a benign documentation decision.', 'craft_reason': 'Space and clarity conflict.',
              'constitution_compatibility': 'Full nine principles considered.', 'nonmoral_eligibility': 'No moral stakes.',
              'source_quotes': [conv['user']], 'answer_quotes': [conv['response']]}
    if 'author_input' in frozen:
        review['input_sha256'] = o.sha(frozen['author_input'])
    for field in o.FIELDS:
        review[{'system': 'source_system', 'user': 'source_user'}.get(field, field)+'_sha256'] = o.base.digest(conv[field].encode())
    native_input = packet['tmp']/'native.jsonl'
    native_input.write_text(json.dumps({'messages': o.messages(conv), 'metadata': {'scenario_id': 'native-id'}})+'\n', encoding='utf-8')
    native = {'status': 'passed', 'failures': [], 'untruncated': True, 'tokenizer': 'Qwen/Qwen3.6-27B',
              'max_train_tokens': 8192, 'input_sha256': o.sha(native_input), 'rows': 1,
              'diagnostics': [{'id': 'native-id', 'training_tokens': 1100, 'supervised_tokens': 900,
                               'raw_reasoning_tokens': 500, 'raw_assistant_content_tokens': 390}]}
    review_path = write(packet['tmp']/'review.json', review)
    native_path = write(packet['tmp']/'native_audit.json', native)
    duplicate = write(packet['tmp']/'duplicates.json', {'decision': 'distinct', 'scope': 'All currently selected sources'})
    approval = {'approved': True, 'route': o.ROUTE, 'root_actor': '/root', 'reason': 'Reviewed actual source and corrected answer.',
                'dossier_sha256': o.sha(dossier_path), 'conversation_sha256': d['conversation_sha256'],
                'independent_review_sha256': o.sha(review_path), 'native_audit_sha256': o.sha(native_path),
                'native_input_sha256': o.sha(native_input), 'duplicate_check_passed': True,
                'duplicate_evidence_path': str(duplicate), 'duplicate_evidence_sha256': o.sha(duplicate)}
    approval_path = write(packet['tmp']/'approval.json', approval)
    return [dossier_path, review_path, approval_path, native_path, native_input]


def test_revision_adoption_is_portable_preserves_failed_origin(packet):
    args = review_packet(packet)
    before = {str(p): p.read_bytes() for p in packet['row'].glob('*.json')}
    o.accept(*args)
    acceptance = packet['out']/'acceptance.json'
    row, audit = o.validate_accepted(acceptance, o.sha(acceptance))
    assert row['messages'] == o.messages(packet['conv'])
    assert row['metadata']['offline_original_status'] == 'failed'
    assert row['metadata']['source_record_sha256'] == 'preserve-me'
    assert 'rewrite_changes' not in row['metadata']
    assert row['metadata']['offline_original_judges_apply_to_new_answer'] is False
    assert audit['acceptance']['automatic_acceptance'] is False
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    # Portable validation needs archived evidence, not mutable original paths or a live ledger.
    packet['raw_path'].unlink()
    packet['ledger'].unlink()
    o.validate_accepted(acceptance, o.sha(acceptance))
    with pytest.raises(ValueError, match='immutable'):
        o.accept(*args)


@pytest.mark.parametrize('where,key,value', [
    ('response', 'response_model', 'anthropic/claude-haiku-4.5'),
    ('response', 'finish_reason', 'length'), ('accounting', 'status', 'billing_verified_failure'),
    ('accounting', 'candidate_id', 'another_source'),
])
def test_author_rejects_failed_wrong_model_or_source(packet, where, key, value):
    raw = copy.deepcopy(packet['raw']); raw[where][key] = value
    with pytest.raises(ValueError):
        o.validate_author(raw, packet['spec']['physical_receipt'], packet['conv'], 't1_000_v0')


def test_author_rejects_changed_message_bytes(packet):
    conv = {**packet['conv'], 'response': 'An invented replacement.'}
    with pytest.raises(ValueError, match='physical author'):
        o.validate_author(packet['raw'], packet['spec']['physical_receipt'], conv, 't1_000_v0')


def test_shared_ledger_binding_rejects_copy(packet):
    packet['spec']['ledger_path'] = str(write(packet['tmp']/'fake/spend.json', [packet['raw']['accounting']]))
    write(packet['spec_path'], packet['spec'])
    with pytest.raises(ValueError, match='original shared ledger'):
        prepare(packet)


def test_namespaced_campaign_call_accepts_only_exact_source_key(packet):
    spec = packet['spec']
    key = packet['root'].name+'::t1_000_v0'
    inp = o.read(spec['input_path'])
    inp.update(case_key=key, source_ref=spec['source_ref'])
    o.base.save_checkpoint(Path(spec['input_path']), inp)
    packet['raw']['accounting']['candidate_id'] = key
    write(packet['raw_path'], packet['raw'])
    write(packet['ledger'], [packet['raw']['accounting']])
    spec['physical_receipt'].update(raw_sha256=o.sha(packet['raw_path']), ledger_entry_sha256=o.base.digest(packet['raw']['accounting']))
    result = o.read(spec['result_path']); result['physical_receipt'] = spec['physical_receipt']
    o.base.save_checkpoint(Path(spec['result_path']), result)
    spec.update(input_sha256=o.sha(spec['input_path']), result_sha256=o.sha(spec['result_path']))
    write(packet['spec_path'], spec)
    path = prepare(packet)
    d, frozen = o.validate_dossier(path)
    assert o.read(frozen['author_raw'])['accounting']['candidate_id'] == key
    inp['case_key'] = 'different::t1_000_v0'
    with pytest.raises(ValueError, match='case key'):
        o.validate_revision_input(inp, result, spec['source_ref'], packet['conv'], inp['full_working_preference'], spec['physical_receipt'])


def test_input_claim_does_not_replace_actual_preference_in_request(packet):
    inp = o.read(packet['spec']['input_path'])
    inp['request']['messages'][0]['content'] = packet['conv']['system']+'\n'+packet['conv']['user']
    inp['request_sha256'] = o.base.digest(inp['request'])
    with pytest.raises(ValueError, match='omitted'):
        o.validate_revision_input(inp, o.read(packet['spec']['result_path']), packet['spec']['source_ref'],
                                  packet['conv'], inp['full_working_preference'], packet['spec']['physical_receipt'])


@pytest.mark.parametrize('field,value', [('author_kind', 'automatic_pass_alias'), ('implementation_sha256', 'changed-code')])
def test_portable_dossier_does_not_relabel_route_or_implementation(packet, field, value):
    path = prepare(packet)
    d = o.read(path); d[field] = value; o.base.save_checkpoint(path, d)
    with pytest.raises(ValueError, match='route|implementation'):
        o.validate_dossier(path)


@pytest.mark.parametrize('field,value', [('accepted', False), ('issues', ['actual defect']),
                                      ('constitution_sha256', 'wrong'), ('full_working_preference_sha256', 'wrong'),
                                      ('conversation_sha256', 'old-message'), ('answer_quotes', ['Not present'])])
def test_review_failures_cannot_be_adopted(packet, field, value):
    args = review_packet(packet)
    review = o.read(args[1]); review[field] = value; write(args[1], review)
    with pytest.raises(ValueError):
        o.accept(*args)


def test_all_ten_gates_required(packet):
    args = review_packet(packet)
    review = o.read(args[1]); del review['gates']['constitution_compatible']; write(args[1], review)
    with pytest.raises(ValueError, match='every content gate'):
        o.accept(*args)


@pytest.mark.parametrize('field,value', [('approved', False), ('approved', 1), ('conversation_sha256', 'old'),
                                      ('duplicate_check_passed', False), ('root_actor', '/root/some_agent')])
def test_root_approval_is_explicit_and_message_bound(packet, field, value):
    args = review_packet(packet)
    approval = o.read(args[2]); approval[field] = value; write(args[2], approval)
    with pytest.raises(ValueError, match='root acceptance'):
        o.accept(*args)


@pytest.mark.parametrize('field,value', [('input_sha256', 'wrong'), ('untruncated', False),
                                      ('failures', ['too long']), ('max_train_tokens', 9999)])
def test_native_audit_required(packet, field, value):
    args = review_packet(packet)
    audit = o.read(args[3]); audit[field] = value; write(args[3], audit)
    with pytest.raises(ValueError, match='Native'):
        o.accept(*args)


def test_dossier_tampering_fails(packet):
    path = prepare(packet)
    d, frozen = o.validate_dossier(path)
    frozen['author_raw'].write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash differs'):
        o.validate_dossier(path)


def test_source_ineligible_cannot_be_repaired_into_adoption(packet):
    o.base.save_checkpoint(packet['row']/'preflight.json', {'accepted': False})
    with pytest.raises(ValueError, match='ineligible'):
        prepare(packet)


def test_untouched_final_has_fresh_review_not_old_judge_alias(packet):
    spec = packet['spec']
    saved = {k: packet['conv'][k] for k in ('reasoning', 'response')}
    o.base.save_checkpoint(packet['row']/'revise_responses.json', saved)
    raw = packet['raw']
    raw['accounting'].update(run_root=str(packet['root']), arm='nonmoral-advice', stage='revise_responses')
    write(packet['raw_path'], raw); write(packet['ledger'], [raw['accounting']])
    spec.update(author_kind='untouched_saved_final', saved_stage='revise_responses',
                saved_stage_sha256=o.sha(packet['row']/'revise_responses.json'))
    spec['physical_receipt'].update(raw_sha256=o.sha(packet['raw_path']), ledger_entry_sha256=o.base.digest(raw['accounting']))
    write(packet['spec_path'], spec)
    args = review_packet(packet)
    o.accept(*args)
    acceptance = packet['out']/'acceptance.json'
    row, audit = o.validate_accepted(acceptance, o.sha(acceptance))
    assert row['metadata']['offline_author_kind'] == 'untouched_saved_final'
    assert audit['acceptance']['automatic_acceptance'] is False


def test_final_release_exact_quotas_and_no_duplicate_sources():
    rows = []
    for trait, count in o.base.quotas().items():
        for index in range(count):
            cid = f'{trait}_{index}'
            rows.append({'messages': o.messages({'system': 's', 'user': cid, 'reasoning': 'r', 'response': 'a'}),
                         'metadata': {'scenario_id': cid, 'trait_id': trait}})
    assert o.validate_release(rows)['rows'] == 716
    with pytest.raises(ValueError, match='716'):
        o.validate_release(rows[:-1])
    rows[-1]['messages'][1]['content'] = rows[0]['messages'][1]['content']
    with pytest.raises(ValueError, match='Duplicate'):
        o.validate_release(rows)


def test_archived_origin_names_are_cross_platform():
    assert o.origin_name(r'C:\Users\research\output\qualified') == 'qualified'
    assert o.origin_name('C:/Users/research/output/qualified') == 'qualified'
    assert o.origin_name('/research/output/qualified') == 'qualified'
