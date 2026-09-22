# ABOUTME: Offline integrity tests for the separately authorized saved-input completion campaign.
# ABOUTME: Exercise real cumulative reservations, receipts, one-attempt markers and dispatch guards with a send spy.
import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scratch.dataset_refresh import saved_input_completion as m
from scratch.dataset_refresh import run as base
from src.infra.endpoints.openrouter import ChatResult


@pytest.fixture
def trial(tmp_path, monkeypatch):
    budget = tmp_path/'budget'
    campaign = tmp_path/'campaign'
    budget.mkdir()
    campaign.mkdir()
    monkeypatch.setattr(m, 'BUDGET', budget)
    monkeypatch.setattr(m, 'CAMPAIGN', campaign)
    monkeypatch.setattr(m, 'BASELINE_COUNT', 1)
    baseline = [{'call_id': 0, 'status': 'settled', 'charged_or_reserved_usd': 249.5}]
    base.write_json(budget/'spend.json', baseline)
    baseline_sha = m.sha(budget/'spend.json')
    monkeypatch.setattr(m, 'BASELINE_SHA', baseline_sha)
    (campaign/'baseline_ledger.json').write_bytes((budget/'spend.json').read_bytes())
    review = tmp_path/'review.json'
    base.write_json(review, {'full_constitution_text': 'Frozen review contract'})
    (campaign/'independent_review_contract.json').write_bytes(review.read_bytes())
    qualified = tmp_path/'qualified.json'
    base.write_json(qualified, {})
    inventory = [{'root': 'fixture', 'candidate_id': f't4_{i:03d}_v0',
                  'classification': sorted(m.CLASSES)[0],
                  'one_revision_reservation_upper_usd': 16.45/96} for i in range(96)]
    inventory_path = tmp_path/'inventory.jsonl'
    base.write_rows(inventory_path, inventory)
    config = {'user_approval_quote': m.CONSENT, 'shared_cap_usd': 270,
              'maximum_physical_calls': 96, 'maximum_workers': 4,
              'maximum_appended_instruction_bytes': 2048, 'paid_critic_calls': 0,
              'automatic_retries': 0, 'inventory_path': str(inventory_path),
              'inventory_sha256': m.sha(inventory_path),
              'qualified_config_path': str(qualified), 'qualified_config_sha256': m.sha(qualified),
              'review_contract_path': str(review), 'review_contract_sha256': m.sha(review)}
    config_path = tmp_path/'policy.yaml'
    OmegaConf.save(config, config_path)
    base.save_checkpoint(campaign/'manifest.json', {
        'config_path': str(config_path), 'config_sha256': m.sha(config_path),
        'code_files': m.code_hashes(), 'baseline_ledger_sha256': baseline_sha,
        'baseline_entries_digest': base.digest(baseline), 'baseline_count': 1,
        'budget_root': str(budget), 'shared_cap_usd': 270, 'user_approval_quote': m.CONSENT})
    source = tmp_path/'source.json'
    base.save_checkpoint(source, {'system': 'Assist the user.', 'user': 'Help organize notes.'})
    source_receipt = source.with_suffix('.receipt.json')
    payloads = {}
    for item in inventory:
        key = item['root']+'::'+item['candidate_id']
        request = {'model': m.MODEL, 'temperature': .7, 'max_tokens': 12288,
                   'messages': [{'role': 'system', 'content': 'Assist the user.'},
                                {'role': 'user', 'content': 'Help organize notes. '+key}]}
        size = len(json.dumps(request['messages'], ensure_ascii=False).encode())
        payloads[key] = {'candidate_id': item['candidate_id'], 'case_key': key,
                         'request': request, 'request_sha256': base.digest(request),
                         'reservation_usd': (1.25*(size+2048)*2+12288*10)/1e6,
                         'actual_conversation': {'system': 'Assist the user.', 'user': 'Help organize notes.'},
                         'bound_files': {str(source): m.sha(source), str(source_receipt): m.sha(source_receipt)},
                         'root_source_decision': {'case_key': key}}
    # Isolate origin rendering only; policy, dispatch, reservations, physical send, accounting,
    # raw lineage and checkpoint validation remain real throughout these tests.
    monkeypatch.setattr(m, 'make_payload', lambda cfg, item, decision:
                        copy.deepcopy(payloads[item['root']+'::'+item['candidate_id']]))
    return {'campaign': campaign, 'budget': budget, 'config': config_path, 'source': source,
            'payloads': payloads, 'keys': list(payloads), 'tmp': tmp_path}


def prepare(trial, count=1, name='batch', keys=None, enabled=True):
    keys = keys or trial['keys'][:count]
    decisions = trial['tmp']/(name+'_decisions.json')
    base.write_json(decisions, [{'case_key': k} for k in keys])
    folder = Path(m.prepare_batch(decisions, name))
    if enabled:
        base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.sha(folder/'batch.json')})
    return folder


def good(content='<reasoning>Use the supplied structure and preserve the labels.</reasoning><response>Keep the labels and add a short index.</response>', **kw):
    return ChatResult(content=content, prompt_tokens=100, completion_tokens=100,
                      finish_reason=kw.get('finish', 'stop'), provider='Anthropic',
                      cost=kw.get('cost', .01), response_model=kw.get('model', m.MODEL))


def no_send(**kwargs):
    pytest.fail('Unexpected physical send')


def test_four_physical_calls_preserve_source_and_never_auto_accept(trial):
    folder = prepare(trial, 4)
    calls = []
    def send(**kw):
        calls.append(kw)
        return good()
    result = m.execute_batch('batch', send=send)
    assert len(calls) == 4
    assert result['automatic_accepted_rows'] == 0
    assert all(x['status'] == 'awaiting_independent_full_review' for x in result['results'])
    assert len(m.read(trial['budget']/'spend.json')) == 5
    for index in range(4):
        saved = base.load_checkpoint(folder/f'{index:02d}.result.json')
        assert saved['automatic_acceptance'] is False
        assert saved['conversation']['user'] == 'Help organize notes.'
        raw = folder/f'{index:02d}.raw.json'
        assert m.sha(raw) == saved['physical_receipt']['raw_sha256']
        assert base.digest(m.read(raw)['request']) == saved['physical_receipt']['request_sha256']
    assert m.execute_batch('batch', send=no_send) == result


@pytest.mark.parametrize('content,options', [
    ('missing tags', {}),
    ('<reasoning>x</reasoning><response>x</response><response>y</response>', {}),
    ('<reasoning>I am Claude.</reasoning><response>Advice.</response>', {}),
    ('<reasoning>x</reasoning><response>[TODO fill]</response>', {}),
    ('<reasoning>x</reasoning><response>y</response>', {'model': 'anthropic/claude-haiku-4.5'}),
    ('<reasoning>x</reasoning><response>y</response>', {'finish': 'length'}),
])
def test_bad_answer_one_attempt_and_no_false_acceptance(trial, content, options):
    folder = prepare(trial)
    calls = []
    result = m.execute_batch('batch', send=lambda **kw: (calls.append(kw), good(content, **options))[1])
    assert len(calls) == 1
    assert result['results'][0]['status'] == 'failed'
    assert result['automatic_accepted_rows'] == 0
    assert (folder/'00.raw.json').exists()
    assert m.execute_batch('batch', send=no_send) == result
    prepare(trial, name='again')
    with pytest.raises(ValueError, match='already attempted'):
        m.execute_batch('again', send=no_send)


@pytest.mark.parametrize('field,value', [('shared_cap_usd', 271), ('user_approval_quote', 'Continue'),
    ('maximum_workers', 5), ('maximum_physical_calls', 97), ('automatic_retries', 1), ('paid_critic_calls', 1)])
def test_exact270_authority_required(trial, field, value):
    cfg = OmegaConf.to_container(OmegaConf.load(trial['config']))
    cfg[field] = value
    OmegaConf.save(cfg, trial['config'])
    with pytest.raises(ValueError):
        m.Authorized270Client(send=no_send)
    with pytest.raises(ValueError, match='250'):
        base.BudgetClient(trial['budget'], 270, {m.MODEL}, send=no_send)


@pytest.mark.parametrize('change', ['source', 'receipt', 'request', 'code', 'decisions'])
def test_changed_bound_data_refused_before_send(trial, change):
    folder = prepare(trial)
    if change in {'source', 'receipt'}:
        path = trial['source'] if change == 'source' else trial['source'].with_suffix('.receipt.json')
        path.write_text('{}', encoding='utf-8')
    elif change == 'request':
        path = folder/'00.input.json'
        item = base.load_checkpoint(path)
        item['request']['messages'][0]['content'] += ' edited'
        base.save_checkpoint(path, item)
        batch = base.load_checkpoint(folder/'batch.json')
        batch['inputs'][0]['sha256'] = m.sha(path)
        base.save_checkpoint(folder/'batch.json', batch)
        base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.sha(folder/'batch.json')})
    elif change == 'code':
        path = trial['campaign']/'manifest.json'
        manifest = base.load_checkpoint(path)
        manifest['code_files'][str(Path(m.__file__).resolve())] = '0'*64
        base.save_checkpoint(path, manifest)
    else:
        (trial['tmp']/'batch_decisions.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError):
        m.execute_batch('batch', send=no_send)
    assert len(m.read(trial['budget']/'spend.json')) == 1
    assert not (folder/'started.json').exists()


def test_missing_attempt_history_cannot_reissue_paid_candidate(trial):
    prepare(trial)
    m.execute_batch('batch', send=lambda **_: good())
    (trial['campaign']/'attempts.json').unlink()
    with pytest.raises(ValueError, match='Missing attempt history'):
        m.verify_campaign()
    (trial['campaign']/'attempts.receipt.json').unlink()
    with pytest.raises(ValueError, match='Missing attempt history'):
        m.verify_campaign()


def test_orphan_attempt_receipt_blocks_fresh_calls(trial):
    base.save_checkpoint(trial['campaign']/'attempts.json', [])
    (trial['campaign']/'attempts.json').unlink()
    with pytest.raises(ValueError, match='orphan receipt'):
        m.verify_campaign()


def test_duplicate_physical_ledger_candidate_is_rejected(trial):
    prepare(trial)
    m.execute_batch('batch', send=lambda **_: good())
    ledger = m.read(trial['budget']/'spend.json')
    ledger.append({**ledger[-1], 'call_id': 2})
    base.write_json(trial['budget']/'spend.json', ledger)
    with pytest.raises(ValueError, match='Unexpected'):
        m.verify_campaign()


def test_disabled_or_interrupted_dispatch_cannot_send(trial):
    folder = prepare(trial, enabled=False)
    with pytest.raises(ValueError, match='not enabled'):
        m.execute_batch('batch', send=no_send)
    base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.sha(folder/'batch.json')})
    base.save_checkpoint(folder/'started.json', {'case_keys': trial['keys'][:1]})
    with pytest.raises(ValueError, match='Interrupted'):
        m.execute_batch('batch', send=no_send)


def test_whole_batch_reservation_enforces_cumulative270_cap(trial):
    # Valid campaign tail spends almost all the extension before this batch.
    key = trial['keys'][-1]
    ledger = m.read(trial['budget']/'spend.json')
    ledger.append({'call_id': 1, 'status': 'settled', 'charged_or_reserved_usd': 20.4,
                   'candidate_id': key, 'model': m.MODEL, 'arm': m.ARM,
                   'run_root': str(m.CAMPAIGN), 'stage': 'single_saved_revision'})
    base.write_json(trial['budget']/'spend.json', ledger)
    base.save_checkpoint(trial['campaign']/'attempts.json', [key])
    folder = prepare(trial, 4)
    with pytest.raises(ValueError, match='Whole batch'):
        m.execute_batch('batch', send=no_send)
    assert not (folder/'started.json').exists()


def test_manifest_authorized_accounting_can_cross_old250_ceiling(trial):
    key = trial['keys'][-1]
    ledger = m.read(trial['budget']/'spend.json')
    ledger.append({'call_id': 1, 'status': 'settled', 'charged_or_reserved_usd': .7,
                   'candidate_id': key, 'model': m.MODEL, 'arm': m.ARM,
                   'run_root': str(m.CAMPAIGN), 'stage': 'single_saved_revision'})
    base.write_json(trial['budget']/'spend.json', ledger)
    base.save_checkpoint(trial['campaign']/'attempts.json', [key])
    prepare(trial)
    calls = []
    result = m.execute_batch('batch', send=lambda **kw: (calls.append(kw), good())[1])
    assert len(calls) == 1
    assert 250 < result['shared_exposure_usd'] < 270


@pytest.mark.parametrize('change', ['source', 'source_receipt', 'raw', 'result_bytes',
                                  'result_answer_resealed', 'result_identity_resealed',
                                  'missing_physical_receipt', 'summary_resealed'])
def test_completed_resume_revalidates_evidence_without_another_call(trial, change):
    folder = prepare(trial)
    calls = []
    m.execute_batch('batch', send=lambda **kw: (calls.append(kw), good())[1])
    assert len(calls) == 1
    if change in {'source', 'source_receipt'}:
        path = trial['source'] if change == 'source' else trial['source'].with_suffix('.receipt.json')
        path.write_text('{}', encoding='utf-8')
    elif change == 'raw':
        (folder/'00.raw.json').write_text('{}', encoding='utf-8')
    elif change == 'result_bytes':
        (folder/'00.result.json').write_text('{}', encoding='utf-8')
    elif change == 'summary_resealed':
        summary = base.load_checkpoint(folder/'summary.json')
        summary['automatic_accepted_rows'] = 1
        base.save_checkpoint(folder/'summary.json', summary)
    else:
        result = base.load_checkpoint(folder/'00.result.json')
        if change == 'result_answer_resealed':
            result['conversation']['response'] = 'An answer the author never wrote.'
        elif change == 'result_identity_resealed':
            result['case_key'] = trial['keys'][1]
        else:
            del result['physical_receipt']
        base.save_checkpoint(folder/'00.result.json', result)
    before = (trial['budget']/'spend.json').read_bytes()
    with pytest.raises((ValueError, base.BudgetStop)):
        m.execute_batch('batch', send=no_send)
    assert (trial['budget']/'spend.json').read_bytes() == before
    assert len(calls) == 1


@pytest.mark.parametrize('change', ['raw_request', 'copied_raw'])
def test_changed_raw_cannot_claim_verified_proposal_lineage(trial, monkeypatch, change):
    folder = prepare(trial)
    if change == 'raw_request':
        original_write = base.write_json
        def write(path, value):
            if Path(path).parent.name == 'raw_calls' and 'response' in value:
                value = copy.deepcopy(value)
                value['request']['messages'][0]['content'] += ' altered after call'
            return original_write(path, value)
        monkeypatch.setattr(base, 'write_json', write)
    else:
        original_copy = m.shutil.copyfile
        def copy_file(src, dst, *args, **kwargs):
            result = original_copy(src, dst, *args, **kwargs)
            if str(dst).endswith('.raw.json'):
                Path(dst).write_text('{}', encoding='utf-8')
            return result
        monkeypatch.setattr(m.shutil, 'copyfile', copy_file)
    try:
        summary = m.execute_batch('batch', send=lambda **_: good())
    except ValueError:
        pass
    else:
        assert summary['results'][0]['status'] != 'awaiting_independent_full_review'
    result_path = folder/'00.result.json'
    if result_path.exists():
        assert base.load_checkpoint(result_path)['status'] != 'awaiting_independent_full_review'


@pytest.mark.parametrize('mode', ['five', 'duplicate', 'count', 'reservation'])
def test_runtime_revalidates_batch_bounds(trial, mode):
    folder = prepare(trial)
    batch = base.load_checkpoint(folder/'batch.json')
    if mode in {'five', 'duplicate'}:
        batch['inputs'] *= 5 if mode == 'five' else 2
        batch['maximum_calls'] = len(batch['inputs'])
        batch['reservation_sum_usd'] *= len(batch['inputs'])
    elif mode == 'count':
        batch['maximum_calls'] = 2
    else:
        batch['reservation_sum_usd'] = 0
    base.save_checkpoint(folder/'batch.json', batch)
    base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.sha(folder/'batch.json')})
    with pytest.raises(ValueError, match='Runtime batch'):
        m.execute_batch('batch', send=no_send)


def test_operational_conversation_implementation_is_frozen():
    assert str(m.REPO/'scratch/dataset_refresh/per_row.py') in m.code_hashes()


def test_real_saved_payload_rejects_empty_independent_review(tmp_path):
    inventory = m.REPO/'output/2026-09-15_dataset_refresh_saved_input_pilot_review/completion_inventory/inventory.jsonl'
    if not inventory.exists():
        pytest.skip('Repository-local frozen saved-source inventory unavailable')
    item = next(x for x in base.read_rows(inventory)
                if x['root'].endswith('_sonnet_qualified') and x['candidate_id'] == 't4_011_v0')
    row = m.REPO/'output'/item['root']/item['arm']/'records'/item['candidate_id']
    original = base.load_checkpoint(row/'result.json')
    saved = base.load_checkpoint(row/(item['selected_input_stage']+'.json'))
    record = dict(original['record'])
    pair = ('draft_reasoning', 'draft_response') if item['selected_input_stage'] == 'draft_responses' else ('reasoning', 'response')
    record.update(reasoning=saved[pair[0]], response=saved[pair[1]])
    empty = tmp_path/'empty_review.json'
    base.save_checkpoint(empty, {})
    decision = {'case_key': item['root']+'::'+item['candidate_id'],
                'source_review_path': str(empty), 'source_review_sha256': m.sha(empty),
                'source_eligible': True, 'decision': 'one_revision',
                'source_conversation_sha256': base.digest(m.conversation(record)),
                'instruction': 'Improve the grounded advice using the actual supplied facts.'}
    cfg = {'qualified_config_path': str(m.REPO/'output/2026-09-15_dataset_refresh_sonnet_qualified/nonmoral-advice/config.json')}
    with pytest.raises(ValueError, match='review|decision|evidence|bound'):
        m.make_payload(cfg, item, decision)

    # A normalized root receipt and the independent exact-row evidence together
    # must suffice; the test only renders a request and never dispatches it.
    ref = {'root': str(row.parents[2]), 'arm': item['arm'], 'candidate_id': item['candidate_id'],
           'result_sha256': item['source_files_sha256']['result.json']}
    entry = {'candidate_id': item['candidate_id'], 'result_sha256': ref['result_sha256'],
             'conversation_sha256': decision['source_conversation_sha256'],
             'selected_input_stage': item['selected_input_stage'],
             'selected_input_sha256': item['selected_input_sha256'],
             'full_system_user_reasoning_final_read': True, 'source_eligible': True}
    independent = tmp_path/'independent.json'
    base.write_json(independent, {'rows': [entry]})
    review = {'source_ref': ref, 'conversation_sha256': decision['source_conversation_sha256'],
              'source_eligible': True, 'root_actor': '/root',
              'review_scope': 'full_source_and_saved_answer', 'decision': 'one_revision',
              'material_defect': 'Exact factual premise requires correction.',
              'independent_review_path': str(independent), 'independent_review_sha256': m.sha(independent),
              'independent_row_selector': ['rows', 0], 'independent_row_sha256': base.digest(entry)}
    base.save_checkpoint(empty, review)
    decision['source_review_sha256'] = m.sha(empty)
    payload = m.make_payload(cfg, item, decision)
    assert payload['actual_conversation'] == m.conversation(record)
    assert payload['request']['model'] == m.MODEL
    assert payload['request']['messages'][-1]['content'].endswith(decision['instruction'])
    assert payload['reservation_usd'] <= item['one_revision_reservation_upper_usd']
    assert payload['bound_files'][str(empty.with_suffix('.receipt.json'))] == m.sha(empty.with_suffix('.receipt.json'))
    assert payload['bound_files'][str(independent)] == m.sha(independent)
    for field in ('result_sha256', 'conversation_sha256', 'selected_input_sha256',
                  'full_system_user_reasoning_final_read', 'source_eligible'):
        invalid = {**entry, field: False if field.endswith('read') or field == 'source_eligible' else '0'*64}
        base.write_json(independent, {'rows': [invalid]})
        invalid_review = {**review, 'independent_review_sha256': m.sha(independent),
                          'independent_row_sha256': base.digest(invalid)}
        base.save_checkpoint(empty, invalid_review)
        decision['source_review_sha256'] = m.sha(empty)
        with pytest.raises(ValueError, match='Independent review'):
            m.make_payload(cfg, item, decision)
