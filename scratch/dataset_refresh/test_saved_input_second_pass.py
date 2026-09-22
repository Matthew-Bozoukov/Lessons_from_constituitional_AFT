# ABOUTME: Offline tests for the single additional focused revision route and its immutable accounting.
# ABOUTME: Exercise real reservations and raw receipts with a mock sender; never call a provider.
import copy
import json
from pathlib import Path
import pytest
from filelock import FileLock, Timeout
from omegaconf import OmegaConf
from scratch.dataset_refresh import saved_input_second_pass as m
from scratch.dataset_refresh import run as base
from src.infra.endpoints.openrouter import ChatResult


@pytest.fixture
def trial(tmp_path, monkeypatch):
    budget, root, prior = (tmp_path/x for x in ('budget', 'second', 'first'))
    for p in (budget, root, prior/'batches'): p.mkdir(parents=True)
    monkeypatch.setattr(m, 'ROOT', root); monkeypatch.setattr(m, 'BUDGET', budget)
    monkeypatch.setattr(m.first, 'CAMPAIGN', prior)
    baseline = [{'call_id': 0, 'status': 'settled', 'charged_or_reserved_usd': 249.5}]
    base.write_json(budget/'spend.json', baseline)
    (root/'baseline_ledger.json').write_bytes((budget/'spend.json').read_bytes())
    base.save_checkpoint(prior/'manifest.json', {'historical': True})
    base.save_checkpoint(prior/'attempts.json', [])
    closure = tmp_path/'closure.json'; base.save_checkpoint(closure, {'closed': True})
    (root/'first_campaign_closure.json').write_bytes(closure.read_bytes())
    contract = root/'independent_review_contract.json'; base.write_json(contract, {'frozen': True})
    cfg = tmp_path/'policy.yaml'
    OmegaConf.save({'user_approval_quote': m.CONSENT, 'shared_cap_usd': 270,
        'maximum_batch_size': 4, 'maximum_workers': 1, 'maximum_extra_calls_per_source': 1,
        'automatic_retries': 0, 'paid_critic_calls': 0, 'execution_enabled': False,
        'maximum_appended_instruction_bytes': 2048, 'allowed_max_tokens': [8192, 12288]}, cfg)
    base.save_checkpoint(root/'manifest.json', {'config_path': str(cfg), 'config_sha256': m.offline.sha(cfg),
        'code_files': m.code_hashes(), 'user_approval_quote': m.CONSENT, 'shared_cap_usd': 270,
        'budget_root': str(budget), 'baseline_count': 1, 'baseline_ledger_sha256': m.offline.sha(budget/'spend.json'),
        'baseline_entries_digest': base.digest(baseline), 'first_campaign_manifest_sha256': m.offline.sha(prior/'manifest.json'),
        'first_attempts_sha256': m.offline.sha(prior/'attempts.json'), 'first_batches_sha256': {},
        'closure_path': str(closure), 'closure_sha256': m.offline.sha(closure), 'review_contract_sha256': m.offline.sha(contract)})
    source = tmp_path/'source.json'; base.save_checkpoint(source, {'user': 'Help organize notes.'})
    payloads = {}
    for i in range(5):
        cid = f't4_{i:03d}_v0'; key = 'origin::'+cid
        request = {'model': m.MODEL, 'temperature': .7, 'max_tokens': 8192,
            'messages': [{'role': 'system', 'content': 'Assist the user.'}, {'role': 'user', 'content': 'Help organize notes. '+key}]}
        size = len(json.dumps(request['messages'], ensure_ascii=False).encode())
        payloads[key] = {'candidate_id': cid, 'case_key': key, 'request': request, 'request_sha256': base.digest(request),
            'reservation_usd': (1.25*(size+2048)*2+8192*10)/1e6,
            'actual_conversation': {'system': 'Assist the user.', 'user': 'Help organize notes.'},
            'bound_files': {str(source): m.offline.sha(source), str(source.with_suffix('.receipt.json')): m.offline.sha(source.with_suffix('.receipt.json'))},
            'root_second_pass_decision': {'case_key': key}}
    monkeypatch.setattr(m, 'make_payload', lambda decision: copy.deepcopy(payloads[decision['case_key']]))
    return dict(budget=budget, root=root, prior=prior, cfg=cfg, source=source, payloads=payloads, keys=list(payloads), tmp=tmp_path)


def prepare(t, count=1, name='batch', enabled=True, keys=None):
    path = t['tmp']/(name+'.json'); base.write_json(path, [{'case_key': k} for k in (keys or t['keys'][:count])])
    folder = Path(m.prepare_batch(path, name))
    if enabled: base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.offline.sha(folder/'batch.json')})
    return folder


def good(content='<reasoning>Preserve supplied labels.</reasoning><response>Add a short index.</response>', **kw):
    return ChatResult(content=content, prompt_tokens=100, completion_tokens=100,
        finish_reason=kw.get('finish', 'stop'), provider='Anthropic', cost=kw.get('cost', .01), response_model=kw.get('model', m.MODEL))


def no_send(**kw): pytest.fail('Unexpected physical send')


def test_four_calls_and_cached_completion_zero_sends(trial):
    folder = prepare(trial, 4); calls = []
    result = m.execute_batch('batch', send=lambda **kw: (calls.append(kw), good())[1])
    assert len(calls) == 4 and result['automatic_accepted_rows'] == 0
    assert all(r['status'] == 'awaiting_independent_full_review' for r in result['results'])
    assert result['unattempted_case_keys'] == []
    for i in range(4):
        item = base.load_checkpoint(folder/f'{i:02d}.result.json')
        assert item['automatic_acceptance'] is False
        assert item['conversation']['user'] == 'Help organize notes.'
        assert m.offline.sha(folder/f'{i:02d}.raw.json') == item['physical_receipt']['raw_sha256']
    assert m.execute_batch('batch', send=no_send) == result
    with pytest.raises(ValueError, match='already used'): prepare(trial, name='again')


@pytest.mark.parametrize('content,options', [('missing tags', {}),
    ('<reasoning>I am Claude.</reasoning><response>Advice.</response>', {}),
    ('<reasoning>x</reasoning><response>y</response>', {'model': 'wrong'}),
    ('<reasoning>x</reasoning><response>y</response>', {'finish': 'length'})])
def test_failure_retained_one_call_stops_rest(trial, content, options):
    folder = prepare(trial, 4); calls = []
    result = m.execute_batch('batch', send=lambda **kw: (calls.append(kw), good(content, **options))[1])
    assert len(calls) == 1 and result['results'][0]['status'] == 'failed'
    assert len(result['unattempted_case_keys']) == 3
    assert (folder/'00.raw.json').exists()
    assert m.execute_batch('batch', send=no_send) == result


@pytest.mark.parametrize('change', ['source', 'receipt', 'code', 'first_batch', 'prefix', 'decisions'])
def test_changed_binding_prevents_send(trial, change):
    folder = prepare(trial)
    if change in ('source', 'receipt'):
        p = trial['source'] if change == 'source' else trial['source'].with_suffix('.receipt.json')
        p.write_text('{}')
    elif change == 'first_batch': base.write_json(trial['prior']/'batches/new.json', {})
    elif change == 'prefix':
        ledger = m.first.read(trial['budget']/'spend.json'); ledger[0]['charged_or_reserved_usd'] += .1
        base.write_json(trial['budget']/'spend.json', ledger)
    elif change == 'code':
        p = trial['root']/'manifest.json'; manifest = base.load_checkpoint(p)
        manifest['code_files'][str(Path(m.__file__).resolve())] = '0'*64; base.save_checkpoint(p, manifest)
    else: (trial['tmp']/'batch.json').write_text('[]')
    with pytest.raises(ValueError): m.execute_batch('batch', send=no_send)
    assert len(m.first.read(trial['budget']/'spend.json')) == 1


def test_missing_marker_and_duplicate_tail_refused(trial):
    prepare(trial); m.execute_batch('batch', send=lambda **_: good())
    (trial['root']/'attempts.json').unlink()
    with pytest.raises(ValueError, match='attempt history'): m.verify()
    base.save_checkpoint(trial['root']/'attempts.json', trial['keys'][:1])
    ledger = m.first.read(trial['budget']/'spend.json'); ledger.append({**ledger[-1], 'call_id': 2})
    base.write_json(trial['budget']/'spend.json', ledger)
    with pytest.raises(ValueError, match='physical tail'): m.verify()


@pytest.mark.parametrize('enabled', [False, 1])
def test_disabled_or_nonboolean_dispatch_refused(trial, enabled):
    folder = prepare(trial, enabled=False)
    base.write_json(folder/'dispatch.json', {'enabled': enabled, 'batch_sha256': m.offline.sha(folder/'batch.json')})
    with pytest.raises(ValueError, match='disabled'): m.execute_batch('batch', send=no_send)


def test_runtime_limit_and_interrupted_state(trial):
    folder = prepare(trial); batch = base.load_checkpoint(folder/'batch.json')
    batch['maximum_calls'] = 5; base.save_checkpoint(folder/'batch.json', batch)
    base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.offline.sha(folder/'batch.json')})
    with pytest.raises(ValueError, match='bounded batch'): m.execute_batch('batch', send=no_send)
    batch['maximum_calls'] = 1; base.save_checkpoint(folder/'batch.json', batch)
    base.write_json(folder/'dispatch.json', {'enabled': True, 'batch_sha256': m.offline.sha(folder/'batch.json')})
    base.save_checkpoint(folder/'started.json', {})
    with pytest.raises(ValueError, match='already started'): m.execute_batch('batch', send=no_send)


def test_aggregate_headroom_and_budget_lock(trial):
    folder = prepare(trial, 4)
    def locked_send(**kw):
        with pytest.raises(Timeout):
            with FileLock(str(trial['budget']/'spend.lock'), timeout=0): pass
        return good()
    m.execute_batch('batch', send=locked_send)
    # A separate valid fixture is unnecessary: cumulative tail now carries all remaining headroom.
    prepare(trial, name='last', keys=trial['keys'][4:])
    ledger = m.first.read(trial['budget']/'spend.json'); ledger[-1]['charged_or_reserved_usd'] = 20.44
    base.write_json(trial['budget']/'spend.json', ledger)
    with pytest.raises(ValueError, match='headroom'): m.execute_batch('last', send=no_send)


def review_fixture():
    conv = {'system': 'A source.', 'user': 'A question.', 'reasoning': 'A false premise.', 'response': 'A recommendation.'}
    inp = {'candidate_id': 't1_001', 'source_ref': {'result_sha256': 'source'}, 'full_working_preference': 'Full preference', 'request_sha256': 'request'}
    result = {'conversation': conv, 'physical_receipt': {'call_id': 1}}
    review = {'candidate_id': inp['candidate_id'], 'source_result_sha256': 'source', 'result_sha256': 'result',
        'input_sha256': 'input', 'conversation_sha256': base.digest(conv), 'review_contract_sha256': 'contract',
        'constitution_sha256': m.offline.CONSTITUTION_SHA, 'full_working_preference_sha256': base.digest(b'Full preference'),
        'request_sha256': 'request', 'physical_receipt': result['physical_receipt'], 'accepted': False, 'decision': 'hold',
        'full_read': True, 'source_eligible': True, 'reviewer_provenance': {'kind': 'independent_codex_agent', 'human_review': False},
        'gates': {k: True for k in m.offline.GATES}, 'issues': [{'severity': 'material', 'reason': 'Changed source premise.', 'quotes': ['A false premise.']}]}
    review['gates'][m.offline.GATES[0]] = False
    for field in m.offline.FIELDS:
        review[{'system': 'source_system', 'user': 'source_user'}.get(field, field)+'_sha256'] = base.digest(conv[field].encode())
    return review, inp, result


def test_review_exact_binding_and_uncertainty_preserved():
    review, inp, result = review_fixture()
    m.validate_material_review(review, inp, result, 'result', 'input', 'contract')
    review['decision'] = 'uncertain'; review['gates'] = {}; review['issues'][0]['severity'] = 'uncertain'
    with pytest.raises(ValueError): m.validate_material_review(review, inp, result, 'result', 'input', 'contract')
    decision = {'root_selected_material_cleanup': True, 'material_cleanup_reason': 'Delete the disputed invented premise from both blocks.'}
    m.validate_material_review(review, inp, result, 'result', 'input', 'contract', decision)
    assert review['decision'] == 'uncertain' and review['issues'][0]['severity'] == 'uncertain'


@pytest.mark.parametrize('field,value', [('accepted', True), ('constitution_sha256', 'wrong'), ('input_sha256', 'wrong'),
    ('full_read', False), ('source_eligible', False), ('issues', [{'severity': 'material', 'reason': 'x', 'quotes': ['not in answer']}])])
def test_bad_review_cannot_authorize_new_call(field, value):
    review, inp, result = review_fixture(); review[field] = value
    with pytest.raises(ValueError): m.validate_material_review(review, inp, result, 'result', 'input', 'contract')


@pytest.mark.parametrize('field,value', [('status', 'accepted'), ('automatic_acceptance', True), ('additional_revision_number', 2)])
def test_cached_fake_acceptance_flags_rejected(trial, field, value):
    folder = prepare(trial); m.execute_batch('batch', send=lambda **_: good())
    item = base.load_checkpoint(folder/'00.result.json'); item[field] = value
    base.save_checkpoint(folder/'00.result.json', item)
    with pytest.raises(ValueError, match='identity'): m.execute_batch('batch', send=no_send)


def test_exception_keeps_reservation_and_no_next_call(trial):
    folder = prepare(trial, 4)
    def failure(**kw): raise RuntimeError('Unknown provider outcome')
    summary = m.execute_batch('batch', send=failure)
    assert summary['results'][0]['status'] == 'failed' and len(summary['unattempted_case_keys']) == 3
    ledger = m.first.read(trial['budget']/'spend.json')
    assert len(ledger) == 2 and ledger[-1]['status'] == 'uncertain_failure'
    assert ledger[-1]['charged_or_reserved_usd'] > 0
    assert (folder/'00.raw.json').exists()
    assert m.execute_batch('batch', send=no_send) == summary


@pytest.fixture
def closing(trial, monkeypatch):
    newroot = trial['tmp']/'not_initialized'
    monkeypatch.setattr(m, 'ROOT', newroot)
    monkeypatch.setattr(m.first, 'BASELINE_COUNT', 1)
    monkeypatch.setattr(m.first, 'verify_campaign', lambda: None)
    monkeypatch.setattr(m.first, 'completed_summary', lambda folder, batch: {'closed': True})
    monkeypatch.setattr(m, 'code_hashes', lambda: {})
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw: 'a'*40)
    base.write_json(trial['prior']/'independent_review_contract.json', {'frozen': True})
    closure = trial['tmp']/'actual_closure.json'
    base.save_checkpoint(closure, {'root_actor': '/root', 'first_campaign_closed': True,
        'first_campaign_manifest_sha256': m.offline.sha(trial['prior']/'manifest.json'),
        'shared_ledger_sha256': m.offline.sha(trial['budget']/'spend.json'), 'shared_ledger_count': 1,
        'reason': 'All prepared first attempts completed; root closes new first dispatch.'})
    return trial, closure, newroot


def test_initializer_binds_actual_variable_closing_prefix(closing):
    t, closure, newroot = closing
    assert m.initialize(t['cfg'], closure) == str(newroot)
    manifest, ledger, attempts = m.verify()
    assert manifest['baseline_count'] == 1 and len(ledger) == 1 and attempts == []
    assert manifest['closure_sha256'] == m.offline.sha(closure)
    with pytest.raises(ValueError, match='already exists'): m.initialize(t['cfg'], closure)


@pytest.mark.parametrize('change', ['pending_batch', 'unsettled', 'wrong_closure', 'orphan_first_attempt'])
def test_initializer_rejects_unclosed_first_campaign(closing, change):
    t, closure, newroot = closing
    if change == 'pending_batch': (t['prior']/'batches/notdone').mkdir()
    elif change == 'orphan_first_attempt': base.save_checkpoint(t['prior']/'attempts.json', ['origin::missing'])
    else:
        if change == 'unsettled':
            ledger = m.first.read(t['budget']/'spend.json'); ledger[0]['status'] = 'reserved'
            base.write_json(t['budget']/'spend.json', ledger)
        c = base.load_checkpoint(closure)
        c['shared_ledger_sha256'] = m.offline.sha(t['budget']/'spend.json')
        if change == 'wrong_closure': c['first_campaign_closed'] = False
        base.save_checkpoint(closure, c)
    with pytest.raises(ValueError): m.initialize(t['cfg'], closure)
    assert not newroot.exists()
