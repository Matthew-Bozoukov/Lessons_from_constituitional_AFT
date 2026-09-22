# ABOUTME: Offline tests for exact four-case saved-input dispatch, budget binding and no repeated physical calls.
# ABOUTME: Mock all model calls while retaining the real shared-budget implementation and immutable output receipts.
import copy
import json
from pathlib import Path

import pytest
import yaml

from scratch.dataset_refresh import saved_input_pilot as m
from scratch.dataset_refresh import run as base
from src.infra.endpoints.openrouter import ChatResult


PACKET = m.REPO/'output/2026-09-15_dataset_refresh_correction_review/process/nonmoral_repair_pilot_proposal/proposed_config.yaml'


@pytest.fixture(scope='module')
def packet(tmp_path_factory):
    """Exercise dispatch on invented inputs without requiring a past paid run."""
    root = tmp_path_factory.mktemp('saved-input-packet')
    contract = root/'independent_review_contract.json'
    base.write_json(contract, {'full_frozen_acceptance': {'gates': ['constitution_compatible']}})
    payloads = []
    for cid in m.IDS:
        request = dict(model=m.MODEL, temperature=0.7, max_tokens=12288,
                       messages=[{'role': 'user', 'content': cid + ': ' + 'x' * 4000}])
        payloads.append(dict(candidate_id=cid, request=request,
                             request_sha256=base.digest(request),
                             reservation_usd=m.MAX_RESERVATION/len(m.IDS),
                             actual_conversation=dict(system='You give practical advice.',
                                                      user='Help me choose a hobby project.')))
    proposal = dict(rows=[dict(candidate_id=cid) for cid in m.IDS],
                    review_contract_sha256=m.sha(contract))
    return {}, proposal, payloads, root/'proposal.json'


@pytest.fixture
def trial(tmp_path, monkeypatch, packet):
    config, proposal, payloads, source_path = copy.deepcopy(packet)
    budget = tmp_path/'budget'
    budget.mkdir()
    monkeypatch.setattr(m, 'ORIGINAL_BUDGET_ROOT', budget)
    entries = [{'call_id': 0, 'status': 'settled', 'charged_or_reserved_usd': 249.2113677}]
    base.write_json(budget/'spend.json', entries)
    root = tmp_path/'source'
    root.mkdir()
    base.write_json(root/'run_meta.json', {'budget_root': str(budget)})
    config_path = tmp_path/'proposal_config.yaml'
    config_path.write_text('execution_enabled: false\n', encoding='utf-8')
    for row in proposal['rows']:
        row['root'] = str(root)
    proposal['budget_ledger_sha256'] = m.sha(budget/'spend.json')
    proposal['closed_call_count'] = 1
    monkeypatch.setattr(m, 'verify_packet', lambda _: (config, proposal, payloads, source_path))
    output = tmp_path/'pilot'
    m.prepare(config_path, output)
    return output, budget, payloads


def enable(output):
    p = output/'dispatch.yaml'
    d = yaml.safe_load(p.read_text())
    d.update(execution_enabled=True, dispatch_authorized=True)
    p.write_text(yaml.safe_dump(d), encoding='utf-8')


def good(content='<reasoning>Thought about the source.</reasoning><response>Advice.</response>', finish='stop', cost=.04):
    return ChatResult(content=content, prompt_tokens=5000, completion_tokens=1000,
                      finish_reason=finish, provider='Anthropic', cost=cost, response_model=m.MODEL)


def test_actual_four_sources_requests_and_full_constitution_validate_offline():
    if not PACKET.is_file():
        pytest.skip('Optional historical replay: archived September-15 proposal not downloaded')
    _, proposal, payloads, path = m.verify_packet(PACKET)
    assert tuple(x['candidate_id'] for x in payloads) == m.IDS
    assert sum(x['reservation_usd'] for x in payloads) == m.MAX_RESERVATION
    assert proposal['existing_conservative_charge_usd']+m.MAX_RESERVATION < 250
    review = m.read(path.parent/'independent_review_contract.json')
    assert len(review['full_constitution_text']) > 20000
    assert 'constitution_compatible' in review['full_frozen_acceptance']['gates']


def test_disabled_dispatch_makes_no_calls(trial):
    output, budget, _ = trial
    before = (budget/'spend.json').read_bytes()
    with pytest.raises(ValueError, match='disabled'):
        m.execute(output, send=lambda **_: pytest.fail('unexpected call'))
    assert (budget/'spend.json').read_bytes() == before


def test_exact_four_calls_no_critics_no_adoption_and_no_repeat(trial):
    output, budget, payloads = trial
    enable(output)
    calls = []
    def send(**kw):
        calls.append(kw)
        return good()
    result = m.execute(output, send=send)
    assert result['status'] == 'awaiting_independent_full_review'
    assert calls == [x['request'] for x in payloads]
    assert len(calls) == result['physical_calls_added'] == 4
    assert result['automatic_accepted_rows'] == 0
    assert all(x['audit']['changes_format_omission'] for x in result['results'])
    assert all(x['status'] == 'awaiting_independent_full_review' for x in result['results'])
    assert {x['stage'] for x in m.read(budget/'spend.json')[1:]} == {'single_saved_revision'}
    for row, payload in zip(result['results'], payloads):
        assert row['conversation']['system'] == payload['actual_conversation']['system']
        assert row['conversation']['user'] == payload['actual_conversation']['user']
        raw = output/(row['candidate_id']+'.raw.json')
        assert m.sha(raw) == row['physical_receipt']['raw_sha256']
        assert m.read(raw)['request'] == payload['request']
    assert m.execute(output, send=lambda **_: pytest.fail('repeat')) == result


@pytest.mark.parametrize('failure', ['exception', 'length', 'missing_tag', 'duplicate_tag', 'unknown_finish', 'bound_exceeded',
                                    'wrong_model', 'identity_leak', 'placeholder'])
def test_any_failed_output_stops_without_retries_and_preserves_raw(trial, failure):
    output, budget, _ = trial
    enable(output)
    calls = []
    def send(**kw):
        calls.append(kw)
        if failure == 'exception':
            raise RuntimeError('uncertain network outcome')
        if failure == 'length':
            return good(finish='length')
        if failure == 'missing_tag':
            return good('<reasoning>Complete reasoning.</reasoning>')
        if failure == 'duplicate_tag':
            return good('<reasoning>a</reasoning><response>b</response><response>c</response>')
        if failure == 'unknown_finish':
            return good(finish='error')
        if failure == 'wrong_model':
            result = good()
            result.response_model = 'anthropic/claude-haiku-4.5'
            return result
        if failure == 'identity_leak':
            return good('<reasoning>I am Claude.</reasoning><response>Advice.</response>')
        if failure == 'placeholder':
            return good('<reasoning>Reason.</reasoning><response>[TODO fill]</response>')
        return good(cost=10)
    result = m.execute(output, send=send)
    assert result['status'] == 'stopped_no_retry'
    assert result['physical_calls_added'] == len(calls) == 1
    assert (output/(m.IDS[0]+'.raw.json')).exists()
    assert result['automatic_accepted_rows'] == 0
    assert m.execute(output, send=lambda **_: pytest.fail('retry')) == result
    entry = m.read(budget/'spend.json')[-1]
    if failure == 'exception':
        assert entry['status'] == 'uncertain_failure'
        assert entry['charged_or_reserved_usd'] == entry['reserve_usd']
    if failure == 'bound_exceeded':
        assert entry['status'] == 'bound_exceeded'
        assert entry['charged_or_reserved_usd'] == 10


@pytest.mark.parametrize('change', ['ledger', 'config', 'input', 'code', 'review_contract'])
def test_changed_bound_inputs_stop_before_dispatch(trial, change):
    output, budget, _ = trial
    enable(output)
    manifest = base.load_checkpoint(output/'manifest.json')
    if change == 'ledger':
        base.write_json(budget/'spend.json', [{'call_id': 0, 'status': 'settled', 'charged_or_reserved_usd': 249.22}])
    elif change == 'config':
        Path(manifest['config_path']).write_text('changed')
    elif change == 'input':
        p = output/(m.IDS[0]+'.input.json')
        d = base.load_checkpoint(p)
        d['request']['messages'][0]['content'] += 'changed'
        base.save_checkpoint(p, d)
    elif change == 'review_contract':
        (output/'independent_review_contract.json').write_text('{}')
    else:
        manifest['code_files'][str(Path(m.__file__).resolve())] = '0'*64
        base.save_checkpoint(output/'manifest.json', manifest)
        d = yaml.safe_load((output/'dispatch.yaml').read_text())
        d['manifest_sha256'] = m.sha(output/'manifest.json')
        (output/'dispatch.yaml').write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):
        m.execute(output, send=lambda **_: pytest.fail('unexpected call'))


def test_every_row_rechecks_source_hashes_and_stops_on_change(trial, monkeypatch):
    output, _, _ = trial
    enable(output)
    original = m.verify_packet
    checks, calls = [], []
    def check(path):
        checks.append(path)
        if len(checks) >= 3:  # initial + before first + before second
            raise ValueError('Source result changed')
        return original(path)
    monkeypatch.setattr(m, 'verify_packet', check)
    result = m.execute(output, send=lambda **kw: (calls.append(kw) or good()))
    assert len(calls) == 1
    assert result['status'] == 'stopped_no_retry'
    assert result['results'][-1]['error'] == 'Source result changed'


def test_crash_marker_blocks_all_resume_calls(trial):
    output, _, _ = trial
    enable(output)
    base.save_checkpoint(output/'execution_started.json', {'prior': 'unknown'})
    with pytest.raises(ValueError, match='already started'):
        m.execute(output, send=lambda **_: pytest.fail('duplicate'))


def test_copied_ledger_cannot_replace_original_shared_budget(trial, tmp_path):
    output, budget, _ = trial
    enable(output)
    manifest = base.load_checkpoint(output/'manifest.json')
    copy_budget = tmp_path/'copied_budget'
    copy_budget.mkdir()
    (copy_budget/'spend.json').write_bytes((budget/'spend.json').read_bytes())
    manifest['budget_root'] = str(copy_budget)
    base.save_checkpoint(output/'manifest.json', manifest)
    d = yaml.safe_load((output/'dispatch.yaml').read_text())
    d['manifest_sha256'] = m.sha(output/'manifest.json')
    (output/'dispatch.yaml').write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match='original shared budget'):
        m.execute(output, send=lambda **_: pytest.fail('copied ledger'))


def test_nested_or_empty_training_tags_are_not_recovered():
    for text in ['<reasoning><response>x</response></reasoning>',
                 '<reasoning> </reasoning><response>x</response>',
                 '<response>x</response><reasoning>y</reasoning>']:
        with pytest.raises(ValueError):
            m.parsed_fields(text)
