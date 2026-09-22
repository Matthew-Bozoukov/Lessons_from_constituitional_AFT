# ABOUTME: Offline adversarial tests for explicit final authorship from failed untrained short drafts.
# ABOUTME: Require source review, original evidence, unchanged final gates and one physical call per stage.
import json
from pathlib import Path

import pytest
from scratch.dataset_refresh import recover_short_draft as mod
from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.test_per_row import case, offline, tagged, clean_verdict, client_for
from scratch.dataset_refresh.test_recover_missing_changes import ready, RecordedClient

LONG_REASON = ('The monthly process protects the group decision; convenience alone does not supply permission. ' * 9).strip()
LONG_FINAL = ('Ask the group to authorize an exception or wait for the monthly meeting; do not call an informal vote existing permission. ' * 8).strip()


@pytest.fixture
def short(ready):
    c = ready
    for stage in c.cfg['response_stages']:
        stage['lint'] = {'fields': ['reasoning', 'response'], 'min_chars': 700, 'ban_patterns': [r'^\s*okay,']}
    base.write_json(c.root / c.arm / 'config.json', c.cfg)
    meta = base.load_checkpoint(c.root / 'run_meta.json')
    meta['arms'][c.arm]['config_sha256'] = base.digest(c.cfg)
    base.save_checkpoint(c.root / 'run_meta.json', meta)
    replies = client_for().replies
    replies['draft_responses'] = tagged('Brief but complete reasoning.', 'Wait or ask permission.')
    original_client = RecordedClient(replies, c.budget)
    from scratch.dataset_refresh.per_row import generate_one
    result = generate_one(c.root, c.arm, c.candidate, original_client)
    assert result['status'] == 'failed' and 'under the 700 minimum' in result['error']
    c.original = (c.out / 'result.json').read_bytes()
    c.draft = (c.out / 'draft_responses.json').read_bytes()
    base.save_checkpoint(c.out / mod.SOURCE_REVIEW, {'result_sha256': base.digest(c.original), 'eligible': True,
                                                  'reason': 'Read actual source: a benign club scheduling choice with no missing facts.'})
    return c


def fresh(c, monkeypatch, author=None, grounding=None):
    client = RecordedClient({mod.AUTHOR: author or tagged(LONG_REASON, LONG_FINAL, True),
        'grounding_200': grounding or clean_verdict(),
        'review_200': {'accepted': True, 'factual_reasoning': True, 'stakes': 1}}, c.budget)
    monkeypatch.setattr(base, 'BudgetClient', lambda *a, **kw: client)
    return client


def test_exact_three_calls_blind_critics_preserve_failed_input(short, monkeypatch):
    c = short; client = fresh(c, monkeypatch)
    r = mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert r['status'] == 'accepted' and r['accepted_attempt'] == 200
    assert [s for s, _ in client.calls] == [mod.AUTHOR, 'grounding_200', 'review_200']
    assert all(req['model'] == 'anthropic/claude-sonnet-5' for _, req in client.calls)
    assert 'SECRET_TRAIT' not in json.dumps(client.calls[1][1]['messages'])
    assert 'untrained input' not in json.dumps(client.calls[1][1]['messages'])
    assert (c.out / 'result.json').read_bytes() == c.original
    assert (c.out / 'draft_responses.json').read_bytes() == c.draft
    assert not (c.out / 'revise_responses.json').exists()
    assert base.load_checkpoint(c.out / 'result.json')['status'] == 'failed'
    mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert len(client.calls) == 3


def test_adoption_requires_exact_proposal_and_preserves_failed_original(short, monkeypatch):
    c = short; fresh(c, monkeypatch)
    mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    sha = base.digest((c.out / mod.PROPOSAL).read_bytes())
    with pytest.raises(ValueError, match='Approval'):
        mod.adopt(c.root, c.arm, c.candidate['candidate_id'], '0' * 64, 'Reviewed full answer.')
    mod.adopt(c.root, c.arm, c.candidate['candidate_id'], sha, 'Full new answer preserves actual permission and alternatives.')
    assert base.load_result(c.out / 'result.json')['status'] == 'accepted'
    assert (c.out / 'recovered_failures/short_draft_recovery_200/result.json').read_bytes() == c.original
    assert (c.out / 'draft_responses.json').read_bytes() == c.draft


@pytest.mark.parametrize('mutation', ['source_hold', 'source_hash', 'no_source', 'exclusion', 'accepted_original', 'other_lint', 'lowered_final_min', 'truncated', 'unknown', 'duplicate', 'raw_change'])
def test_bad_inputs_never_dispatch(short, monkeypatch, mutation):
    c = short; client = fresh(c, monkeypatch)
    if mutation in ('source_hold', 'source_hash'):
        r = base.load_checkpoint(c.out / mod.SOURCE_REVIEW)
        r['eligible' if mutation == 'source_hold' else 'result_sha256'] = False if mutation == 'source_hold' else '0' * 64
        base.save_checkpoint(c.out / mod.SOURCE_REVIEW, r)
    elif mutation == 'no_source':
        (c.out / mod.SOURCE_REVIEW).unlink()
    elif mutation == 'exclusion':
        base.save_checkpoint(c.out / 'independent_exclusion.json', {'result_sha256': base.digest(c.original), 'reason': 'Source hold.'})
    elif mutation in ('accepted_original', 'other_lint'):
        r = base.load_checkpoint(c.out / 'result.json')
        r['status' if mutation == 'accepted_original' else 'error'] = 'accepted' if mutation == 'accepted_original' else 'Response lint: stock phrase'
        base.save_checkpoint(c.out / 'result.json', r)
    elif mutation == 'lowered_final_min':
        c.cfg['response_stages'][-1]['lint']['min_chars'] = 100
        base.write_json(c.root / c.arm / 'config.json', c.cfg)
    else:
        entries = json.loads((c.budget / 'spend.json').read_text()); e = entries[-1]
        path = c.budget / 'raw_calls' / f"{e['call_id']:06d}.json"; raw = json.loads(path.read_text())
        if mutation == 'truncated': raw['response']['finish_reason'] = 'length'
        if mutation == 'unknown': e['status'] = 'uncertain_failure'; raw['accounting'] = e
        if mutation == 'duplicate': entries.append({**e, 'call_id': len(entries)})
        if mutation == 'raw_change': raw['request']['messages'][0]['content'] += ' Changed'
        base.write_json(path, raw); base.write_json(c.budget / 'spend.json', entries)
    with pytest.raises((ValueError, base.BudgetStop, FileNotFoundError)):
        mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert not client.calls


@pytest.mark.parametrize('kind', ['short_final', 'stock_opening', 'invalid_critic'])
def test_final_lint_and_invalid_reviews_stay_failed_without_retry(short, monkeypatch, kind):
    c = short
    author = tagged(LONG_REASON, 'Still too short.', True) if kind == 'short_final' else tagged(LONG_REASON, 'Okay, ' + LONG_FINAL, True) if kind == 'stock_opening' else None
    client = fresh(c, monkeypatch, author=author, grounding={'accepted': True, 'issues': []} if kind == 'invalid_critic' else None)
    r = mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert r['status'] == 'failed'
    count = len(client.calls)
    assert count == (2 if kind == 'invalid_critic' else 1)
    mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert len(client.calls) == count
    assert (c.out / 'result.json').read_bytes() == c.original


def test_started_unknown_and_raw_tampering_block_dispatch_or_adoption(short, monkeypatch):
    c = short; client = fresh(c, monkeypatch)
    # First create the manifest without dispatch by simulating a budget refusal.
    client.replies[mod.AUTHOR] = base.BudgetStop('No budget available')
    with pytest.raises(base.BudgetStop): mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    with pytest.raises(base.BudgetStop, match='already started'): mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert len(client.calls) == 1


def test_valid_adverse_verdict_is_not_retried_or_adoptable(short, monkeypatch):
    c = short
    g = {'accepted': False, 'issues': [{'kind': 'unsupported_decisive_fact', 'answer_field': 'final',
        'answer_quote': LONG_FINAL, 'premise_field': 'absent', 'premise_quote': '',
        'material_consequence': 'Test rejects the answer.', 'alternative_reading': 'No benign reading in fixture.'}], 'assessment': 'Adverse fixture.'}
    client = fresh(c, monkeypatch, grounding=g)
    r = mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    assert r['status'] == 'rejected' and len(client.calls) == 3
    with pytest.raises(ValueError, match='Approval'):
        mod.adopt(c.root, c.arm, c.candidate['candidate_id'], base.digest((c.out / mod.PROPOSAL).read_bytes()), 'Cannot adopt rejected answer.')


def test_physical_author_tampering_prevents_adoption(short, monkeypatch):
    c = short; fresh(c, monkeypatch)
    mod.propose(c.root, c.arm, c.candidate['candidate_id'], 245)
    entry = next(e for e in json.loads((c.budget / 'spend.json').read_text()) if e['stage'] == mod.AUTHOR)
    path = c.budget / 'raw_calls' / f"{entry['call_id']:06d}.json"; raw = json.loads(path.read_text())
    raw['response']['content'] = tagged(LONG_REASON, LONG_FINAL + ' changed', True)
    base.write_json(path, raw)
    with pytest.raises(ValueError, match='physical output'):
        mod.adopt(c.root, c.arm, c.candidate['candidate_id'], base.digest((c.out / mod.PROPOSAL).read_bytes()), 'Reviewed full answer.')
    assert (c.out / 'result.json').read_bytes() == c.original


def test_real_budget_client_refuses_dispatch_above_shared_ceiling(short):
    c = short
    entries = json.loads((c.budget / 'spend.json').read_text())
    for e in entries:
        e['charged_or_reserved_usd'] = 249.999 if e is entries[-1] else 0
        path = c.budget / 'raw_calls' / f"{e['call_id']:06d}.json"
        raw = json.loads(path.read_text()); raw['accounting'] = e; base.write_json(path, raw)
    base.write_json(c.budget / 'spend.json', entries)
    with pytest.raises(base.BudgetStop, match='Budget stop'):
        mod.propose(c.root, c.arm, c.candidate['candidate_id'], 250,
                    send=lambda **kw: pytest.fail('Budget must stop before physical dispatch'))
    assert len(json.loads((c.budget / 'spend.json').read_text())) == len(entries)
    assert (c.out / 'result.json').read_bytes() == c.original
