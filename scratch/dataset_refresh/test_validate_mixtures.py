# ABOUTME: Offline exact synthetic release checks for both final mixture arms.
# ABOUTME: Uses no network or model calls; verifies changed responses and duplicate multiplicities fail.
from copy import deepcopy

import pytest

from scratch.dataset_refresh import validate_mixtures as mod
from scratch.dataset_refresh.run import quotas, write_rows


@pytest.fixture
def released(tmp_path):
    rows = []
    for trait, n in quotas().items():
        for i in range(n):
            rows.append({'metadata': {'trait_id': trait, 'scenario_id': f'{trait}_{i}'},
                         'messages': [{'role': 'system', 'content': 'Advice.', 'reasoning_content': None},
                                      {'role': 'user', 'content': f'{trait} choice {i}.'},
                                      {'role': 'assistant', 'content': 'A supported choice.', 'reasoning_content': 'Weigh the facts.'}]})
    path = tmp_path / 'dataset.jsonl'
    write_rows(path, rows)
    source = {'path': path, 'style': 'nonmoral-advice', 'repo': 'test/corpus', 'revision': 'a' * 40}
    mixture = [{'source': source['style'], 'n_tokens': 123, 'supervise': 'all',
                'messages': mod.clean_messages(r['messages'])} for r in reversed(rows)]
    return source, mixture


def test_full716_released_payloads_preserved_after_balancing_shuffle(released):
    source, mixture = released
    evidence = mod.check_synthetic_source(mixture, source)
    assert evidence['payloads_equal'] and evidence['rows'] == 716
    assert evidence['sha256'] == mod.digest(source['path'].read_bytes())


@pytest.mark.parametrize('change', ['response', 'reasoning', 'duplicate', 'supervise', 'source'])
def test_changed_synthetic_payload_or_multiplicity_refused(released, change):
    source, mixture = released
    mixture = deepcopy(mixture)
    if change == 'response':
        mixture[0]['messages'][-1]['content'] += ' Changed fact.'
    elif change == 'reasoning':
        mixture[0]['messages'][-1]['reasoning_content'] += ' Unreviewed thought.'
    elif change == 'duplicate':
        mixture[0] = mixture[1]
    elif change == 'supervise':
        mixture[0]['supervise'] = 'final'
    else:
        mixture[0]['source'] = 'wrong-arm'
    with pytest.raises(ValueError, match='payloads/multiplicities'):
        mod.check_synthetic_source(mixture, source)


def test_actual_cached_qwen_token_count_matches_builder():
    tokenizer = mod.AutoTokenizer.from_pretrained(mod.TOKENIZER, local_files_only=True)
    profile = mod.model_profile(mod.TOKENIZER)
    row = {'messages': [{'role': 'system', 'content': 'Offer advice.'}, {'role': 'user', 'content': 'Choose a format.'},
                        {'role': 'assistant', 'content': 'Use prose here.', 'reasoning_content': 'The relationships need explanation.'}],
           'source': 'nonmoral-advice', 'supervise': 'all'}
    row['n_tokens'] = len(mod.render_chat(tokenizer, row['messages'], None, render_kwargs=profile.render_kwargs,
                                         tokenize=True, return_dict=True)['input_ids'])
    assert mod.token_audit(row, tokenizer, profile, 8192)['training_tokens'] == row['n_tokens']
    row['n_tokens'] -= 1
    with pytest.raises(ValueError, match='Stored token count'):
        mod.token_audit(row, tokenizer, profile, 8192)
