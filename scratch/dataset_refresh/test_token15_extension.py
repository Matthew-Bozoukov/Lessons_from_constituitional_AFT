# ABOUTME: Verify append-only token targeting and teardown boundaries without paid calls.
# ABOUTME: Run: uv run pytest -q scratch/dataset_refresh/test_token15_extension.py
from collections import Counter
from copy import deepcopy

import pytest

from scratch.dataset_refresh.extend_practical_mixture import (
    balanced_order, closest_prefix, validate_cached_census)
from scratch.nonmoral.result_backup import may_terminate_training


def test_token_target_keeps_replay_constant_and_uses_nearest_prefix():
    counts = [936] * 200
    share, n = closest_prefix(counts, 670033, 4512849, .15)
    assert n == 135
    assert abs(share-.15) < .0001
    assert (670033+sum(counts[:n])) / (5182882+sum(counts[:n])) == share


def test_round_robin_is_balanced_and_preserves_every_payload():
    rows = [dict(metadata=dict(scenario_id=f't{t}_{d}_{i}', trait_id=f't{t}', assigned_domain=d),
                 messages=[dict(content='unchanged')])
            for t in range(1, 10) for d in ['a','b','c'] for i in range(3)]
    ordered = balanced_order(rows, ['a','b','c'], 0)
    assert sorted(r['metadata']['scenario_id'] for r in ordered) == sorted(r['metadata']['scenario_id'] for r in rows)
    assert Counter(r['metadata']['trait_id'] for r in ordered[:27]) == {f't{t}':3 for t in range(1,10)}
    assert all(r in rows for r in ordered)
    assert ordered == balanced_order(list(reversed(rows)), ['a','b','c'], 0)


def test_hub_backup_requires_both_adapter_and_archive_verification():
    state = dict(training_started=True, hub_backup=dict(verified=True))
    assert not may_terminate_training(state)
    state['publication'] = dict(verified=True)
    assert may_terminate_training(state)
    state['hub_backup']['verified'] = False
    assert not may_terminate_training(state)


@pytest.mark.parametrize('changed', ['tokenizer', 'max_seq_len', 'code_hashes', 'parent', 'counts'])
def test_cached_census_rejects_changed_inputs_and_incomplete_rows(changed):
    cfg = dict(parent={'revision': 'a' * 40}, tokenizer={'revision': 'b' * 40}, max_seq_len=8192)
    hashes = {'masking': 'c' * 64}
    census = dict(**deepcopy(cfg), code_hashes=hashes, counts=[12, 34])
    validate_cached_census(census, cfg, hashes, 2)
    census[changed] = [] if changed == 'counts' else 'different'
    with pytest.raises(ValueError, match='Cached token audit'):
        validate_cached_census(census, cfg, hashes, 2)
