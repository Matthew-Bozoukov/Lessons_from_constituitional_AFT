# ABOUTME: Verify exclusive claims and live pause/drain control across concurrent workers.
# ABOUTME: Tests use a temporary SQLite database and never start model servers.
from concurrent.futures import ThreadPoolExecutor

import pytest
from scratch.delegated_harm.episode_queue import EpisodeQueue


def test_claims_are_exclusive_under_concurrent_workers(tmp_path):
    q = EpisodeQueue(tmp_path / 'queue.sqlite')
    q.enqueue('control', [{'id': str(i)} for i in range(80)])
    for i in range(8):
        q.register(str(i), 'control', {})
    def consume(worker):
        found = []
        while True:
            state, cell = q.claim(worker)
            if state == 'empty':
                return found
            found.append(cell['id'])
            q.finish(worker, 'control', cell['id'], tmp_path / cell['id'], 'hash')
    with ThreadPoolExecutor(8) as pool:
        rows = [x for group in pool.map(consume, map(str, range(8))) for x in group]
    assert len(rows) == len(set(rows)) == 80
    assert all(j['state'] == 'done' for j in q.snapshot()['jobs'])


def test_pause_drain_and_wrong_owner_preserve_active_claim(tmp_path):
    q = EpisodeQueue(tmp_path / 'queue.sqlite')
    q.enqueue('da', [{'id': 'one'}, {'id': 'two'}])
    q.register('first', 'da', {})
    q.register('second', 'da', {})
    assert q.claim('first')[0] == 'claimed'
    q.pause(True)
    assert q.claim('second') == ('paused', None)
    q.mode('first', 'drain')
    q.pause(False)
    assert q.claim('first') == ('drain', None)
    with pytest.raises(AssertionError):
        q.finish('second', 'da', 'one', tmp_path / 'one', 'hash')
    q.finish('first', 'da', 'one', tmp_path / 'one', 'hash')
    assert q.claim('second')[1]['id'] == 'two'


def test_legacy_claims_and_different_adapters_cannot_be_taken(tmp_path):
    q = EpisodeQueue(tmp_path / 'queue.sqlite')
    q.enqueue('control', [{'id': 'same'}], {'same': 'legacy'})
    q.enqueue('da', [{'id': 'same'}])
    q.register('c', 'control', {})
    q.register('d', 'da', {})
    assert q.claim('c') == ('empty', None)
    assert q.claim('d')[0] == 'claimed'
