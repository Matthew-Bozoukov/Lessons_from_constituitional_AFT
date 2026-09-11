# ABOUTME: Keep failed cleanup responses from blocking saved results, without stealing live work.
# ABOUTME: All provider and process checks are fake and no infrastructure is mutated.
import pytest
from scratch.delegated_harm import scale_live
from src.eval.misalignment.delegated_harm.source import save, digest


@pytest.mark.parametrize('live_process,live_pod,expected', [(False, False, True), (True, False, False), (False, True, False)])
def test_cleanup_reconciliation_requires_stopped_worker_and_absent_pod(tmp_path, monkeypatch, live_process, live_pod, expected):
    path = tmp_path / 'workers/worker/controller.json'
    save(path, {'worker': 'worker', 'pod_id': 'owned-pod', 'pid': 123, 'status': 'evaluating'})
    result = tmp_path / 'result.json'
    save(result, {'status': 'complete'})
    snap = {'workers': [{'id': 'worker', 'mode': 'stopped'}], 'jobs': [
        {'owner': 'worker', 'state': 'done', 'result_path': str(result), 'result_hash': digest({'status': 'complete'})}]}
    monkeypatch.setattr(scale_live.runpod, '_process_identity', lambda _: 'birth' if live_process else '')
    monkeypatch.setattr(scale_live.runpod, 'active_pods', lambda: [{'id': 'owned-pod'}] if live_pod else [])
    monkeypatch.setattr(scale_live.runpod, 'teardown', lambda _: pytest.fail('Reconciliation must be read only'))
    scale_live.reconcile_finished_workers(tmp_path, snap)
    assert scale_live.read(path).get('terminated', False) is expected
