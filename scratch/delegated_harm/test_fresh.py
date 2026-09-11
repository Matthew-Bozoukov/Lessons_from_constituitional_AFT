# ABOUTME: Offline checks for fresh queue execution and ownership-verified merging.
# ABOUTME: Run: uv run pytest -q scratch/delegated_harm/test_fresh.py.
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scratch.delegated_harm import fresh, scale_worker
from scratch.delegated_harm.episode_queue import EpisodeQueue
from src.eval.misalignment.delegated_harm.source import digest, save


@dataclass
class Spec:
    revision: str = 'adapter-pin'
    base_revision: str = 'base-pin'


def test_fresh_author_failure_does_not_block_human(tmp_path, monkeypatch):
    cells = [dict(id=f'example__{delivery}__v0__r0', scenario='example', delivery=delivery, variant=0, repeat=0)
             for delivery in ('chat', 'subagent', 'chat_human')]
    queue = EpisodeQueue(tmp_path / 'queue.sqlite')
    queue.enqueue('nonmoral', cells)
    inputs = tmp_path / 'inputs.json'
    save(inputs, {'target': {'revision': 'adapter-pin', 'base_revision': 'base-pin'},
                  'authors': {'example__v0': {'accepted': False}}})
    cfg = OmegaConf.create({'scaling': {'queue': queue.path, 'worker': 'worker', 'arm': 'nonmoral', 'inputs': str(inputs)},
        'generation': {'request_timeout': 1200, 'parallel': 2, 'max_tokens': 16384, 'max_turns': 90},
        'serving': {'context_window': 131072}, 'subject_model_label': 'Qwen3.6-27B'})
    target = SimpleNamespace(spec=Spec(), base_url='http://invalid', api_key='unused', model_name='test')
    monkeypatch.setattr(scale_worker, 'prepare', lambda c: ({'example': object()}, {'example': 'human wording'}, None, None))
    monkeypatch.setattr(scale_worker, 'OpenAI', lambda **kw: object())
    monkeypatch.setattr(scale_worker, 'TokenBudget', lambda *a: object())
    def episode(client, model, label, scenario, cell, brief, cfg, path, **kw):
        assert cell['delivery'] == 'chat_human' and brief == 'human wording'
        return {'status': 'complete', 'mechanical': {'compliance': 0}}
    monkeypatch.setattr('src.eval.misalignment.delegated_harm.runner.episode', episode)
    out = tmp_path / 'worker-run'
    scale_worker.run_worker(target, cfg, out)
    rows = [json.loads(p.read_text()) for p in (out / 'results/episodes').glob('*.json')]
    assert sum(r['status'] == 'author_failure' for r in rows) == 2
    human = next(r for r in rows if r['delivery'] == 'chat_human')
    assert human['status'] == 'complete' and 'execution' in human and 'recovery' not in human
    assert all(j['state'] == 'done' for j in queue.snapshot()['jobs'])


def make_merge_fixture(tmp_path):
    queue = EpisodeQueue(tmp_path / 'queue.sqlite')
    cell = {'id': 'example', 'status': 'complete'}
    worker = tmp_path / 'workers/worker/runs/test'
    queue.enqueue('nonmoral', [cell])
    queue.register('worker', 'nonmoral', {'out_dir': str(worker)})
    queue.claim('worker')
    result = worker / 'results/episodes/example.json'
    save(result, cell)
    save(worker / 'rollouts/example.json', cell)
    queue.finish('worker', 'nonmoral', 'example', result, digest(cell))
    save(tmp_path / 'workers/worker/controller.json', {'terminated': True})
    save(tmp_path / 'metadata/handoff.json', {'arm': 'nonmoral', 'target': {'model_key': 'qwen36_nonmoral'}})
    save(tmp_path / 'metadata/schedule.json', [cell])
    save(tmp_path / 'metadata/protocol.json', {'protocol': {}})
    return result


def test_merge_checks_hashes_and_carries_traces(tmp_path):
    make_merge_fixture(tmp_path)
    out = fresh.merge(tmp_path)
    assert (out / 'rollouts/example.json').exists()
    assert (out / 'metadata/run_meta.json').exists()
    metadata = json.loads((out / 'metadata/horizontal_scaling.json').read_text())
    assert metadata['queue']['jobs'][0]['owner'] == 'worker'


def test_merge_rejects_changed_record(tmp_path):
    result = make_merge_fixture(tmp_path)
    save(result, {'id': 'example', 'status': 'changed'})
    with pytest.raises(AssertionError, match='Result changed'):
        fresh.merge(tmp_path)


def test_merge_requires_cleanup(tmp_path):
    make_merge_fixture(tmp_path)
    save(tmp_path / 'workers/worker/controller.json', {'terminated': False})
    with pytest.raises(AssertionError, match='cleanup'):
        fresh.merge(tmp_path)
