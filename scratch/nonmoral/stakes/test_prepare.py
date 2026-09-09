# ABOUTME: Offline checks for stakes identity, source integrity, complete pairs and numerical constraints.
# ABOUTME: Run: uv run pytest -q scratch/nonmoral/stakes/test_prepare.py; no network or paid APIs.
import copy
import json

import pytest
from omegaconf import OmegaConf

from scratch.nonmoral.stakes import checks
from scratch.nonmoral.stakes.fixtures import FIXTURES
from scratch.nonmoral.stakes.prepare import arm_rows, join_answers, approved, digest, phase_config


def answer_rows():
    rows = arm_rows([dict(FIXTURES[0], eligible='yes', original_system='Original system')])
    return [dict(r, reasoning='Compare complete options.', response='A complete artifact.') for r in rows]


def test_pause_file_prevents_paid_dispatch(tmp_path,monkeypatch):
    from scratch.nonmoral.stakes import production
    monkeypatch.setattr(production,'OUT',tmp_path)
    calls=[]
    class Client:
        def chat(self,**kwargs):
            calls.append(kwargs)
            return 'completed'
    client=production.PauseAwareClient(Client())
    assert client.chat(model='offline-test')=='completed'
    (tmp_path/'STOP_DISPATCH').write_text('pause')
    with pytest.raises(RuntimeError,match='no new charge'):
        client.chat(model='offline-test')
    assert len(calls)==1


def test_exact_task_preserved_and_answers_not_forced_identical():
    rows = answer_rows()
    rows[1]['response'] = 'A different decision can be a stakes effect.'
    pair = join_answers(rows)[0]
    assert pair['low_user'].startswith(FIXTURES[0]['core_user']+'\n\n')
    assert pair['high_user'].startswith(FIXTURES[0]['core_user']+'\n\n')
    assert pair['low_response'] != pair['high_response']


@pytest.mark.parametrize('mutate', [
    lambda r: r.pop(),
    lambda r: r.append(copy.deepcopy(r[0])),
    lambda r: r[0].update(user='Silently repaired request'),
    lambda r: r[1].update(original_system='Changed system'),
    lambda r: r[0].update(response=''),
    lambda r: r[1].update(arm='medium'),
])
def test_broken_pair_rejected(mutate):
    rows = answer_rows()
    mutate(rows)
    with pytest.raises(ValueError):
        join_answers(rows)


def test_hash_linked_review_cannot_approve_changed_source(tmp_path):
    source = tmp_path/'source.jsonl'
    source.write_text('{"scenario_id":"a"}\n')
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(source_sha256=digest(source), reviewer='Local review',
        dispositions={'a':dict(decision='accept', reason='Complete original source')})))
    assert len(approved([{'scenario_id':'a'}], source, review)) == 1
    source.write_text('{"scenario_id":"a","user":"changed"}\n')
    with pytest.raises(ValueError):
        approved([{'scenario_id':'a'}], source, review)


def test_material_checks_catch_wrong_dimensions_counts_bounds_and_duration():
    spec = FIXTURES[0]['checks']['symbol_grid']
    assert checks.grid(['BCBC', 'CBCB', 'BCBC', 'CBCB'], spec)
    assert not checks.grid(['BCBC', 'CBCB', 'BCBC'], spec)
    p = FIXTURES[2]['checks']['points']
    assert checks.points([[0, 0], [0, 10], [10, 0], [10, 10], [4, 6], [2, 7]], p)
    assert not checks.points([[0, 0]]*6, p)
    assert not checks.points([[0, 0], [0, 10], [10, 0], [10, 11], [4, 6], [2, 7]], p)
    assert checks.exact_items(list('ABCDEF'), list('ABCDEF'))
    assert not checks.exact_items(list('ABCDEE'), list('ABCDEF'))
    assert checks.bars([[4], [1, 1, 1, 1], [4], [4]])
    assert not checks.bars([[4], [1, 1, 1], [4], [4]])


@pytest.mark.parametrize('phase', ['frames', 'answers', 'review'])
def test_prepared_configs_use_shared_stages_and_zero_spend(tmp_path, phase):
    cfg = OmegaConf.to_container(OmegaConf.load('configs/data/synth/nonmoral-stakes.yaml'), resolve=True)
    phase_config(cfg, phase, [dict(scenario_id='example')], tmp_path)
    result = OmegaConf.load(tmp_path/'prepared_config.yaml')
    assert result.budget_usd == 0
    assert result.stages[0].kind == 'load_source_run'
    assert result.stages[1].kind == 'llm_tagged'
