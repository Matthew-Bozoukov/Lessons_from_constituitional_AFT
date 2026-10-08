# ABOUTME: The concurrency migration preserves exact paid outcomes and rejects unsafe snapshots.
# ABOUTME: Run with pytest scratch/gptoss_swe/test_resume.py; no network or inference required.
from copy import deepcopy
import pytest
from scratch.gptoss_swe.resume import validate_snapshot


def fixture():
    cfg = {key: key for key in ('protocol','dataset_revision','dataset_sha256','sampling','worker',
        'tinker','qualification_instance','cached_campaign','campaign')}
    cfg['workers'] = 80
    tasks = {str(i):dict(status='pending',attempts=[]) for i in range(300)}
    tasks['0'] = dict(status='valid',attempts=[dict(valid=True,prediction={'instance_id':'0','model_patch':'saved patch'})])
    state = dict(tasks=tasks,halt=None,deadline=1234)
    manifest = dict(target='tinker://base', config=cfg | {'workers':16}, selected_ids=list(tasks))
    return state, manifest, cfg


def test_preserves_outcomes_and_original_snapshot():
    state, manifest, cfg = fixture()
    original = deepcopy(state)
    result = validate_snapshot(state,manifest,cfg,'tinker://base')
    assert result['tasks'] == original['tasks']
    assert result['deadline'] is None and state == original


@pytest.mark.parametrize('status', ['running','invalid'])
def test_rejects_undrained_or_failed_task(status):
    state,manifest,cfg = fixture(); state['tasks']['1']['status']=status
    with pytest.raises(AssertionError): validate_snapshot(state,manifest,cfg,'tinker://base')


@pytest.mark.parametrize('key',['sampling','worker','tinker','protocol','dataset_sha256'])
def test_rejects_recipe_or_budget_drift(key):
    state,manifest,cfg = fixture(); cfg[key]='changed'
    with pytest.raises(AssertionError): validate_snapshot(state,manifest,cfg,'tinker://base')


def test_rejects_target_change_and_pending_previous_attempt():
    state,manifest,cfg=fixture()
    with pytest.raises(AssertionError): validate_snapshot(state,manifest,cfg,'another-target')
    state['tasks']['1']['attempts']=[{'valid':False}]
    with pytest.raises(AssertionError): validate_snapshot(state,manifest,cfg,'tinker://base')
