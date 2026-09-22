# ABOUTME: Checks hard launch bounds and deterministic balanced selection under missing cells.
# ABOUTME: Run: uv run --no-sync python -m pytest -q scratch/dataset_refresh/test_native_full.py
import copy
import pytest
from omegaconf import OmegaConf
from scratch.dataset_refresh.run_native_smoke import validate_launch
from scratch.dataset_refresh.select_native_lowstakes import select_rows


def config():
    launch = OmegaConf.to_container(OmegaConf.load('scratch/dataset_refresh/native_lowstakes_full.yaml'), resolve=True)
    recipe = OmegaConf.to_container(OmegaConf.load(launch['recipe']), resolve=True)
    return launch, recipe


def fixture(spec):
    return [dict(messages=[dict(role='user', content=f'{t}/{d}/{i}')],
                 metadata=dict(trait_id=t, assigned_domain=d, scenario_id=f'{t}_{d}_{i}'))
            for t in spec['trait_quotas'] for d in spec['domains'] for i in range(12)]


def test_selection_is_order_independent_and_exact_subset():
    launch, _ = config()
    spec = launch['selection']
    rows = fixture(spec)
    chosen, report = select_rows(rows, spec)
    reverse, _ = select_rows(list(reversed(rows)), spec)
    assert chosen == reverse and len(chosen) == 716
    assert all(r in rows for r in chosen)
    assert report['selected_by_trait'] == spec['trait_quotas']
    assert set(report['selected_by_cell'].values()) == {8, 9}


def test_shortfall_and_empty_cell_never_silently_relax():
    launch, _ = config()
    spec = launch['selection']
    rows = fixture(spec)
    chosen, report = select_rows([r for r in rows if r['metadata']['trait_id'] != 't1'], spec)
    assert not chosen and report['shortfalls'] == {'t1': 80}
    chosen, report = select_rows([r for r in rows if not (r['metadata']['trait_id'] == 't1' and r['metadata']['assigned_domain'] == spec['domains'][0])], spec)
    assert not chosen and not report['shortfalls'] and len(report['empty_cells']) == 1


def test_duplicate_ids_rejected():
    launch, _ = config()
    rows = fixture(launch['selection'])
    with pytest.raises(ValueError, match='Duplicate'):
        select_rows(rows + rows[:1], launch['selection'])


def test_launch_ceiling_and_candidate_count():
    launch, recipe = config()
    assert validate_launch(launch, recipe) == 'full'
    for field, value in [('ceiling_usd', 120.01), ('expected_candidates', 973), ('mode', 'unknown')]:
        bad = copy.deepcopy(launch)
        bad[field] = value
        with pytest.raises((AssertionError, ValueError)):
            validate_launch(bad, recipe)
    recipe['scenarios_per_call'] = 13
    with pytest.raises(AssertionError):
        validate_launch(launch, recipe)
