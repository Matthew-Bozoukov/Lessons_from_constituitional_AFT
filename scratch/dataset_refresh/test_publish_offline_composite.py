# ABOUTME: Validate exact650-plus66 release composition, provenance and final closed quality/budget gates offline.
# ABOUTME: Exercise real original-row validators and mocked already-tested offline adoption without API or publication.
from copy import deepcopy
from pathlib import Path

import pytest

from scratch.dataset_refresh import publish_offline_composite as m
from scratch.dataset_refresh.test_publish_composite import corpus, preserve


@pytest.fixture
def mixed(corpus, tmp_path, monkeypatch):
    all_rows, phases, _ = m.composite.validate_selection(corpus)
    base = {'arm': m.ARM, 'entries': corpus['entries'][:650]}
    base_path = tmp_path/'base.json'; m.base.write_json(base_path, base)
    monkeypatch.setattr(m, 'BASE_SELECTION_SHA', m.offline.sha(base_path))
    offline_entries, answers, dossiers = [], {}, {}
    cfg_path = Path(corpus['entries'][0]['root'])/m.ARM/'config.json'
    for index, (entry, row) in enumerate(zip(corpus['entries'][650:], all_rows[650:])):
        path = tmp_path/f'offline{index}'/'acceptance.json'
        digest = f'{index:064x}'
        d = {'source_ref': entry, 'physical_receipt': {'call_id': index}, 'author_kind': 'single_saved_revision'}
        answers[str(path)] = (deepcopy(row), {'dossier': d, 'acceptance': {'automatic_acceptance': False}})
        dossiers[str(path.parent/'dossier.json')] = (d, {'review_config': cfg_path})
        offline_entries.append({'acceptance_path': str(path), 'acceptance_sha256': digest})
    def accepted(path, expected):
        selected = next(e for e in offline_entries if e['acceptance_path'] == path)
        if selected['acceptance_sha256'] != expected:
            raise ValueError('Offline acceptance hash changed')
        return answers[path]
    monkeypatch.setattr(m.offline, 'validate_accepted', accepted)
    monkeypatch.setattr(m.offline, 'validate_dossier', lambda path: dossiers[str(path)])
    selection = {'arm': m.ARM, 'base_selection': {'path': str(base_path), 'sha256': m.BASE_SELECTION_SHA},
                 'offline_entries': deepcopy(offline_entries)}
    path = tmp_path/'selection.json'; m.base.write_json(path, selection)
    return {'path': path, 'selection': selection, 'answers': answers, 'base': base, 'rows': all_rows,
            'phases': phases, 'dossiers': dossiers, 'base_path': base_path}


def test_exact_mixed_release_validates_original_rows_and_preserves_metadata(mixed):
    rows, phases, dossiers = m.validate_selection(mixed['path'])
    assert len(rows) == 716 and len(dossiers) == 66 and len(phases) == 2
    assert m.offline.validate_release(rows)['quotas'] == m.base.quotas()
    assert rows[0]['metadata']['custom_metadata'] == {'retain': 'complete metadata'}
    assert all('rewrite_changes' not in r['metadata'] for r in rows)


@pytest.mark.parametrize('change', ['wrong_base_hash', 'missing_offline', 'duplicate_offline', 'changed_offline_hash', 'same_source_twice'])
def test_composition_rejects_missing_or_replaced_binding(mixed, change):
    selection = mixed['selection']
    if change == 'wrong_base_hash': selection['base_selection']['sha256'] = '0'*64
    elif change == 'missing_offline': selection['offline_entries'].pop()
    elif change == 'duplicate_offline': selection['offline_entries'][1] = selection['offline_entries'][0]
    elif change == 'changed_offline_hash': selection['offline_entries'][0]['acceptance_sha256'] = 'f'*64
    else:
        path = selection['offline_entries'][0]['acceptance_path']
        mixed['answers'][path][1]['dossier']['source_ref'] = mixed['base']['entries'][0]
    m.base.write_json(mixed['path'], selection)
    with pytest.raises(ValueError): m.validate_selection(mixed['path'])


def test_original_missing_grounding_review_cannot_be_relabeled_offline(mixed):
    entry = mixed['base']['entries'][0]
    path = Path(entry['root'])/m.ARM/'records'/entry['candidate_id']/'grounding_0.json'
    with preserve(path):
        path.unlink()
        with pytest.raises((FileNotFoundError, m.base.BudgetStop)):
            m.validate_selection(mixed['path'])


def test_original_hold_still_excludes_row(mixed):
    entry = mixed['base']['entries'][0]
    path = Path(entry['root'])/m.ARM/'records'/entry['candidate_id']/'independent_exclusion.json'
    with preserve(path, path.with_suffix('.receipt.json')):
        m.base.save_checkpoint(path, {'result_sha256': entry['result_sha256'], 'reason': 'Material defect'})
        with pytest.raises(ValueError, match='original acceptance'):
            m.validate_selection(mixed['path'])


def test_cross_route_duplicate_prompt_fails(mixed):
    first = mixed['selection']['offline_entries'][0]['acceptance_path']
    mixed['answers'][first][0]['messages'][1]['content'] = mixed['rows'][0]['messages'][1]['content']
    with pytest.raises(ValueError, match='Duplicate source prompt'):
        m.validate_selection(mixed['path'])


def test_offline_review_contract_mismatch_fails(mixed, tmp_path):
    cfg = m.offline.read(next(iter(mixed['dossiers'].values()))[1]['review_config'])
    cfg['constitution_sha256'] = 'wrong'
    wrong = tmp_path/'wrong.json';m.base.write_json(wrong, cfg)
    next(iter(mixed['dossiers'].values()))[1]['review_config'] = wrong
    with pytest.raises(ValueError, match='different constitution'):
        m.validate_selection(mixed['path'])


def test_closed_budget270_preserves_failed_billing_status():
    ledger = [{'call_id': 0, 'status': 'settled', 'charged_or_reserved_usd': 250., 'api_reported_cost_usd': 250.},
              {'call_id': 1, 'status': 'billing_verified_failure', 'charged_or_reserved_usd': 19.9, 'api_reported_cost_usd': 19.8}]
    report = m.accounting(ledger, 2)
    assert report['charged_usd'] == 269.9 and report['active_or_uncertain_calls'] == 0
    assert report['statuses']['billing_verified_failure'] == 1
    assert ledger[1]['status'] == 'billing_verified_failure'
    with pytest.raises(ValueError, match='cutoff'): m.accounting(ledger, 1)


@pytest.mark.parametrize('field,value', [('status', 'reserved'), ('status', 'uncertain_failure'),
                                      ('charged_or_reserved_usd', 270.001), ('charged_or_reserved_usd', float('nan')),
                                      ('api_reported_cost_usd', -1), ('call_id', 3)])
def test_invalid_or_unclosed_budget_fails(field, value):
    entry = {'call_id': 0, 'status': 'settled', 'charged_or_reserved_usd': .1}
    entry[field] = value
    with pytest.raises(ValueError): m.accounting([entry], 1)


@pytest.fixture
def quality(tmp_path):
    rows = []
    for trait, count in m.base.quotas().items():
        for index in range(count):
            sid = f'{trait}_{index}'
            rows.append({'metadata': {'scenario_id': sid, 'trait_id': trait}})
    diagnostics = [{'scenario_id': r['metadata']['scenario_id'], 'training_tokens': 2500, 'supervised_tokens': 1800,
                    'raw_reasoning_tokens': 1000, 'raw_assistant_content_tokens': 790} for r in rows]
    token = {'dataset_sha256': 'dataset', 'rows': 716, 'status': 'passed', 'failures': [], 'untruncated': True,
             'tokenizer': 'Qwen/Qwen3.6-27B', 'max_train_tokens': 8192, 'diagnostics': diagnostics}
    m.base.write_json(tmp_path/'automatic/token_mask_audit.json', token)
    m.base.write_json(tmp_path/'automatic/corpus_audit.json', {'input_sha256': 'dataset', 'rows': 716})
    for name in ('full_census.jsonl', 'semantic_pairs.jsonl', 'lexical_pairs.jsonl', 'literal_screen.json'):
        (tmp_path/'automatic'/name).write_text('{}', encoding='utf-8')
    m.base.write_json(tmp_path/'independent/adjudication.json', {'dataset_sha256': 'dataset',
        'release_approved': True, 'unresolved_material_issues': [], 'scope': 'Read all surfaced pairs and literal flags.'})
    return tmp_path, rows, token


def test_final_quality_requires_exact_dataset_and_native_rows(quality):
    path, rows, token = quality
    m.validate_quality(path, rows, 'dataset')
    with pytest.raises(ValueError): m.validate_quality(path, rows, 'changed')
    token['diagnostics'][0]['training_tokens'] = 9000
    m.base.write_json(path/'automatic/token_mask_audit.json', token)
    with pytest.raises(ValueError, match='lengths/masks'): m.validate_quality(path, rows, 'dataset')


def test_final_quality_needs_actual_independent_release_approval(quality):
    path, rows, _ = quality
    value = m.offline.read(path/'independent/adjudication.json'); value['unresolved_material_issues'] = ['real defect']
    m.base.write_json(path/'independent/adjudication.json', value)
    with pytest.raises(ValueError, match='adjudication'): m.validate_quality(path, rows, 'dataset')


def test_route_provenance_compatible_with_existing_mixture_validator(tmp_path, monkeypatch):
    cfg = {'pipeline': m.ARM, 'constitution': m.prepare_mixtures.CONSTITUTION,
           'constitution_sha256': m.offline.CONSTITUTION_SHA, 'craft_spec': 'preferences/qualified.md',
           'craft_spec_sha256': 'qualified-sha', 'original_craft_spec_sha256': 'original-sha',
           'models': {'author': {'model': m.offline.MODEL}, 'old_automatic_review': {'model': m.offline.MODEL}}}
    phases = [{'phase_id': 'original', 'config': cfg, 'config_sha256': m.base.digest(cfg), 'selected_ids': list(range(650))}]
    raw = tmp_path/'raw.json'; m.base.write_json(raw, {'request': {'model': m.offline.MODEL, 'temperature': .7, 'max_tokens': 12288, 'messages': []}})
    monkeypatch.setattr(m.offline, 'validate_dossier', lambda _: ({'author_kind': 'untouched_saved_final'}, {'author_raw': raw}))
    prov = m.provenance([], phases, [{'path': str(tmp_path)}]*66, 'exact-dataset')
    path = tmp_path/'generation_provenance.json'; m.base.write_json(path, prov)
    m.prepare_mixtures.synthetic_provenance(path, m.ARM)
    assert prov['origins'][-1]['config']['author_kinds'] == {'untouched_saved_final': 66}
    assert prov['origins'][-1]['config']['acceptance_route'] == m.offline.ROUTE
    assert 'old_automatic_review' not in prov['models']['offline_full_read']
