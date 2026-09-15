# ABOUTME: Offline end-to-end selection validation across two receipt-bound phases, without model or Hub calls.
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from scratch.dataset_refresh import publish_composite as mod


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp('composite')
    constitution = base / 'constitution.md'
    constitution.write_text('Maintain the supplied facts.', encoding='utf-8')
    roots = [base / 'phase0', base / 'phase1']
    arm = 'nonmoral-advice'
    entries, phase_candidates = [], [[], []]
    configs = []
    for n, root in enumerate(roots):
        cfg = {'pipeline': arm, 'per_row_regime': True, 'constitution': str(constitution),
               'constitution_sha256': mod.runtime.digest(constitution.read_bytes()), 'phase_prompt': str(n),
               'source': {'revision': 'a' * 40},
               'models': {role: {'model': 'anthropic/claude-sonnet-5'} for role in ('scenario', 'respond', 'rewrite', 'review')},
               'preflight': {'acceptance': {'required_true': ['eligible']}},
               'acceptance': {'required_true': ['accepted']}}
        configs.append(cfg)
        mod.runtime.write_json(root / arm / 'config.json', cfg)
    for trait, count in mod.runtime.quotas().items():
        for i in range(count):
            phase = i % 2
            root, cfg = roots[phase], configs[phase]
            cid = f'{trait}_{i // 2:03d}_v0'
            source = {'scenario_id': f'original_{trait}_{i}', 'trait_id': trait, 'facts': 'Historic inspiration.'}
            candidate = {'candidate_id': cid, 'trait_id': trait, 'source_id': source['scenario_id'], 'source': source}
            phase_candidates[phase].append(candidate)
            record = {'scenario_id': cid, 'trait_id': trait, 'system': 'Offer advice.',
                      'user': f'Phase{phase} {trait} decision{i}.', 'reasoning': 'Weigh the actual alternatives.',
                      'response': 'Choose the supported alternative.', 'source_id': candidate['source_id'],
                      'source_record_sha256': mod.runtime.digest(source), 'parent_revision': 'a' * 40,
                      'paired_counterfactual': False, 'lineage_kind': 'unpaired_mechanism_inspiration',
                      'custom_metadata': {'retain': 'complete metadata'}, 'response_repair_count': 0,
                      'rewrite_changes': 'Nontraining author audit detail.'}
            record['source_facts'] = [record['user']]
            verdict = {'accepted': True, 'issues': [], 'assessment': 'No material issue.'}
            result = {'candidate_id': cid, 'trait_id': trait, 'status': 'accepted', 'record': record,
                      'accepted_attempt': 0, 'review': {'accepted': True}, 'grounding_review': verdict}
            row = root / arm / 'records' / cid
            for name, value in [('identity', {'candidate_sha256': mod.runtime.digest(candidate), 'config_sha256': mod.runtime.digest(cfg)}),
                    ('preflight', {'eligible': True}), ('grounding_0', verdict), ('review_0', result['review']),
                    ('answer_0', mod.per_row.conversation(record)), ('result', result)]:
                mod.runtime.save_checkpoint(row / (name + '.json'), value)
            entries.append({'root': str(root), 'arm': arm, 'candidate_id': cid,
                            'result_sha256': mod.runtime.digest((row / 'result.json').read_bytes())})
    for n, root in enumerate(roots):
        mod.runtime.write_rows(root / arm / 'source.jsonl', [c['source'] for c in phase_candidates[n]])
        mod.runtime.write_rows(root / arm / 'candidates.jsonl', phase_candidates[n])
        mod.runtime.save_checkpoint(root / 'run_meta.json', {'arms': {arm: {
            'config_sha256': mod.runtime.digest(configs[n]),
            'source_snapshot_sha256': mod.runtime.digest((root / arm / 'source.jsonl').read_bytes()),
            'candidates_sha256': mod.runtime.digest((root / arm / 'candidates.jsonl').read_bytes())}}})
        phase = {'phase': 'production', 'code_sha256': mod.runtime.digest(Path(mod.runtime.__file__).read_bytes()),
                 'per_row_code_sha256': mod.runtime.digest(Path(mod.per_row.__file__).read_bytes())}
        mod.runtime.write_json(root / 'active_phase.json', phase)
        mod.runtime.save_checkpoint(root / 'phases' / 'execution.json', phase)
    return {'arm': arm, 'entries': entries}


@contextmanager
def preserve(*paths):
    data = {p: p.read_bytes() if p.exists() else None for p in paths}
    try:
        yield
    finally:
        for path, value in data.items():
            if value is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(value)


def test_two_phase716_exact_quotas_namespaces_full_metadata_and_source_preserved(corpus):
    rows, phases, selection = mod.validate_selection(corpus)
    assert len(rows) == 716 and len(phases) == 2
    assert len({r['metadata']['scenario_id'] for r in rows}) == 716
    assert len({r['metadata']['original_scenario_id'] for r in rows}) < 716
    assert selection['quotas'] == mod.runtime.quotas()
    for row, entry in zip(rows, corpus['entries']):
        metadata = row['metadata']
        assert metadata['custom_metadata'] == {'retain': 'complete metadata'}
        assert 'rewrite_changes' not in metadata
        assert metadata['origin']['result_sha256'] == entry['result_sha256']
        assert metadata['source_record_sha256']
        assert metadata['source_facts'] == [row['messages'][1]['content']]
        assert metadata['paired_counterfactual'] is False


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'bad_hash', 'mixed_arm', 'unsafe_id'])
def test_invalid_explicit_selection_refused(corpus, mutation):
    selection = deepcopy(corpus)
    if mutation == 'missing':
        selection['entries'].pop()
    elif mutation == 'duplicate':
        selection['entries'][1] = dict(selection['entries'][0])
    elif mutation == 'bad_hash':
        selection['entries'][0]['result_sha256'] = '0' * 64
    elif mutation == 'mixed_arm':
        selection['entries'][0]['arm'] = 'da-lowstakes-refresh'
    else:
        selection['entries'][0]['candidate_id'] = '../../outside'
    with pytest.raises(ValueError):
        mod.validate_selection(selection)


def test_missing_actual_review_and_bound_independent_exclusion_refused(corpus):
    e = corpus['entries'][0]
    row = Path(e['root']) / e['arm'] / 'records' / e['candidate_id']
    path = row / 'grounding_0.json'
    with preserve(path):
        path.unlink()
        with pytest.raises((FileNotFoundError, mod.runtime.BudgetStop)):
            mod.validate_selection(corpus)
    exclusion = row / 'independent_exclusion.json'
    with preserve(exclusion, exclusion.with_suffix('.receipt.json')):
        mod.runtime.save_checkpoint(exclusion, {'result_sha256': e['result_sha256'], 'reason': 'Independent material defect.'})
        with pytest.raises(ValueError, match='not accepted'):
            mod.validate_selection(corpus)


def test_record_lineage_change_refused_even_with_updated_terminal_receipt(corpus):
    selection = deepcopy(corpus)
    e = selection['entries'][0]
    path = Path(e['root']) / e['arm'] / 'records' / e['candidate_id'] / 'result.json'
    with preserve(path, path.with_suffix('.receipt.json')):
        result = mod.runtime.load_checkpoint(path)
        result['record']['source_record_sha256'] = '0' * 64
        mod.runtime.save_checkpoint(path, result)
        e['result_sha256'] = mod.runtime.digest(path.read_bytes())
        with pytest.raises(ValueError, match='source lineage'):
            mod.validate_selection(selection)


def test_duplicate_prompt_across_phases_refused_after_review_checks(corpus):
    selection = deepcopy(corpus)
    a, b = selection['entries'][:2]
    pa = Path(a['root']) / a['arm'] / 'records' / a['candidate_id'] / 'result.json'
    pb = Path(b['root']) / b['arm'] / 'records' / b['candidate_id'] / 'result.json'
    answer = pb.parent / 'answer_0.json'
    with preserve(pb, pb.with_suffix('.receipt.json'), answer, answer.with_suffix('.receipt.json')):
        result = mod.runtime.load_checkpoint(pb)
        result['record']['user'] = mod.runtime.load_checkpoint(pa)['record']['user']
        result['record']['source_facts'] = [result['record']['user']]
        mod.runtime.save_checkpoint(pb, result)
        mod.runtime.save_checkpoint(answer, mod.per_row.conversation(result['record']))
        b['result_sha256'] = mod.runtime.digest(pb.read_bytes())
        with pytest.raises(ValueError, match='Duplicate selected prompt'):
            mod.validate_selection(selection)


def test_wrong_quota_cannot_hide_inside716():
    with pytest.raises(ValueError, match='quotas'):
        mod.publication.validate_export([{'metadata': {'trait_id': 't1'}}] * 716, {})


def test_empty_lock_is_accounted_but_nonempty_unknown_file_refused(tmp_path):
    source = tmp_path / 'arm'
    row = source / 'records' / 't1_000_v0'
    row.mkdir(parents=True)
    (row / 'independent_repair.lock').touch()
    mod.runtime.save_checkpoint(row / 'result.json', {'status': 'failed'})
    mod.freeze_origin_arm(source, tmp_path / 'archive')
    assert len(mod.publication.read_json(tmp_path / 'archive' / 'operational_lock_files.json')) == 1
    (row / 'unexpected.txt').write_text('Unaccounted', encoding='utf-8')
    with pytest.raises(ValueError, match='Unexpected'):
        mod.freeze_origin_arm(source, tmp_path / 'refused')


def test_adopted_independent_repair_requires_full_bound_adoption_chain(tmp_path):
    row = tmp_path / 'row'
    archive = row / 'recovered_failures' / 'independent_repair_100'
    original = {'record': {'system': 'Advice.', 'user': 'Fixed request.', 'reasoning': 'Old.', 'response': 'Old.'}}
    mod.runtime.save_checkpoint(archive / 'result.json', original)
    old_sha = mod.runtime.digest((archive / 'result.json').read_bytes())
    exclusion = {'result_sha256': old_sha, 'reason': 'Material old error.'}
    mod.runtime.save_checkpoint(archive / 'independent_exclusion.json', exclusion)
    manifest = {'source_result_sha256': old_sha, 'exclusion': exclusion,
                'script_sha256': mod.runtime.digest(Path(mod.__file__).with_name('repair_independent.py').read_bytes())}
    mod.runtime.save_checkpoint(row / 'independent_repair_100.json', manifest)
    result = {'accepted_attempt': 100, 'source_result_sha256': old_sha,
              'independent_repair_manifest_sha256': mod.runtime.digest((row / 'independent_repair_100.json').read_bytes()),
              'record': {**original['record'], 'reasoning': 'Corrected.', 'response': 'Corrected.'}}
    candidate_path = row / 'independent_candidate_100.json'
    mod.runtime.save_checkpoint(candidate_path, result)
    adoption = {'source_result_sha256': old_sha, 'source_exclusion': exclusion,
                'candidate_sha256': mod.runtime.digest(candidate_path.read_bytes()), 'independent_audit_reason': 'Fresh answer independently checked.'}
    adoption_path = row / 'independent_adoption_100.json'
    mod.runtime.save_checkpoint(adoption_path, adoption)
    mod.validate_adoption(row / 'result.json', result)
    adoption['independent_audit_reason'] = ''
    mod.runtime.save_checkpoint(adoption_path, adoption)
    with pytest.raises(ValueError, match='adoption provenance'):
        mod.validate_adoption(row / 'result.json', result)


def test_snapshot_cannot_be_nested_inside_source(corpus, tmp_path):
    manifest = tmp_path / 'selection.json'
    mod.runtime.write_json(manifest, corpus)
    with pytest.raises(ValueError, match='outside every'):
        mod.prepare(manifest, Path(corpus['entries'][0]['root']) / 'nested_snapshot', 'HEAD', 0, '2026-09-15')


def quality_fixture(tmp_path, audit_sha='a' * 64):
    source = tmp_path / 'quality'
    mod.runtime.write_json(source / 'corpus_audit.json', {'input_sha256': audit_sha, 'rows': 716})
    (source / 'selection_policy.md').write_text('Explicit selection with independent judgment.', encoding='utf-8')
    files = {p.name: mod.runtime.digest(p.read_bytes()) for p in source.iterdir()}
    mod.runtime.write_json(source / 'quality_manifest.json', {'dataset_sha256': 'a' * 64, 'files': files})
    return source


def test_quality_bundle_binds_exact_dataset_and_every_file(tmp_path):
    source = quality_fixture(tmp_path)
    result = mod.freeze_quality(source, tmp_path / 'snapshot', 'a' * 64)
    assert result['dataset_sha256'] == 'a' * 64
    for name in result['files']:
        assert (tmp_path / 'snapshot' / name).read_bytes() == (source / name).read_bytes()
    with pytest.raises(ValueError, match='different selected dataset'):
        mod.freeze_quality(source, tmp_path / 'wrong', 'b' * 64)
    (source / 'unlisted.md').write_text('Forgotten evidence.', encoding='utf-8')
    with pytest.raises(ValueError, match='every supplied artifact'):
        mod.freeze_quality(source, tmp_path / 'unlisted', 'a' * 64)


def test_quality_audit_stale_input_refused(tmp_path):
    source = quality_fixture(tmp_path, 'b' * 64)
    with pytest.raises(ValueError, match='audit input differs'):
        mod.freeze_quality(source, tmp_path / 'snapshot', 'a' * 64)


def test_quality_copy_race_refused(tmp_path, monkeypatch):
    source = quality_fixture(tmp_path)
    original = mod.shutil.copy2
    def changed(src, dst):
        original(src, dst)
        Path(src).write_bytes(Path(src).read_bytes() + b'\n')
    monkeypatch.setattr(mod.shutil, 'copy2', changed)
    with pytest.raises(ValueError, match='changed during copy'):
        mod.freeze_quality(source, tmp_path / 'snapshot', 'a' * 64)


def test_quality_symlink_refused(tmp_path, monkeypatch):
    source = quality_fixture(tmp_path)
    try:
        (source / 'escape').symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        # Windows may deny unprivileged symlink creation; still exercise the guard.
        (source / 'escape').mkdir()
        original = Path.is_symlink
        monkeypatch.setattr(Path, 'is_symlink', lambda p: p.name == 'escape' or original(p))
    with pytest.raises(ValueError, match='escapes|link'):
        mod.freeze_quality(source, tmp_path / 'snapshot', 'a' * 64)


def test_complete_snapshot_freezes_quality_origin_audits_and_scoped_accounting(corpus, tmp_path, monkeypatch):
    rows, phases, _ = mod.validate_selection(corpus)
    selection_file = tmp_path / 'explicit.json'
    mod.runtime.write_json(selection_file, corpus)
    preview = tmp_path / 'preview.jsonl'
    mod.runtime.write_rows(preview, rows)
    dataset_sha = mod.runtime.digest(preview.read_bytes())
    quality = tmp_path / 'quality'
    mod.runtime.write_json(quality / 'tokenizer.json', {'dataset_sha256': dataset_sha, 'max_tokens': 8192})
    mod.runtime.write_json(quality / 'quality_manifest.json', {'dataset_sha256': dataset_sha,
        'files': {'tokenizer.json': mod.runtime.digest((quality / 'tokenizer.json').read_bytes())}})
    budget, shared = tmp_path / 'budget', []
    for n, phase in enumerate(phases):
        request = {'model': 'anthropic/claude-sonnet-5', 'messages': [{'role': 'user', 'content': 'Fixture only.'}]}
        call = {'call_id': n, 'run_root': phase['root'], 'arm': phase['arm'], 'stage': 'scenario',
                'model': request['model'], 'request_sha256': mod.runtime.digest(request),
                'status': 'settled', 'api_reported_cost_usd': 0.01, 'charged_or_reserved_usd': 0.01}
        shared.append(call)
        mod.runtime.write_json(budget / 'raw_calls' / f'{n:06d}.json', {'request': request, 'accounting': call})
    monkeypatch.setattr(mod.publication, 'scoped_ledger', lambda meta, root, arm, end:
                        (budget, shared, [c for c in shared if c['run_root'] == str(root)]))
    def freeze_code(out, commit, cfg):
        target = out / 'source_code/scratch/dataset_refresh'
        target.mkdir(parents=True)
        for path in Path(mod.__file__).parent.glob('*.py'):
            (target / path.name).write_bytes(path.read_bytes())
        return 'f' * 40
    monkeypatch.setattr(mod.publication, 'freeze_code', freeze_code)
    independent = Path(phases[0]['root']) / corpus['arm'] / 'independent_all.json'
    with preserve(independent):
        mod.runtime.write_json(independent, {'fixture': 'Independent origin evidence.'})
        out = Path(mod.prepare(selection_file, tmp_path / 'snapshot', 'f' * 40, 2, '2026-09-15', quality))
    manifest = mod.publication.read_json(out / 'publication_manifest.json')
    assert manifest['dataset_sha256'] == dataset_sha
    assert manifest['selected_quality']['dataset_sha256'] == dataset_sha
    assert mod.publication.read_json(out / 'audit/budget/shared_budget_snapshot.json') == shared
    assert len(mod.runtime.read_rows(out / 'audit/budget/raw_calls.jsonl')) == 2
    assert len(list((out / 'audit/phases').glob('*/arm/independent_all.json'))) == 1
    assert len(list((out / 'audit/phases').glob('*/arm/record_checkpoints.jsonl'))) == 2
    assert mod.publication.read_json(out / 'card_front_matter.json')['configs'][0]['data_files'] == 'dataset.jsonl'
    assert all(mod.runtime.digest((out / name).read_bytes()) == sha for name, sha in manifest['files'].items())
