# ABOUTME: Exercise changed-evidence refusal and reversible exclusion restoration without network or models.
# ABOUTME: Verify original accepted content, checkpoint receipts, and rollback survive corrective review operations.
from pathlib import Path
import json

import pytest

from scratch.dataset_refresh import reconsider_exclusions as correction, run as runtime


def fixture_row(tmp_path):
    row = tmp_path / 'records' / 't1_001_v0'
    row.mkdir(parents=True)
    runtime.save_checkpoint(row / 'result.json', {'status': 'accepted'})
    result_sha = runtime.digest((row / 'result.json').read_bytes())
    runtime.save_checkpoint(row / 'independent_exclusion.json', {'result_sha256': result_sha, 'reason': 'Prior review'})
    evidence = tmp_path / 'evidence.json'
    entry = {'candidate_id': row.name, 'result_sha256': result_sha,
             'exclusion_sha256': runtime.digest((row / 'independent_exclusion.json').read_bytes()),
             'decision': 'restore', 'review_scope': 'full_conversation', 'decision_owner': 'root',
             'correction_kind': 'adjudication_error', 'reason': 'Full review'}
    evidence.write_text(json.dumps({'rows': [{**entry, 'result_path': str(row / 'result.json'),
                          'scope': 'full_system_user_reasoning_response_reread'}]}), encoding='utf-8')
    entry['evidence_sha256'] = runtime.digest(evidence.read_bytes())
    return row, evidence, entry


def test_bound_evidence_rejects_change(tmp_path):
    evidence = tmp_path / 'evidence.json'
    evidence.write_bytes(b'original')
    sha = runtime.digest(evidence.read_bytes())
    evidence.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed bound evidence'):
        correction.bound_file(evidence, sha)


def test_restore_preserves_exact_exclusion_and_result(tmp_path):
    row, evidence, entry = fixture_row(tmp_path)
    originals = {p.name: p.read_bytes() for p in row.iterdir()}
    archive = correction.archive_and_restore(row, entry, evidence, 'manifest')
    assert runtime.load_result(row / 'result.json')['status'] == 'accepted'
    for name in ('independent_exclusion.json', 'independent_exclusion.receipt.json'):
        assert not (row / name).exists()
        assert (archive / name).read_bytes() == originals[name]
    assert (row / 'result.json').read_bytes() == originals['result.json']
    assert (row / 'result.receipt.json').read_bytes() == originals['result.receipt.json']
    assert runtime.load_checkpoint(archive / 'correction.json')['manifest_sha256'] == 'manifest'
    assert correction.verify_history(row / 'result.json')[0]['reason'] == 'Full review'
    (archive / 'review_evidence.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed bound evidence'):
        correction.verify_history(row / 'result.json')


def test_restore_rolls_back_partial_removal(tmp_path, monkeypatch):
    row, evidence, entry = fixture_row(tmp_path)
    originals = {p.name: p.read_bytes() for p in row.iterdir()}
    unlink = Path.unlink

    def fail_second(path, *args, **kwargs):
        if path.name == 'independent_exclusion.receipt.json':
            raise OSError('simulated receipt removal failure')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_second)
    with pytest.raises(OSError):
        correction.archive_and_restore(row, entry, evidence, 'manifest')
    assert runtime.load_result(row / 'result.json')['status'] == 'rejected'
    for name, content in originals.items():
        assert (row / name).read_bytes() == content


@pytest.mark.parametrize('status', ['failed', 'rejected'])
def test_validator_never_relabels_original_failure(tmp_path, monkeypatch, status):
    root = tmp_path
    row = root / 'nonmoral-advice' / 'records' / 't1_001_v0'
    row.mkdir(parents=True)
    runtime.save_checkpoint(row / 'result.json', {'status': status})
    sha = runtime.digest((row / 'result.json').read_bytes())
    runtime.save_checkpoint(row / 'independent_exclusion.json', {'result_sha256': sha, 'reason': 'Review'})
    evidence = root / 'evidence.json'
    evidence.write_text('{}', encoding='utf-8')
    entry = {'root': str(root), 'arm': 'nonmoral-advice', 'candidate_id': row.name,
             'result_sha256': sha, 'exclusion_sha256': runtime.digest((row / 'independent_exclusion.json').read_bytes()),
             'evidence_path': str(evidence), 'evidence_sha256': runtime.digest(evidence.read_bytes()),
             'decision': 'restore', 'review_scope': 'full_conversation', 'reason': 'Reviewed',
             'decision_owner': 'root', 'correction_kind': 'adjudication_error'}
    evidence.write_text(json.dumps({'rows': [{**entry, 'result_path': str(row / 'result.json'),
                          'scope': 'full_system_user_reasoning_response_reread'}]}), encoding='utf-8')
    entry['evidence_sha256'] = runtime.digest(evidence.read_bytes())
    with pytest.raises(ValueError, match='originally accepted'):
        correction.validate_entry(entry, {(str(root.resolve()), 'nonmoral-advice'): {}})


def test_empty_evidence_refused(tmp_path):
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='exactly one'):
        correction.validate_evidence(evidence, {'candidate_id': 't1_001_v0'}, tmp_path / 'result.json')


def test_changed_review_copy_never_exposes_row(tmp_path, monkeypatch):
    row, evidence, entry = fixture_row(tmp_path)
    copyfile = correction.shutil.copyfile

    def altered_copy(source, destination):
        result = copyfile(source, destination)
        if Path(source) == evidence:
            Path(destination).write_text('{}', encoding='utf-8')
        return result

    monkeypatch.setattr(correction.shutil, 'copyfile', altered_copy)
    with pytest.raises(ValueError, match='Changed bound evidence'):
        correction.archive_and_restore(row, entry, evidence, 'manifest')
    assert runtime.load_result(row / 'result.json')['status'] == 'rejected'


@pytest.mark.parametrize('failed_write', ['output.json', 'output.receipt.json'])
def test_final_receipt_failure_rolls_back_all_rows(tmp_path, monkeypatch, failed_write):
    pairs = [fixture_row(tmp_path / str(i)) for i in range(2)]
    entries = [{**e, 'root': str(row.parents[1]), 'arm': 'nonmoral-advice', 'candidate_id': row.name}
               for row, _, e in pairs]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'entries': entries}), encoding='utf-8')
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(json.dumps({'manifest': str(manifest), 'manifest_sha256': runtime.digest(manifest.read_bytes()),
                               'output': str(tmp_path / 'output.json')}), encoding='utf-8')
    indexed = {e['root']: (row, evidence) for e, (row, evidence, _) in zip(entries, pairs)}
    monkeypatch.setattr(correction, 'validate_entry', lambda e, configs: indexed[e['root']])
    write_json = runtime.write_json
    failed = False

    def fail_once(path, value):
        nonlocal failed
        if Path(path).name == failed_write and not failed:
            failed = True
            raise OSError('simulated final receipt failure')
        return write_json(path, value)

    monkeypatch.setattr(runtime, 'write_json', fail_once)
    with pytest.raises(OSError, match='final receipt'):
        correction.reconsider(cfg, apply=True)
    for row, _, _ in pairs:
        assert runtime.load_result(row / 'result.json')['status'] == 'rejected'
    assert runtime.load_checkpoint(tmp_path / 'output.json')['rolled_back'] is True
