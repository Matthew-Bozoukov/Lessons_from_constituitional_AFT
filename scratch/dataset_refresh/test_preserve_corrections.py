# ABOUTME: Test correction archive input bindings and byte-preserving row/evidence export without network.
# ABOUTME: Refuse drifted results, ledger or readiness before publication artifacts can be produced.
import json
from pathlib import Path
import tarfile

import pytest
from scratch.dataset_refresh import preserve_partial as preservation, run as runtime


def setup_inputs(tmp_path, monkeypatch):
    root = tmp_path / 'origin'
    row = root / 'nonmoral-advice/records/t1_001_v0'
    row.mkdir(parents=True)
    runtime.save_checkpoint(row / 'result.json', {'status': 'accepted', 'record': {
        'system': 'system', 'user': 'user', 'reasoning': 'reasoning', 'response': 'response'}})
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    origin = {'root': str(root), 'arm': 'nonmoral-advice', 'candidate_id': row.name,
              'result_sha256': runtime.digest((row / 'result.json').read_bytes())}
    runtime.write_rows(evidence / 'conversations_for_audit.jsonl', [{'messages': [
        {'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'user'},
        {'role': 'assistant', 'content': 'response', 'reasoning_content': 'reasoning'}], 'metadata': {'origin': origin}}])
    runtime.write_json(evidence / 'analysis_readiness.json', {'analysis_rows': 1, 'token_failures': [],
        'input_sha256': runtime.digest((evidence / 'conversations_for_audit.jsonl').read_bytes())})
    budget = tmp_path / 'budget'
    budget.mkdir()
    (budget / 'spend.json').write_text('[]', encoding='utf-8')
    manifest = {'parent_archive': {'repo': 'dougalldeepmind/2026-09-15-dataset-refresh-incomplete-audit',
                 'revision': 'f455cc9a2224d65c3861fd83a7f57c4c49c8072a'},
                'budget_root': str(budget), 'ledger_sha256': runtime.digest(b'[]'),
                'evidence_root': str(evidence), 'summary': 'Test correction',
                'selection_audits': {'nonmoral-advice': str(evidence)},
                'readiness': {'selected_counts': {'nonmoral-advice': 1}},
                'rows': [{'root': str(root), 'arm': 'nonmoral-advice', 'candidate_id': row.name,
                          'result_sha256': runtime.digest((row / 'result.json').read_bytes())}]}
    path = tmp_path / 'manifest.json'
    runtime.write_json(path, manifest)
    monkeypatch.setattr(runtime, 'validate_arm', lambda *a: {})
    monkeypatch.setattr(preservation, 'freeze_code', lambda *a: {'commit': 'test'})
    return path, manifest, row


def test_correction_archive_preserves_exact_row(tmp_path, monkeypatch):
    path, manifest, row = setup_inputs(tmp_path, monkeypatch)
    out = preservation.prepare_corrections(path, tmp_path / 'published', 'test', '2026-09-15')
    with tarfile.open(out / 'rows/row_000.tar.gz') as archive:
        assert archive.extractfile('result.json').read() == (row / 'result.json').read_bytes()
    files = json.loads((out / 'files_manifest.json').read_text(encoding='utf-8'))
    assert files == preservation.inventory(out, ('files_manifest.json',))
    assert json.loads((out / 'preparation_receipt.json').read_text())['inference_calls'] == 0
    assert (out / 'source_manifest.json').read_bytes() == path.read_bytes()


@pytest.mark.parametrize('drift', ['ledger', 'result', 'count', 'excluded'])
def test_correction_drift_refused(tmp_path, monkeypatch, drift):
    path, manifest, row = setup_inputs(tmp_path, monkeypatch)
    if drift == 'ledger':
        (Path(manifest['budget_root']) / 'spend.json').write_text('[{}]', encoding='utf-8')
    elif drift == 'result':
        (row / 'result.json').write_text('{}', encoding='utf-8')
    elif drift == 'excluded':
        runtime.save_checkpoint(row / 'independent_exclusion.json', {
            'result_sha256': runtime.digest((row / 'result.json').read_bytes()), 'reason': 'New hold'})
    else:
        manifest['readiness']['selected_counts']['nonmoral-advice'] = 2
        runtime.write_json(path, manifest)
    with pytest.raises((ValueError, runtime.BudgetStop)):
        preservation.prepare_corrections(path, tmp_path / 'published', 'test', '2026-09-15')
    assert not (tmp_path / 'published').exists()


def test_manifest_drift_during_archive_refused(tmp_path, monkeypatch):
    path, manifest, row = setup_inputs(tmp_path, monkeypatch)
    archive_tree = preservation.archive_tree

    def mutate_manifest(*args, **kwargs):
        result = archive_tree(*args, **kwargs)
        path.write_text('{}', encoding='utf-8')
        return result

    monkeypatch.setattr(preservation, 'archive_tree', mutate_manifest)
    with pytest.raises(ValueError, match='changed during preservation'):
        preservation.prepare_corrections(path, tmp_path / 'published', 'test', '2026-09-15')
    assert (tmp_path / 'published/PREPARATION_FAILED').exists()
