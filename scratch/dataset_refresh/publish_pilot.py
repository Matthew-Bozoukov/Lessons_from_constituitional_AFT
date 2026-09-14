# ABOUTME: Publish a closed failed pilot with independent evidence and its original code.
# ABOUTME: Stages immutable copies, excludes active calls, and marks it as audit data rather than training data.
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from scratch.dataset_refresh.run import digest, load_checkpoint, save_checkpoint, read_rows, write_json, write_rows
from src.infra.huggingface import push_run_dir
from src.naming import artifact_name, to_local


def publish(root, ledger_end, source_commit, push=False, subject='dataset-refresh-pilot-audit'):
    root = Path(root)
    meta = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    date = meta['created_at'][:8]
    date = date[:4] + '-' + date[4:6] + '-' + date[6:]
    name = artifact_name(subject, date=date)
    out = root.parent / to_local(name)
    if out.exists():
        raise ValueError('Publication snapshot already exists; inspect/reuse it explicitly')
    out.mkdir()
    index, gates = [], {}
    for arm in meta['arms']:
        source = root / arm
        if not (source / 'pilot_gate.json').exists() or load_checkpoint(source / 'pilot_gate.json')['approved'] is not False:
            raise ValueError('Only closed failed pilots belong in this audit artifact')
        gates[arm] = load_checkpoint(source / 'pilot_gate.json')
        annotations = json.loads((source / 'pilot_independent_review.json').read_text(encoding='utf-8'))
        annotation_rows = {r['candidate_id']: r for r in annotations['rows']}
        shutil.copytree(source, out / 'arms' / arm)
        for path in sorted((source / 'records').glob('*/result.json')):
            result = load_checkpoint(path)
            annotation = annotation_rows[result['candidate_id']]
            index.append({'arm': arm, 'candidate_id': result['candidate_id'], 'trait_id': result['trait_id'],
                          'automated_status': result['status'],
                          'independent_status': annotation.get('independent_disposition', annotation.get('full_row_decision', annotation.get('decision'))),
                          'evidence_path': f'arms/{arm}/records/{result["candidate_id"]}/result.json'})
    write_rows(out / 'pilot_index.jsonl', index)
    for filename in ('run_meta.json', 'active_phase.json', 'status.json'):
        shutil.copy2(root / filename, out / filename)
    budget_root = Path(meta.get('budget_root', str(root / 'budget')))
    shared_entries = json.loads((budget_root / 'spend.json').read_text(encoding='utf-8'))[:ledger_end]
    entries = [e for e in shared_entries if e.get('run_root') == str(root.resolve()) or
               (not meta.get('revision_of') and e.get('run_root') is None)]
    if any(e['status'] == 'reserved' for e in entries):
        raise ValueError('Pilot still has an active request')
    write_json(out / 'budget/spend.json', entries)
    write_json(out / 'budget/shared_budget_snapshot.json', shared_entries)
    for entry in entries:
        filename = f'{entry["call_id"]:06d}.json'
        target = out / 'budget/raw_calls' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(budget_root / 'raw_calls' / filename, target)
    code = subprocess.check_output(['git', 'show', source_commit + ':scratch/dataset_refresh/run.py'])
    (out / 'original_runner.py').write_bytes(code)
    write_json(out / 'publication_manifest.json', {'source_commit': source_commit,
        'original_runner_git_blob_sha256': digest(code), 'ledger_end_exclusive': ledger_end,
        'files': {str(p.relative_to(out)).replace('\\', '/'): digest(p.read_bytes()) for p in out.rglob('*') if p.is_file()}})
    fields = {
        'experiment': 'Failed pilots for moral low-stakes and nonmoral craft advice refresh; audit evidence only',
        'date_generated': meta['created_at'],
        'constitution': 'constitutions/claude_distilled_09_principles/constitution.md; low-stakes principle generation, nonmoral compatibility review only',
        'source_repo': 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ ' + source_commit,
        'models': 'anthropic/claude-haiku-4.5 and anthropic/claude-sonnet-5 via Anthropic/OpenRouter; mutable API IDs, exact requests and observed provider usage preserved',
        'generation_config': 'Frozen per-arm config.json; eighteen attempts per arm. Independent disposition counts and immutable recipe hashes in arms/*/pilot_gate.json. ' + json.dumps({arm: {k: v for k, v in gate.items() if k in ('independent_pass', 'independent_reject', 'incomplete', 'attempted')} for arm, gate in gates.items()}),
        'schema': 'pilot_index.jsonl is a flat evidence index. arms/ holds frozen inputs, full stage records, independent reviews and rejected pilot gates. No approved training dataset.',
        'provenance': f'uv run python scratch/dataset_refresh/run.py pilot --root {root.as_posix()} --ceiling 20 --workers 8; original runner source preserved. Source revisions in run_meta and per-arm config.',
        'budget': f'{len(entries)} physical calls in this pilot; provider-reported ${sum(e.get("api_reported_cost_usd") or 0 for e in entries):.6f}; charged/reserved ${sum(e["charged_or_reserved_usd"] for e in entries):.6f}. Shared-budget snapshot includes earlier phases separately.',
        'limitations': 'Failed pilot evidence must not be consumed as training data. Local reviewers are independent agent contexts, not human annotations. No production, training or evaluation is approved. source_repo and original_runner identify the actual paid pilot implementation; the initial pilot was prepared before its code commit.',
    }
    write_json(out / 'card_fields.json', fields)
    if push:
        return push_run_dir(out, name, fields, private=False,
            front_matter={'tags': ['dataset-audit', 'failed-pilot'],
                          'configs': [{'config_name': 'default', 'default': True, 'data_files': 'pilot_index.jsonl'}]})
    return str(out)


def refresh_annotations(root):
    root = Path(root)
    meta = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    date = meta['created_at'][:8]
    date = date[:4] + '-' + date[4:6] + '-' + date[6:]
    name = artifact_name('dataset-refresh-pilot-audit', date=date)
    out = root.parent / to_local(name)
    for arm in meta['arms']:
        source = root / arm
        gate = load_checkpoint(source / 'pilot_gate.json')
        current_hash = digest((source / 'pilot_independent_review.json').read_bytes())
        if gate['report_sha256'] != current_hash:
            gate.update(original_report_sha256=gate['report_sha256'], report_sha256=current_hash,
                        adjudication='Original annotations retained; two ambiguous nonmoral content labels withdrawn, full-row metadata failures unchanged.')
            save_checkpoint(source / 'pilot_gate.json', gate)
        for path in source.glob('pilot_*'):
            if path.is_file():
                shutil.copy2(path, out / 'arms' / arm / path.name)
    calibration = root / 'reviewer_calibration'
    if calibration.exists():
        shutil.copytree(calibration, out / 'reviewer_calibration', dirs_exist_ok=True)
        entries = json.loads((root / 'budget/spend.json').read_text(encoding='utf-8'))
        calibration_calls = [e for e in entries if e.get('run_root') == str(calibration.resolve())]
        if any(e['status'] == 'reserved' for e in calibration_calls):
            raise ValueError('Calibration still active')
        write_json(out / 'reviewer_calibration/spend.json', calibration_calls)
        for entry in calibration_calls:
            filename = f'{entry["call_id"]:06d}.json'
            target = out / 'reviewer_calibration/raw_calls' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / 'budget/raw_calls' / filename, target)
    fields = json.loads((out / 'card_fields.json').read_text(encoding='utf-8'))
    fields['adjudication'] = 'See per-arm pilot_independent_review_adjudication files. Two ambiguous nonmoral content-negative labels were withdrawn; full-row metadata failures and2/18 pass counts unchanged. Revised reviewers rejected all five clear calibration defects; seven raw checks plus corrected label adjudication retained. Calibration implementation commit3815eab2; no prompt retuning after calibration.'
    write_json(out / 'card_fields.json', fields)
    manifest = json.loads((out / 'publication_manifest.json').read_text(encoding='utf-8'))
    manifest['files'] = {str(p.relative_to(out)).replace('\\', '/'): digest(p.read_bytes())
                         for p in out.rglob('*') if p.is_file() and p.name not in ('publication_manifest.json', 'README.md')}
    write_json(out / 'publication_manifest.json', manifest)
    return push_run_dir(out, name, fields, private=False,
                        front_matter={'tags': ['dataset-audit', 'failed-pilot'],
                          'configs': [{'config_name': 'default', 'default': True, 'data_files': 'pilot_index.jsonl'}]})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--ledger-end', type=int)
    p.add_argument('--source-commit')
    p.add_argument('--push', action='store_true')
    p.add_argument('--refresh-annotations', action='store_true')
    p.add_argument('--subject', default='dataset-refresh-pilot-audit')
    a = p.parse_args()
    print(refresh_annotations(a.root) if a.refresh_annotations else publish(a.root, a.ledger_end, a.source_commit, a.push, a.subject))
