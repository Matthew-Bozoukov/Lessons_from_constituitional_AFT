# ABOUTME: Publish a closed failed pilot with independent evidence and its original code.
# ABOUTME: Stages immutable copies, excludes active calls, and marks it as audit data rather than training data.
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from scratch.dataset_refresh.run import digest, load_checkpoint, read_rows, write_json, write_rows
from src.infra.huggingface import push_run_dir
from src.naming import artifact_name, to_local


def publish(root, ledger_end, source_commit, push=False):
    root = Path(root)
    meta = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    date = meta['created_at'][:8]
    date = date[:4] + '-' + date[4:6] + '-' + date[6:]
    name = artifact_name('dataset-refresh-pilot-audit', date=date)
    out = root.parent / to_local(name)
    if out.exists():
        raise ValueError('Publication snapshot already exists; inspect/reuse it explicitly')
    out.mkdir()
    index = []
    for arm in meta['arms']:
        source = root / arm
        if not (source / 'pilot_gate.json').exists() or load_checkpoint(source / 'pilot_gate.json')['approved'] is not False:
            raise ValueError('Only closed failed pilots belong in this audit artifact')
        shutil.copytree(source, out / 'arms' / arm)
        for path in sorted((source / 'records').glob('*/result.json')):
            result = load_checkpoint(path)
            index.append({'arm': arm, 'candidate_id': result['candidate_id'], 'trait_id': result['trait_id'],
                          'automated_status': result['status'],
                          'evidence_path': f'arms/{arm}/records/{result["candidate_id"]}/result.json'})
    write_rows(out / 'pilot_index.jsonl', index)
    for filename in ('run_meta.json', 'active_phase.json', 'status.json'):
        shutil.copy2(root / filename, out / filename)
    entries = json.loads((root / 'budget/spend.json').read_text(encoding='utf-8'))[:ledger_end]
    if any(e['status'] == 'reserved' for e in entries):
        raise ValueError('Pilot still has an active request')
    write_json(out / 'budget/spend.json', entries)
    for entry in entries:
        filename = f'{entry["call_id"]:06d}.json'
        target = out / 'budget/raw_calls' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / 'budget/raw_calls' / filename, target)
    code = subprocess.check_output(['git', 'show', source_commit + ':scratch/dataset_refresh/run.py'])
    (out / 'original_runner.py').write_bytes(code)
    write_json(out / 'publication_manifest.json', {'source_commit': source_commit,
        'original_runner_git_blob_sha256': digest(code), 'ledger_end_exclusive': ledger_end,
        'files': {str(p.relative_to(out)).replace('\\', '/'): digest(p.read_bytes()) for p in out.rglob('*') if p.is_file()}})
    fields = {
        'experiment': 'Failed first pilots for moral low-stakes and nonmoral craft advice refresh; audit evidence only',
        'date_generated': meta['created_at'],
        'constitution': 'constitutions/claude_distilled_09_principles/constitution.md; low-stakes principle generation, nonmoral compatibility review only',
        'source_repo': 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ ' + source_commit,
        'models': 'anthropic/claude-haiku-4.5 and anthropic/claude-sonnet-5 via Anthropic/OpenRouter; mutable API IDs, exact requests and observed provider usage preserved',
        'generation_config': 'Frozen per-arm config.json; eighteen attempts per arm. Independent review: each arm2 pass,14 reject,2 failed. Automated judge accepted all16 complete outputs in each arm.',
        'schema': 'pilot_index.jsonl is a flat evidence index. arms/ holds frozen inputs, full stage records, independent reviews and rejected pilot gates. No approved training dataset.',
        'provenance': 'uv run python scratch/dataset_refresh/run.py pilot --root output/2026-09-14_dataset_refresh --ceiling 20 --workers 8; original runner source preserved. Source revisions in run_meta and per-arm config.',
        'budget': f'{ledger_end} physical calls; provider-reported ${sum(e.get("api_reported_cost_usd") or 0 for e in entries):.6f}; charged/reserved ${sum(e["charged_or_reserved_usd"] for e in entries):.6f}, including uncertain provider failure.',
        'limitations': 'Failed pilot evidence must not be consumed as training data. Local reviewers are independent agent contexts, not human annotations. Two independent gates failed; one recipe revision is separately authorized. Preparation git SHA predates original code commit; source_repo and original_runner identify the actual paid pilot implementation.',
    }
    write_json(out / 'card_fields.json', fields)
    if push:
        return push_run_dir(out, name, fields, private=False,
            front_matter={'tags': ['dataset-audit', 'failed-pilot'],
                          'configs': [{'config_name': 'default', 'default': True, 'data_files': 'pilot_index.jsonl'}]})
    return str(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--ledger-end', type=int, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    print(publish(a.root, a.ledger_end, a.source_commit, a.push))
