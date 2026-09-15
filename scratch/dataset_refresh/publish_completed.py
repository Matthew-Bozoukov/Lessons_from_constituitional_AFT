# ABOUTME: Freeze and publish a validated completed 716-row synthetic refresh corpus.
# ABOUTME: Preserve stages, source pins and scoped API evidence; publication requires an explicit push command.
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess

from scratch.dataset_refresh.run import digest, export, load_checkpoint, load_result, quotas, read_rows, validate_arm, write_json, write_rows
from src.infra.huggingface import card_markdown, push_run_dir, training_data_tags
from src.naming import synth_name, to_local


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_audit(value):
    """Fail closed on transport credentials, without mistaking ordinary message text for a key."""
    if isinstance(value, dict):
        for key, item in value.items():
            if re.sub('[^a-z]', '', key.lower()) in {
                'headers', 'authorization', 'apikey', 'accesstoken', 'hftoken', 'cookie', 'setcookie'}:
                raise ValueError('Credential/transport field in audit artifact: ' + key)
            safe_audit(item)
    elif isinstance(value, list):
        for item in value:
            safe_audit(item)


def validate_export(rows, selection):
    if len(rows) != 716 or Counter(r['metadata']['trait_id'] for r in rows) != quotas():
        raise ValueError('Release needs exactly 716 rows and the frozen nine-trait quotas')
    ids = [r['metadata']['scenario_id'] for r in rows]
    if len(set(ids)) != 716 or ids != selection['scenario_ids'] or selection['quotas'] != quotas():
        raise ValueError('Selection identity/quotas differ from exported data')
    for row in rows:
        if [m['role'] for m in row['messages']] != ['system', 'user', 'assistant']:
            raise ValueError('Unexpected training schema')
        if not all(isinstance(m['content'], str) and m['content'].strip() for m in row['messages']):
            raise ValueError('Empty conversation content')
        if not row['messages'][-1].get('reasoning_content', '').strip():
            raise ValueError('Missing detailed rationale')


def scoped_ledger(meta, root, arm, ledger_end):
    budget = Path(meta.get('budget_root', root / 'budget'))
    ledger = read_json(budget / 'spend.json')
    if not 0 <= ledger_end <= len(ledger):
        raise ValueError('ledger_end must explicitly bound the existing shared ledger')
    shared = ledger[:ledger_end]
    # New production calls always identify their run. Never import unscoped historic calls.
    scoped = [e for e in shared if e.get('run_root') == str(root.resolve()) and e.get('arm') == arm]
    if not scoped:
        raise ValueError('No explicitly run-scoped calls found for the completed arm')
    if any(e.get('status') == 'reserved' for e in scoped):
        raise ValueError('This arm still has an active API reservation')
    for entry in ledger[ledger_end:]:
        if entry.get('run_root') == str(root.resolve()) and entry.get('arm') == arm:
            raise ValueError('Ledger cutoff excludes calls belonging to this arm')
    safe_audit(shared)
    return budget, shared, scoped


def freeze_code(out, commit, cfg):
    commit = subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], text=True).strip()
    tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], text=True).splitlines()
    required = {'scratch/dataset_refresh/run.py', 'scratch/dataset_refresh/per_row.py',
                'scratch/dataset_refresh/publish_completed.py', cfg['constitution'],
                'pyproject.toml', 'uv.lock', 'configs/data/synth/' + cfg['pipeline'] + '.yaml'}
    if cfg.get('craft_spec'):
        required.add(cfg['craft_spec'])
    if required - set(tracked):
        raise ValueError('Commit lacks required recipe/code files: ' + str(sorted(required - set(tracked))))
    selected = set(required) | {p for p in tracked if p.endswith('.py') and
                               (p.startswith('src/') or p.startswith('scratch/dataset_refresh/'))}
    for name in sorted(selected):
        data = subprocess.check_output(['git', 'show', commit + ':' + name])
        dest = out / 'source_code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return commit


def freeze_arm(source, destination):
    """Keep all checkpoint values and physical-file hashes without a 20k-file Hub commit."""
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] == 'records':
            if path.suffix != '.json':
                raise ValueError('Unexpected non-JSON stage artifact: ' + str(relative))
            checkpoints.append({'path': relative.as_posix(), 'sha256': digest(path.read_bytes()),
                                'value': read_json(path)})
        else:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_rows(destination / 'record_checkpoints.jsonl', checkpoints)


def prepare(root, arm, source_commit, ledger_end, destination=None):
    root = Path(root).resolve()
    meta = read_json(root / 'run_meta.json')
    cfg = validate_arm(root, arm)
    if not cfg.get('per_row_regime'):
        raise ValueError('Only the current per-row production regime can be released here')
    from scratch.dataset_refresh.per_row import assert_models
    assert_models(cfg)
    rows = export(root, arm)  # Root's validator rechecks receipts, judges, repairs, identity and duplicates.
    selection = read_json(root / arm / 'selection.json')
    validate_export(rows, selection)
    if selection['dataset_sha256'] != digest((root / arm / 'dataset.jsonl').read_bytes()):
        raise ValueError('Export SHA differs from selection')
    budget, shared, calls = scoped_ledger(meta, root, arm, ledger_end)
    allowed = {m['model'] for m in cfg['models'].values()}
    if any(e['model'] not in allowed or 'haiku' in e['model'].lower() for e in calls):
        raise ValueError('A new run-scoped call used a forbidden or unconfigured model')
    phase = read_json(root / 'active_phase.json')
    stamp = re.sub(r'^(\d{4})-?(\d{2})-?(\d{2}).*$', r'\1-\2-\3', meta['created_at'])
    name = synth_name(cfg['pipeline'], date=stamp)
    out = Path(destination).resolve() if destination else root.parent / to_local(name)
    if out.exists():
        raise ValueError('Immutable publication snapshot already exists: ' + str(out))
    out.mkdir(parents=True)
    try:
        commit = freeze_code(out, source_commit, cfg)
        for filename in ('run.py', 'per_row.py'):
            actual = Path('scratch/dataset_refresh') / filename
            frozen = out / 'source_code/scratch/dataset_refresh' / filename
            if actual.read_bytes().replace(b'\r\n', b'\n') != frozen.read_bytes().replace(b'\r\n', b'\n'):
                raise ValueError('Commit differs from currently validating implementation: ' + filename)
            key = 'code_sha256' if filename == 'run.py' else 'per_row_code_sha256'
            if phase.get(key) != digest(actual.read_bytes()):
                raise ValueError('Active production phase used a different implementation: ' + filename)
        freeze_arm(root / arm, out / 'audit/arm')
        shutil.copy2(root / 'run_meta.json', out / 'run_meta.json')
        shutil.copy2(root / 'active_phase.json', out / 'audit/active_phase.json')
        if (root / 'phases').exists():
            shutil.copytree(root / 'phases', out / 'audit/phases')
        for filename in ('dataset.jsonl', 'selection.json', 'config.json'):
            shutil.copy2(root / arm / filename, out / filename)
        write_json(out / 'audit/budget/spend.json', calls)
        write_json(out / 'audit/budget/shared_budget_snapshot.json', shared)
        raw_calls = []
        for call in calls:
            filename = f'{call["call_id"]:06d}.json'
            source = budget / 'raw_calls' / filename
            value = read_json(source)
            safe_audit(value)
            raw_calls.append({'call_id': call['call_id'], 'original_filename': filename,
                              'sha256': digest(source.read_bytes()), 'value': value})
        write_rows(out / 'audit/budget/raw_calls.jsonl', raw_calls)
        results = [load_result(p) for p in sorted((root / arm / 'records').glob('*/result.json'))]
        statuses = dict(Counter(r['status'] for r in results))
        repair_counts = dict(Counter(str(r['metadata'].get('response_repair_count', 0)) for r in rows))
        review_summary = {'candidate_statuses': statuses, 'selected_repair_counts': repair_counts,
                          'selected': len(rows), 'completed_candidates': len(results)}
        write_json(out / 'review_summary.json', review_summary)
        fields = {
            'experiment': f'{cfg["pipeline"]}: 716 selected synthetic human-advice conversations',
            'date_generated': meta['created_at'],
            'constitution': cfg['constitution'] + '; role=' + cfg.get('constitution_role', 'generation_and_review') +
                            '; SHA256=' + cfg['constitution_sha256'],
            'source_repo': 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ ' + commit,
            'models': json.dumps(cfg['models'], ensure_ascii=False) + '. All new author roles use Sonnet5; no Haiku in any new role. The mapping above identifies every reviewer separately. Historical inspiration can contain older authors; no historic answers enter the selected corpus. API model IDs are mutable, not immutable weight identities.',
            'generation_config': 'Frozen config.json contains exact models, prompts, settings, checks and quotas. Per-row eligibility, detailed draft/rewrite, independent content and grounding checks, bounded repairs, then fresh checks. ' + json.dumps(review_summary),
            'schema': 'dataset.jsonl: messages=[system,user,assistant]; assistant includes content and reasoning_content. metadata contains mechanically derived conversation/source lineage facts. selection.json pins ordered 716 scenario IDs and dataset SHA256. Only dataset.jsonl is the default training split; audit/ contains all complete, rejected and repaired stage evidence.',
            'provenance': f'uv run python scratch/dataset_refresh/run.py production --root {root.as_posix()}; validated export and immutable snapshot prepared by scratch/dataset_refresh/publish_completed.py. Source pins: ' + json.dumps(meta['arms'][arm]['source'], ensure_ascii=False),
            'lineage': 'Source rows are imperfect scenario inspiration, not answer truth. Fresh variants are explicitly unpaired; source hashes/IDs establish ancestry, not exact counterfactual equivalence. Source facts in release metadata are extracted mechanically from actual conversation text. Raw author metadata is retained in audit stages and is not certified factual truth. Review labels are model judgments, not human annotation or guarantees.',
            'review_limitations': 'Content and narrowly focused grounding reviewers can disagree. Rejected judgments, exact evidence, author attempts and repairs remain in audit/arm/record_checkpoints.jsonl; each entry preserves original path, physical-file SHA256 and parsed JSON value, including checkpoint receipts. API evidence is similarly flattened in audit/budget/raw_calls.jsonl. Acceptance means the final candidate passed all configured checks. Repair may change the answer but cannot silently change the user facts. No training or evaluation was performed by this dataset workflow.',
            'budget': f'{len(calls)} run-and-arm-scoped physical API calls; provider-reported USD {sum(e.get("api_reported_cost_usd") or 0 for e in calls):.6f}; charged/reserved USD {sum(e["charged_or_reserved_usd"] for e in calls):.6f}. Shared ledger snapshot includes previous pilots/calibration and both arms through exclusive index {ledger_end}; raw requests/responses include only this run and arm, no transport headers or credentials.',
            'mixing': 'This artifact contains synthetic rows only. The separate planned mixture combines these716 rows with the common9284-row replay selection from the pinned nosynth source: ' + json.dumps(meta.get('base', {}), ensure_ascii=False),
        }
        if cfg.get('craft_spec'):
            fields['craft_preference'] = cfg['craft_spec'] + '; SHA256=' + cfg['craft_spec_sha256'] + '. This craft preference guides generation. Ethical constitution is a separate compatibility check and is not injected into nonmoral author prompts.'
        front = {'tags': training_data_tags('synth', cfg['pipeline'], cfg['constitution']),
                 'configs': [{'config_name': 'default', 'default': True, 'data_files': 'dataset.jsonl'}]}
        write_json(out / 'card_fields.json', fields)
        write_json(out / 'card_front_matter.json', front)
        (out / 'README.md').write_text(card_markdown(fields, front), encoding='utf-8')
        write_json(out / 'publication_manifest.json', {
            'name': name, 'source_commit': commit, 'run_root': str(root), 'arm': arm,
            'ledger_end_exclusive': ledger_end, 'dataset_sha256': selection['dataset_sha256'],
            'selection_sha256': digest((out / 'selection.json').read_bytes()),
            'files': {p.relative_to(out).as_posix(): digest(p.read_bytes()) for p in out.rglob('*') if p.is_file()}})
    except Exception:
        # Keep incomplete evidence explicit and refuse push; never delete another run's files.
        (out / 'PREPARATION_FAILED').write_text('Snapshot preparation failed; do not publish.', encoding='utf-8')
        raise
    return str(out)


def push(snapshot):
    out = Path(snapshot)
    if (out / 'PREPARATION_FAILED').exists():
        raise ValueError('Incomplete snapshot cannot be published')
    manifest = read_json(out / 'publication_manifest.json')
    actual = {p.relative_to(out).as_posix(): digest(p.read_bytes()) for p in out.rglob('*')
              if p.is_file() and p.name != 'publication_manifest.json'}
    if actual != manifest['files']:
        raise ValueError('Snapshot changed after preparation')
    validate_export(read_rows(out / 'dataset.jsonl'), read_json(out / 'selection.json'))
    return push_run_dir(out, manifest['name'], read_json(out / 'card_fields.json'), private=False,
                        front_matter=read_json(out / 'card_front_matter.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--root', required=True)
    prep.add_argument('--arm', required=True)
    prep.add_argument('--source-commit', required=True)
    prep.add_argument('--ledger-end', required=True, type=int)
    prep.add_argument('--destination')
    publication = sub.add_parser('push')
    publication.add_argument('--snapshot', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    print(prepare(**args) if command == 'prepare' else push(**args))
