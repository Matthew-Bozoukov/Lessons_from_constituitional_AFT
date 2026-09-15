# ABOUTME: Prepare a closed byte-preserving research audit archive without exporting a train split or uploading.
# ABOUTME: Binds immutable source inventory, all shared physical calls, billing evidence and committed code.
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
from filelock import FileLock
from scratch.dataset_refresh import run as runtime, publish_completed as publication
from scratch.dataset_refresh.billing_evidence import validate_billing_call
from src.infra.huggingface import card_markdown
from src.naming import artifact_name


def inventory(root, exclusions=()):
    root = Path(root).absolute()
    if any(p.is_symlink() or p.is_junction() for p in (root, *root.parents)):
        raise ValueError('Source traverses a link')
    root = root.resolve(strict=True)
    result = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(rel == Path(e) or Path(e) in rel.parents for e in exclusions):
            continue
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
            raise ValueError('Archive source contains an unsafe link')
        if path.is_file():
            if path.suffix == '.lock':
                if path.stat().st_size > 1:
                    raise ValueError('Nontrivial lock file cannot be silently omitted')
                continue  # Windows holds the one-byte OS locking region unreadable.
            if path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example':
                raise ValueError('Credential file is not an audit artifact')
            result[rel.as_posix()] = runtime.digest(path.read_bytes())
    return result


def archive_tree(root, destination, exclusions=()):
    root, destination = Path(root), Path(destination)
    before = inventory(root, exclusions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, 'w:gz') as archive:
        for name, sha in before.items():
            data = (root / name).read_bytes()
            if runtime.digest(data) != sha:
                raise ValueError('Source changed before archive: ' + name)
            if name.endswith('.json'):
                publication.safe_audit(json.loads(data))
            elif name.endswith('.jsonl'):
                for line in data.splitlines():
                    if line.strip(): publication.safe_audit(json.loads(line))
            entry = tarfile.TarInfo(name)
            entry.size, entry.mtime = len(data), 0
            archive.addfile(entry, io.BytesIO(data))
    if before != inventory(root, exclusions):
        raise ValueError('Source changed during archive')
    return {'source': str(root.resolve()), 'archive': destination.name,
            'archive_sha256': runtime.digest(destination.read_bytes()), 'exclusions': list(exclusions), 'files': before,
            'operational_lock_markers': [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size}
                                         for p in root.rglob('*.lock') if p.is_file() and p.stat().st_size <= 1]}


def freeze_code(out, commit):
    resolved = subprocess.check_output(['git', 'rev-parse', commit + '^{commit}'], text=True).strip()
    if commit != resolved or len(commit) != 40:
        raise ValueError('Source commit must be the full commit SHA')
    changed = subprocess.check_output(['git', 'diff', '--name-only', commit, '--',
        'src', 'scratch/dataset_refresh', 'configs', 'constitutions', 'preferences', 'pyproject.toml', 'uv.lock'], text=True)
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', 'scratch/dataset_refresh'], text=True)
    if changed.strip() or any(p.endswith('.py') for p in untracked.splitlines()):
        raise ValueError('Commit all generating/validating helpers and recipes before preparation')
    path = out / 'source_code.tar.gz'
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output', str(path), commit], check=True)
    return {'commit': commit, 'archive': path.name, 'sha256': runtime.digest(path.read_bytes())}


def prepare(manifest_path, destination, commit, ledger_end, date):
    manifest_path, out = Path(manifest_path).resolve(), Path(destination).resolve()
    manifest_bytes = manifest_path.read_bytes(); manifest = json.loads(manifest_bytes)
    if not isinstance(manifest.get('final_summary'), str) or not manifest['final_summary'].strip():
        raise ValueError('Root must supply readable final counts, shortages and review limits before preparation')
    for source in manifest['root_directories'] + manifest['supporting_evidence_directories'] + [manifest['shared_budget_root']]:
        p = Path(source).absolute()
        if any(x.is_symlink() or x.is_junction() for x in (p, *p.parents)):
            raise ValueError('Source manifest traverses a link')
    roots = [Path(p).resolve() for p in manifest['root_directories']]
    evidence = [Path(p).resolve() for p in manifest['supporting_evidence_directories']]
    budget = Path(manifest['shared_budget_root']).resolve()
    sources = roots + evidence + [budget]
    if len(set(roots)) != len(roots) or out.exists() or any(out.is_relative_to(p) for p in sources):
        raise ValueError('Use distinct roots and a new destination outside all sources')
    if type(ledger_end) is not int or ledger_end < 0:
        raise ValueError('Explicit closed ledger cutoff required')
    name = artifact_name('dataset-refresh-incomplete-audit', date=date)
    with ExitStack() as locks:
        for root in sorted(roots): locks.enter_context(FileLock(str(root / 'execution.lock'), timeout=1))
        locks.enter_context(FileLock(str(budget / 'spend.lock'), timeout=1))
        ledger = publication.read_json(budget / 'spend.json')
        if len(ledger) != ledger_end or any(e['status'] == 'reserved' for e in ledger):
            raise ValueError('Ledger cutoff is not closed or calls remain reserved')
        amounts = [e['charged_or_reserved_usd'] for e in ledger]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in amounts) or sum(amounts) > 250:
            raise ValueError('Invalid accounting or hard250 exceeded')
        cache = {}
        for n, call in enumerate(ledger):
            raw = publication.read_json(budget / 'raw_calls' / f'{n:06d}.json')
            if call['call_id'] != n or runtime.digest(raw['request']) != call['request_sha256']:
                raise ValueError('Raw physical call missing or request binding changed')
            validate_billing_call(budget, call, raw, cache)
        out.mkdir(parents=True)
        try:
            code = freeze_code(out, commit)
            inventories, index, recipes = [], [], []
            for number, root in enumerate(roots):
                meta = publication.read_json(root / 'run_meta.json')
                receipt_state = 'absent_in_historical_phase'
                if (root / 'run_meta.receipt.json').exists():
                    runtime.load_checkpoint(root / 'run_meta.json')
                    receipt_state = 'verified_present'
                exclusions = manifest.get('root_archive_exclusions', {}).get(str(root), [])
                if budget.is_relative_to(root) and str(budget.relative_to(root)).replace('\\', '/') not in exclusions:
                    raise ValueError('Nested shared budget must be explicitly excluded from origin archive')
                for arm, spec in meta['arms'].items():
                    cfg = publication.read_json(root / arm / 'config.json')
                    if runtime.digest(cfg) != spec['config_sha256']:
                        raise ValueError('Origin frozen config hash differs')
                    statuses, traits = Counter(), Counter()
                    for path in (root / arm / 'records').glob('*/result.json'):
                        result = runtime.load_result(path); statuses[result['status']] += 1
                        if result['status'] == 'accepted': traits[result.get('trait_id', 'missing')] += 1
                    index.append({'phase': number, 'root': str(root), 'arm': arm, 'config_sha256': runtime.digest(cfg),
                                  'run_meta_receipt': receipt_state,
                                  'effective_terminal_counts': dict(statuses), 'accepted_by_trait': dict(traits),
                                  'interpretation': 'Phase-local effective terminals; not a certified final composite selection.'})
                    recipes.append({'root': str(root), 'arm': arm, 'config': cfg, 'run_meta': meta})
                inventories.append(archive_tree(root, out / 'archives' / f'origin_{number:02d}.tar.gz', exclusions))
            inventories.append(archive_tree(budget, out / 'archives/shared_budget.tar.gz'))
            for number, source in enumerate(evidence):
                inventories.append(archive_tree(source, out / 'archives' / f'evidence_{number:02d}.tar.gz'))
            for archived in inventories:
                if inventory(archived['source'], archived['exclusions']) != archived['files']:
                    raise ValueError('Source changed across the closed snapshot')
            (out / 'source_manifest.json').write_bytes(manifest_bytes)
            runtime.write_json(out / 'archive_inventory.json', inventories)
            runtime.write_rows(out / 'audit_index.jsonl', index)
            readiness = {'train_ready': False, 'mixture_ready': False, 'target_rows_per_arm': 716,
                'final_selection_not_certified': True, 'phase_census': index,
                'owner_final_summary': manifest.get('final_summary', 'Not supplied; phase counts are not final accepted quotas.'),
                'ledger_end': ledger_end, 'shared_exposure_usd': sum(amounts),
                'api_reported_usd': sum(e.get('api_reported_cost_usd') or 0 for e in ledger),
                'active_reservations': 0, 'uncertain_failures': sum(e['status'] == 'uncertain_failure' for e in ledger),
                'billing_resolved_failed_outputs': sum(e['status'] == 'billing_verified_failure' for e in ledger),
                'historical_status_note': 'The frozen runner labels every non-settled status unsettled. Its archived status counts therefore include billing_verified_failure even though their billing is resolved; content remains failed. Use the explicit current counts here.',
                'ledger_status_counts': dict(Counter(e['status'] for e in ledger)),
                'accounting_policy': 'Any unknown failed cost retains its reservation; actual unknown count is uncertain_failures above.'}
            runtime.write_json(out / 'readiness.json', readiness)
            runtime.write_json(out / 'phase_recipes.json', recipes)
            fields = {'title': 'INCOMPLETE RESEARCH AUDIT — NOT A TRAINING DATASET',
                'experiment': manifest['final_summary'] + ' Preserve all refresh phases and unsuccessful work without lowering release gates.',
                'date_generated': date, 'constitution': json.dumps([{'arm': r['arm'], 'path': r['config'].get('constitution'), 'sha256': r['config'].get('constitution_sha256'), 'craft_sha256': r['config'].get('craft_spec_sha256')} for r in recipes]),
                'source_repo': 'Repository source snapshot @ ' + commit + '; historical phase hashes remain in original metadata.',
                'models': json.dumps([{'root': r['root'], 'arm': r['arm'], 'models': r['config']['models']} for r in recipes]) + '; historical pilot recipes included; not all archived calls use the final recipe.',
                'generation_config': 'phase_recipes.json and byte-preserved origin archives contain exact resolved prompts, sampling settings, source pins and immutable execution/recovery manifests. Shared hard budget250. No new generation during preservation.',
                'schema': 'Only audit_index.jsonl is exposed, as audit split. No train split, top-level dataset.jsonl or mixture.jsonl. Byte-preserving .tar.gz files contain original evidence with original statuses.',
                'provenance': f'python -m scratch.dataset_refresh.preserve_partial --manifest {manifest_path} --destination {out} --source-commit {commit} --ledger-end {ledger_end} --date {date}. Reconstruct original relative paths using source_manifest.json and archive_inventory.json; verify all SHA256 hashes before extraction. No training/evaluation performed.',
                'closed_budget': f'{ledger_end} physical calls; conservative charged exposure ${sum(amounts):.7f}; API-reported charges ${readiness["api_reported_usd"]:.7f}; 0 active reservations; {readiness["uncertain_failures"]} uncertain failures; {readiness["billing_resolved_failed_outputs"]} billing-resolved failures whose content remains failed.',
                'intended_replay': 'dougalldeepmind/2026-09-08-nosynth-mix @ 7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd, mixture.jsonl. Intended final mixture:716 synthetic +9284 identical replay rows. No completed mixture is represented by this archive.',
                'known_quality_limits': 'Historical pilots, superseded recipes, failed outputs, quality rejections, independent holds and incomplete checkpoints are preserved as evidence. Effective accepted phase counts do not certify independent clearance or a final716 selection. Provisional audits retain their original input hashes; undetected defects remain possible. No mixture, training or evaluation is claimed.'}
            front = {'configs': [{'config_name': 'audit', 'default': True, 'data_files': [{'split': 'audit', 'path': 'audit_index.jsonl'}]}], 'tags': ['research-audit', 'incomplete', 'not-training-ready']}
            runtime.write_json(out / 'card_fields.json', fields); runtime.write_json(out / 'card_front_matter.json', front)
            (out / 'README.md').write_text(card_markdown(fields, front), encoding='utf-8')
            runtime.write_json(out / 'preparation_receipt.json', {'name': name, 'source_code': code, 'source_manifest_sha256': runtime.digest(manifest_bytes),
                'ledger_sha256': runtime.digest((budget / 'spend.json').read_bytes()), 'ledger_end': ledger_end,
                'helper_sha256': runtime.digest(Path(__file__).read_bytes()), 'publication_authorized': False})
            runtime.write_json(out / 'files_manifest.json', inventory(out))
            return out
        except BaseException:
            (out / 'PREPARATION_FAILED').write_text('Incomplete preservation; do not publish.', encoding='utf-8')
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'destination', 'source-commit', 'date'): p.add_argument('--'+flag, required=True)
    p.add_argument('--ledger-end', required=True, type=int)
    args = p.parse_args()
    print(prepare(args.manifest, args.destination, args.source_commit, args.ledger_end, args.date))
