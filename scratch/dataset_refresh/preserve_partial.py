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


def prepare_corrections(manifest_path, destination, commit, date):
    """Archive changed row directories and new evidence, referencing the immutable parent audit."""
    manifest_path, out = Path(manifest_path).resolve(), Path(destination).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    parent = manifest['parent_archive']
    if (parent['repo'] != 'dougalldeepmind/2026-09-15-dataset-refresh-incomplete-audit'
            or parent['revision'] != 'f455cc9a2224d65c3861fd83a7f57c4c49c8072a'):
        raise ValueError('Correction archive requires the verified immutable parent revision')
    budget = Path(manifest['budget_root']).resolve()
    ledger = budget / 'spend.json'
    if runtime.digest(ledger.read_bytes()) != manifest['ledger_sha256']:
        raise ValueError('Budget changed during a zero-call correction')
    roots = sorted({str(Path(e['root']).resolve()) for e in manifest['rows']})
    evidence = Path(manifest['evidence_root']).resolve()
    if (out.exists() or out.is_relative_to(evidence)
            or any(out.is_relative_to(Path(root)) for root in roots)):
        raise ValueError('Use a new publication directory outside evidence and origins')
    validated_evidence_inventory = inventory(evidence)
    selected_conversations = []
    for arm, count in manifest['readiness']['selected_counts'].items():
        audit_dir = Path(manifest['selection_audits'][arm]).resolve()
        if not audit_dir.is_relative_to(evidence):
            raise ValueError('Selection audit must be included in correction evidence')
        audit = json.loads((audit_dir / 'analysis_readiness.json').read_text(encoding='utf-8'))
        conversations = audit_dir / 'conversations_for_audit.jsonl'
        rows = runtime.read_rows(conversations)
        if (audit['analysis_rows'] != count or audit['token_failures']
                or len(rows) != count
                or runtime.digest(conversations.read_bytes()) != audit['input_sha256']):
            raise ValueError('Readiness count or exact token-audited selection differs')
        selected_conversations.extend(rows)
    seen, inputs = set(), []
    with ExitStack() as locks:
        for root in roots:
            locks.enter_context(FileLock(str(Path(root) / 'execution.lock'), timeout=1))
        for selected in selected_conversations:
            origin = selected['metadata']['origin']
            selected_root = Path(origin['root']).resolve()
            if str(selected_root) not in roots:
                raise ValueError('Selected origin is not locked')
            path = selected_root / origin['arm'] / 'records' / origin['candidate_id'] / 'result.json'
            result = runtime.load_result(path)
            if (runtime.digest(path.read_bytes()) != origin['result_sha256']
                    or result['status'] != 'accepted'):
                raise ValueError('Audited selected origin changed or is now excluded')
            record = result['record']
            if selected['messages'] != [
                {'role': 'system', 'content': record['system']},
                {'role': 'user', 'content': record['user']},
                {'role': 'assistant', 'content': record['response'], 'reasoning_content': record['reasoning']},
            ]:
                raise ValueError('Selected conversation differs from bound accepted author text')
        configs = {}
        for entry in manifest['rows']:
            root = Path(entry['root']).resolve()
            arm, cid = entry['arm'], entry['candidate_id']
            import re
            if arm not in {'da-lowstakes-refresh', 'nonmoral-advice'} or not re.fullmatch(r't[1-9]_\d+_v\d+', cid):
                raise ValueError('Invalid correction origin')
            identity = (str(root), arm, cid)
            if identity in seen:
                raise ValueError('Duplicate correction origin')
            seen.add(identity)
            if (str(root), arm) not in configs:
                configs[(str(root), arm)] = runtime.validate_arm(root, arm)
            row = root / arm / 'records' / cid
            if runtime.digest((row / 'result.json').read_bytes()) != entry['result_sha256']:
                raise ValueError('Correction terminal changed')
            inputs.append((entry, row))
        # Freeze all inputs before copying; later mutations fail preparation.
        source_inventories = {str(row): inventory(row) for _, row in inputs}
        evidence_inventory = inventory(evidence)
        if evidence_inventory != validated_evidence_inventory:
            raise ValueError('Evidence changed after selection validation')
        out.mkdir(parents=True)
        try:
            code = freeze_code(out, commit)
            (out / 'source_manifest.json').write_bytes(manifest_bytes)
            archives = []
            for index, (entry, row) in enumerate(inputs):
                archived = archive_tree(row, out / 'rows' / f'row_{index:03d}.tar.gz')
                archived['origin'] = entry
                archived['archive'] = 'rows/' + archived['archive']
                archives.append(archived)
            supporting = archive_tree(evidence, out / 'correction_evidence.tar.gz')
            if (inventory(evidence) != evidence_inventory
                    or any(inventory(row) != source_inventories[str(row)] for _, row in inputs)
                    or runtime.digest(ledger.read_bytes()) != manifest['ledger_sha256']
                    or manifest_path.read_bytes() != manifest_bytes):
                raise ValueError('Correction inputs changed during preservation')
            runtime.write_json(out / 'archive_inventory.json', {'rows': archives, 'evidence': supporting})
            runtime.write_json(out / 'readiness.json', manifest['readiness'])
            runtime.write_rows(out / 'audit_index.jsonl', [{'arm': arm, 'selected_rows': n,
                'target_rows': 716, 'training_ready': False} for arm, n in manifest['readiness']['selected_counts'].items()])
            fields = {'title': 'Dataset refresh correction audit; not a training release',
                'experiment': manifest['summary'], 'date_generated': date,
                'constitution': 'constitutions/claude_distilled_09_principles/constitution.md; original exact constitution/craft hashes in parent phase metadata and unchanged origin configs.',
                'source_repo': 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ ' + commit,
                'models': 'Original Sonnet-authored conversations; independent Codex reassessment. No new API inference, training or evaluation during correction.',
                'generation_config': 'source_manifest.json binds exact corrected row directories, evidence and unchanged budget. Full original recipes and physical calls remain in the pinned parent audit.',
                'schema': 'Only audit_index.jsonl is a default audit split. Row archives contain checkpoints, original failures/exclusions and new bound corrections. Selected conversation files inside the evidence archive are research candidates, not a train split.',
                'provenance': f'uv run --no-sync python -m scratch.dataset_refresh.preserve_partial --corrections --manifest {manifest_path} --destination {out} --source-commit {commit} --date {date}',
                'parent_archive': parent['repo'] + '@' + parent['revision'],
                'reconstruction': 'Start with the pinned parent archive. For each archive_inventory.rows entry, its full row archive supersedes the entire corresponding origin row directory (including the absence of old active exclusion sidecars). All superseded exclusions/failures remain inside historical subdirectories. Map absolute original roots using the parent source_manifest and origin archive inventory. Verify every uncompressed-file SHA256 before use. The evidence archive contains the exact independent and root decisions, rejected recovery proposals, successive selections, native mask checks and completion proposal.',
                'limits': 'Acceptance and audit decisions remain fallible. Repeated scenario families and length differences persist. No completed two-arm716+9284 mixtures or training/evaluation claimed. The parent revision remains immutable.'}
            front = {'configs': [{'config_name': 'audit', 'default': True,
                                  'data_files': [{'split': 'audit', 'path': 'audit_index.jsonl'}]}],
                     'tags': ['research-audit', 'dataset-correction', 'not-training-ready']}
            runtime.write_json(out / 'card_fields.json', fields)
            runtime.write_json(out / 'card_front_matter.json', front)
            (out / 'README.md').write_text(card_markdown(fields, front), encoding='utf-8')
            runtime.write_json(out / 'preparation_receipt.json', {
                'name': artifact_name('dataset-refresh-correction-audit', date=date),
                'source_code': code, 'source_manifest_sha256': runtime.digest(manifest_bytes),
                'ledger_sha256': manifest['ledger_sha256'], 'inference_calls': 0,
                'helper_sha256': runtime.digest(Path(__file__).read_bytes())})
            runtime.write_json(out / 'files_manifest.json', inventory(out))
        except BaseException:
            (out / 'PREPARATION_FAILED').write_text('Incomplete correction archive; do not publish.', encoding='utf-8')
            raise
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'destination', 'source-commit', 'date'): p.add_argument('--'+flag, required=True)
    p.add_argument('--ledger-end', type=int)
    p.add_argument('--corrections', action='store_true')
    args = p.parse_args()
    if args.corrections:
        print(prepare_corrections(args.manifest, args.destination, args.source_commit, args.date))
    else:
        if args.ledger_end is None: p.error('--ledger-end required for full preservation')
        print(prepare(args.manifest, args.destination, args.source_commit, args.ledger_end, args.date))
