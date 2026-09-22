# ABOUTME: Validate an explicit716-row selection across immutable recipe phases and freeze a composite corpus.
# ABOUTME: Preserves each origin's prompts, identities, reviews and scoped calls; never generates or selects rows.
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import shutil

from filelock import FileLock

from scratch.dataset_refresh import run as runtime
from scratch.dataset_refresh import per_row
from scratch.dataset_refresh.billing_evidence import validate_billing_call
from scratch.dataset_refresh import publish_completed as publication
from src.infra.huggingface import card_markdown, training_data_tags
from src.naming import synth_name

ARMS = {'da-lowstakes-refresh', 'nonmoral-advice'}
TEXT_KEYS = {'system', 'user', 'reasoning', 'response', 'draft_reasoning', 'draft_response', 'rewrite_changes'}


def freeze_quality(source, destination, dataset_sha256):
    """Copy a caller-frozen quality bundle; no audit or acceptance is invented here."""
    source = Path(source).absolute()
    if any(p.is_symlink() or p.is_junction() for p in (source, *source.parents)):
        raise ValueError('Quality path must not traverse a symlink/junction')
    source = source.resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError('Quality destination must be outside its source')

    def inventory():
        files = {}
        for path in source.rglob('*'):
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(source):
                raise ValueError('Quality artifact escapes its directory or uses a link')
            if path.is_file():
                files[path.relative_to(source).as_posix()] = runtime.digest(path.read_bytes())
        return files

    before = inventory()
    manifest = publication.read_json(source / 'quality_manifest.json')
    if manifest.get('dataset_sha256') != dataset_sha256:
        raise ValueError('Quality bundle describes a different selected dataset')
    if manifest.get('files') != {k: v for k, v in before.items() if k != 'quality_manifest.json'}:
        raise ValueError('Quality manifest must hash every supplied artifact exactly')
    for name in before:
        path = source / name
        if path.suffix == '.json':
            value = publication.read_json(path)
            publication.safe_audit(value)
            if isinstance(value, dict):
                for key in ('input_sha256', 'dataset_sha256', 'selected_dataset_sha256'):
                    if key in value and value[key] != dataset_sha256:
                        raise ValueError('Quality audit input differs from selected dataset: ' + name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if runtime.digest(target.read_bytes()) != before[name]:
            raise ValueError('Quality artifact changed during copy: ' + name)
    if inventory() != before:
        raise ValueError('Quality artifacts changed during copy')
    return {'dataset_sha256': dataset_sha256, 'files': before}


def validate_short_draft_adoption(path, result, cfg):
    from scratch.dataset_refresh import recover_short_draft as recovery
    row, root = path.parent, path.parent.parents[2]
    archive = row / 'recovered_failures/short_draft_recovery_200'
    original = runtime.load_checkpoint(archive / 'result.json')
    manifest_path = row / recovery.MANIFEST
    manifest = runtime.load_checkpoint(manifest_path)
    binding = manifest['binding']
    candidate_path = row / recovery.PROPOSAL
    candidate = runtime.load_checkpoint(candidate_path)
    adoption = runtime.load_checkpoint(row / 'short_draft_adoption_200.json')
    source_sha = runtime.digest((archive / 'result.json').read_bytes())
    if (candidate != result or original.get('status') != 'failed' or original.get('error_type') != 'ValueError'
            or original['error'] != 'Response lint: ' + '; '.join(binding['original_lint'])
            or not binding['original_lint'] or any(not re.fullmatch(r'<(?:reasoning|response)> is \d+ chars, under the 700 minimum', s) for s in binding['original_lint'])
            or {binding['source_result_sha256'], result['source_result_sha256'], adoption['source_result_sha256']} != {source_sha}
            or adoption['candidate_sha256'] != runtime.digest(candidate_path.read_bytes())
            or {result['short_draft_recovery_manifest_sha256'], adoption['manifest_sha256']} != {runtime.digest(manifest_path.read_bytes())}
            or not adoption.get('independent_audit_reason', '').strip()
            or adoption['source_review'] != binding['source_review']
            or binding['source_review'].get('eligible') is not True or binding['source_review']['result_sha256'] != source_sha
            or manifest['implementation_files'] != recovery.implementation_hashes()
            or manifest['appended_recovery_instruction'] != recovery.RECOVERY_INSTRUCTION
            or manifest['appended_recovery_instruction_sha256'] != runtime.digest(recovery.RECOVERY_INSTRUCTION)
            or result['author_stage_mapping'] != manifest['author_stage_mapping']
            or per_row.conversation(result['record'], False) != per_row.conversation(original['record'], False)):
        raise ValueError('Short-draft recovery does not bind failed input, explicit instruction and independent adoption')
    for name, sha in binding['bound_file_sha256'].items():
        source = Path(name)
        if source in (row / 'result.json', row / 'result.receipt.json'):
            source = archive / source.name
        if runtime.digest(source.read_bytes()) != sha:
            raise ValueError('Short-draft bound input changed: ' + name)
    stage = cfg['response_stages'][-1]
    saved = runtime.load_checkpoint(row / (recovery.AUTHOR + '.json'))
    if (stage['lint'].get('min_chars') != 700 or any(result['record'].get(k) != v for k, v in saved.items())
            or runtime.lint_problems({tag: saved[dest] for dest, tag in stage['save'].items()}, stage['lint'], result['record'])):
        raise ValueError('Short-draft final differs from actual author or fails unchanged final lint')
    requests = recovery.expected_recovery_requests(original['record'], result['record'], cfg)
    for name in (recovery.AUTHOR, 'grounding_200', 'review_200'):
        physical = row / (name + '.physical.json')
        value = runtime.load_checkpoint(row / (name + '.json'))
        if (result['physical_receipt_sha256'][name] != runtime.digest(physical.read_bytes())
                or runtime.load_checkpoint(physical) != recovery.physical_receipt(root, row, name, value, requests[name], stage if name == recovery.AUTHOR else None)):
            raise ValueError('Short-draft physical author/review chain changed')


def validate_adoption(path, result, cfg=None):
    from scratch.dataset_refresh import recover_fenced_review
    if (recover_fenced_review.MARKER in result
            or (path.parent / recover_fenced_review.HISTORY).exists()):
        recover_fenced_review.verify_history(path, result, cfg)
    if result.get('accepted_attempt') == 200 and 'short_draft_recovery_manifest_sha256' in result:
        return validate_short_draft_adoption(path, result, cfg)
    if result.get('accepted_attempt', 0) < 100:
        return
    number = result['accepted_attempt']
    row = path.parent
    manifest_path = row / f'independent_repair_{number}.json'
    manifest = runtime.load_checkpoint(manifest_path)
    candidate_path = row / f'independent_candidate_{number}.json'
    candidate = runtime.load_checkpoint(candidate_path)
    adoption = runtime.load_checkpoint(row / f'independent_adoption_{number}.json')
    archive = row / 'recovered_failures' / f'independent_repair_{number}'
    original = runtime.load_checkpoint(archive / 'result.json')
    exclusion = runtime.load_checkpoint(archive / 'independent_exclusion.json')
    old_sha = runtime.digest((archive / 'result.json').read_bytes())
    if (candidate != result or adoption['candidate_sha256'] != runtime.digest(candidate_path.read_bytes())
            or result['independent_repair_manifest_sha256'] != runtime.digest(manifest_path.read_bytes())
            or not adoption.get('independent_audit_reason', '').strip()
            or {result['source_result_sha256'], manifest['source_result_sha256'],
                adoption['source_result_sha256'], exclusion['result_sha256']} != {old_sha}
            or manifest['exclusion'] != exclusion or adoption['source_exclusion'] != exclusion
            or per_row.conversation(result['record'], False) != per_row.conversation(original['record'], False)):
        raise ValueError('Independent adoption provenance does not bind the original and reviewed repair')
    if manifest['script_sha256'] != runtime.digest(Path(__file__).with_name('repair_independent.py').read_bytes()):
        raise ValueError('Independent repair implementation differs from the archived adoption')


def freeze_origin_arm(source, destination):
    """Archive all JSON evidence and the hashes of empty operational lock files."""
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints, locks = [], []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] == 'records':
            if path.suffix == '.lock' and not path.read_bytes():
                locks.append({'path': relative.as_posix(), 'sha256': runtime.digest(b''), 'bytes': 0})
                continue
            if path.suffix != '.json':
                raise ValueError('Unexpected non-JSON/nonempty-lock evidence: ' + str(relative))
            value = publication.read_json(path)
            publication.safe_audit(value)
            checkpoints.append({'path': relative.as_posix(), 'sha256': runtime.digest(path.read_bytes()), 'value': value})
        else:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    runtime.write_rows(destination / 'record_checkpoints.jsonl', checkpoints)
    runtime.write_json(destination / 'operational_lock_files.json', locks)


def validate_selection(selection):
    if not isinstance(selection, dict) or selection.get('arm') not in ARMS:
        raise ValueError('Selection must declare one supported synthetic arm')
    entries = selection.get('entries')
    if not isinstance(entries, list) or len(entries) != 716:
        raise ValueError('Explicit selection must contain exactly716 entries')
    phases, rows, seen, signatures, normalized = {}, [], set(), set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'root', 'arm', 'candidate_id', 'result_sha256'}:
            raise ValueError('Each selection entry must explicitly bind root, arm, candidate_id and result_sha256')
        root, arm, cid = Path(entry['root']).resolve(), entry['arm'], entry['candidate_id']
        if arm != selection['arm'] or not re.fullmatch(r't[1-9]_\d+_v\d+', cid):
            raise ValueError('Mixed arms or invalid candidate path')
        if not isinstance(entry['result_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['result_sha256']):
            raise ValueError('Each result requires its exact SHA256')
        key = (str(root), arm)
        identity_key = (*key, cid)
        if identity_key in seen:
            raise ValueError('Duplicate selected origin candidate')
        seen.add(identity_key)
        if key not in phases:
            cfg = runtime.validate_arm(root, arm)
            if cfg.get('per_row_regime') is not True or cfg['pipeline'] != arm:
                raise ValueError('Selected origin must be the matching frozen per-row pipeline')
            per_row.assert_models(cfg)
            meta = runtime.load_checkpoint(root / 'run_meta.json')
            phase = publication.read_json(root / 'active_phase.json')
            matching = [p for p in (root / 'phases').glob('*.json') if not p.name.endswith('.receipt.json')
                        and runtime.load_checkpoint(p) == phase]
            if len(matching) != 1:
                raise ValueError('Active phase must match exactly one immutable execution-phase receipt')
            for name, field in [('run.py', 'code_sha256'), ('per_row.py', 'per_row_code_sha256')]:
                if phase.get(field) != runtime.digest(Path(__file__).with_name(name).read_bytes()):
                    raise ValueError('Origin phase used different core implementation: ' + name)
            candidates = runtime.read_rows(root / arm / 'candidates.jsonl')
            by_id = {c['candidate_id']: c for c in candidates}
            if len(by_id) != len(candidates):
                raise ValueError('Duplicate candidate IDs in frozen source phase')
            phase_id = 'phase_' + runtime.digest({'root_meta_sha256': runtime.digest((root / 'run_meta.json').read_bytes()),
                                                'config_sha256': runtime.digest(cfg)})[:16]
            if phase_id in {p['phase_id'] for p in phases.values()}:
                raise ValueError('Phase namespace collision; do not silently merge origins')
            phases[key] = {'root': str(root), 'arm': arm, 'phase_id': phase_id, 'config': cfg,
                           'config_sha256': runtime.digest(cfg), 'meta': meta, 'active_phase': phase,
                           'candidates': by_id, 'selected_ids': []}
        phase = phases[key]
        cfg, candidate = phase['config'], phase['candidates'].get(cid)
        if candidate is None:
            raise ValueError('Selected candidate absent from its frozen phase')
        path = root / arm / 'records' / cid / 'result.json'
        if runtime.digest(path.read_bytes()) != entry['result_sha256']:
            raise ValueError('Selected terminal hash changed: ' + str(path))
        result = runtime.load_result(path)
        if result['status'] != 'accepted':
            raise ValueError('Selected result is not accepted or was independently excluded')
        if result.get('candidate_id') != cid or result.get('trait_id') != candidate['trait_id']:
            raise ValueError('Selected terminal identity/trait differs from candidate')
        if runtime.load_checkpoint(path.parent / 'identity.json') != {
                'candidate_sha256': runtime.digest(candidate), 'config_sha256': runtime.digest(cfg)}:
            raise ValueError('Selected identity receipt differs from frozen inputs')
        if not runtime.acceptance(result['review'], cfg) or runtime.simple_checks(result['record']):
            raise ValueError('Selected result fails frozen quality/local checks')
        per_row.verify_accepted(path, result, cfg)
        validate_adoption(path, result, cfg)
        record = result['record']
        if record['scenario_id'] != cid or record['trait_id'] != candidate['trait_id']:
            raise ValueError('Record identity differs from selected candidate')
        if record.get('source_id') != candidate['source_id'] or record.get('source_record_sha256') != runtime.digest(candidate['source']):
            raise ValueError('Record source lineage differs from its frozen candidate')
        if record.get('parent_revision') != cfg['source']['revision']:
            raise ValueError('Record source revision differs from its frozen recipe')
        signature = runtime.digest([record['system'], record['user']])
        plain = re.sub(r'\s+', ' ', record['user']).strip().casefold()
        if signature in signatures or plain in normalized:
            raise ValueError('Duplicate selected prompt across phases')
        signatures.add(signature)
        normalized.add(plain)
        metadata = {k: v for k, v in record.items() if k not in TEXT_KEYS}
        if 'origin' in metadata or 'original_scenario_id' in metadata:
            raise ValueError('Origin metadata keys already occupied')
        metadata.update(original_scenario_id=record['scenario_id'],
                        scenario_id=phase['phase_id'] + '__' + cid,
                        origin={'root': str(root), 'arm': arm, 'phase_id': phase['phase_id'],
                                'config_sha256': phase['config_sha256'], 'candidate_id': cid,
                                'candidate_sha256': runtime.digest(candidate), 'result_sha256': entry['result_sha256']})
        from scratch.dataset_refresh.reconsider_exclusions import verify_history
        corrections = verify_history(path)
        if corrections:
            metadata['exclusion_corrections'] = corrections
        if 'fenced_review_recovery_manifest_sha256' in result:
            metadata['review_format_recovery'] = {
                'method': 'Lossless extraction of the original completed review JSON; unchanged Sonnet answer.',
                'history_sha256': result['fenced_review_recovery_manifest_sha256'], 'inference_calls': 0}
        rows.append({'messages': [{'role': 'system', 'content': record['system']},
                                 {'role': 'user', 'content': record['user']},
                                 {'role': 'assistant', 'content': record['response'], 'reasoning_content': record['reasoning']}],
                     'metadata': metadata})
        phase['selected_ids'].append(cid)
    export_selection = {'scenario_ids': [r['metadata']['scenario_id'] for r in rows], 'quotas': runtime.quotas()}
    publication.validate_export(rows, export_selection)
    # Keep constitution and working craft meaning constant, while allowing explicit prompt/domain revisions.
    contracts = {(p['config']['constitution_sha256'], p['config'].get('craft_spec_sha256'),
                  p['config'].get('original_craft_spec_sha256')) for p in phases.values()}
    if len(contracts) != 1:
        raise ValueError('Composite would mix different constitution/craft contracts')
    return rows, list(phases.values()), export_selection


def prepare(selection_file, destination, source_commit, ledger_end, date, quality_dir=None):
    selection = publication.read_json(selection_file)
    out = Path(destination).resolve()
    if out.exists():
        raise ValueError('Immutable publication destination already exists')
    roots = sorted({str(Path(e['root']).resolve()) for e in selection.get('entries', [])})
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('Publication date must be explicit YYYY-MM-DD')
    if any(out.is_relative_to(Path(root)) for root in roots):
        raise ValueError('Publication destination must be outside every immutable source root')
    with ExitStack() as locks:
        for root in roots:
            locks.enter_context(FileLock(str(Path(root) / 'execution.lock'), timeout=1))
        rows, phases, export_selection = validate_selection(selection)
        ledgers = []
        for phase in phases:
            budget, shared, calls = publication.scoped_ledger(phase['meta'], Path(phase['root']), phase['arm'], ledger_end)
            configured = {m['model'] for m in phase['config']['models'].values()}
            if any('haiku' in e['model'].lower() or e['model'] not in configured for e in calls):
                raise ValueError('Phase API model is unconfigured; explicit repair/reviewer provenance required')
            for call in calls:
                if call['stage'] in {'scenario', 'draft_responses', 'revise_responses'} or re.fullmatch(r'(?:repair|independent_rewrite|short_draft_rewrite)_\d+', call['stage']):
                    if call['model'] != 'anthropic/claude-sonnet-5':
                        raise ValueError('Selected phase contains a non-Sonnet author call')
            ledgers.append((budget, shared, calls))
        if len({str(b) for b, _, _ in ledgers}) != 1:
            raise ValueError('All phases must use the same cumulative budget')
        if any(shared != ledgers[0][1] for _, shared, _ in ledgers):
            raise ValueError('Shared accounting changed during snapshot validation; prepare when settled')
        exposure = [e['charged_or_reserved_usd'] for e in ledgers[0][1]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in exposure) or sum(exposure) > 250:
            raise ValueError('Shared accounting is invalid or exceeds the hard250 budget')
        billing_cache, billing_archives = {}, set()
        for call in ledgers[0][1]:
            if call.get('status') == 'billing_verified_failure':
                budget = ledgers[0][0]
                raw = publication.read_json(budget / 'raw_calls' / f'{call["call_id"]:06d}.json')
                billing_archives.add(validate_billing_call(budget, call, raw, billing_cache))
        out.mkdir(parents=True)
        try:
            commit = publication.freeze_code(out, source_commit, phases[0]['config'])
            for name in ('run.py', 'per_row.py', 'reviewer_probe.py', 'repair_independent.py',
                         'publish_completed.py', 'publish_composite.py', 'recover_scenario_json.py',
                         'retry_technical_review.py', 'recover_missing_changes.py', 'recover_short_draft.py',
                         'billing_evidence.py', 'reconcile_billing.py', 'reconsider_exclusions.py', 'recover_fenced_review.py'):
                actual = Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')
                frozen = (out / 'source_code/scratch/dataset_refresh' / name).read_bytes().replace(b'\r\n', b'\n')
                if actual != frozen:
                    raise ValueError('Source commit does not contain current validating implementation: ' + name)
            runtime.write_rows(out / 'dataset.jsonl', rows)
            export_selection.update(dataset_sha256=runtime.digest((out / 'dataset.jsonl').read_bytes()),
                                    source_selection_sha256=runtime.digest(Path(selection_file).read_bytes()),
                                    origin_entries=selection['entries'])
            runtime.write_json(out / 'selection.json', export_selection)
            shutil.copy2(selection_file, out / 'explicit_selection.json')
            phase_index, calls_all, raw_all = [], [], []
            for phase, (budget, shared, calls) in zip(phases, ledgers):
                source, prefix = Path(phase['root']), out / 'audit/phases' / phase['phase_id']
                freeze_origin_arm(source / phase['arm'], prefix / 'arm')
                for file in ('run_meta.json', 'run_meta.receipt.json', 'active_phase.json'):
                    shutil.copy2(source / file, prefix / file)
                if (source / 'phases').exists():
                    shutil.copytree(source / 'phases', prefix / 'execution_phases')
                phase_index.append({k: phase[k] for k in ('root', 'arm', 'phase_id', 'config_sha256', 'selected_ids')})
                calls_all.extend(calls)
                for call in calls:
                    raw_path = budget / 'raw_calls' / f'{call["call_id"]:06d}.json'
                    raw = publication.read_json(raw_path)
                    publication.safe_audit(raw)
                    billing_archive = validate_billing_call(budget, call, raw, billing_cache)
                    if billing_archive:
                        billing_archives.add(billing_archive)
                    if runtime.digest(raw['request']) != call['request_sha256']:
                        raise ValueError('Scoped raw call differs from ledger')
                    raw_all.append({'call_id': call['call_id'], 'phase_id': phase['phase_id'],
                                    'sha256': runtime.digest(raw_path.read_bytes()), 'value': raw})
            if len({e['call_id'] for e in calls_all}) != len(calls_all):
                raise ValueError('Duplicate physical call in composite provenance')
            runtime.write_json(out / 'origin_phases.json', phase_index)
            common_keys = ('constitution', 'constitution_sha256', 'constitution_role', 'craft_spec',
                           'craft_spec_sha256', 'original_craft_spec', 'original_craft_spec_sha256')
            provenance = {'composite': True, 'pipeline': selection['arm'],
                **{k: phases[0]['config'][k] for k in common_keys if k in phases[0]['config']},
                'models': {p['phase_id']: p['config']['models'] for p in phases},
                'origins': [{'phase_id': p['phase_id'], 'config_sha256': p['config_sha256'],
                             'config': p['config'], 'selected_rows': len(p['selected_ids'])} for p in phases],
                'selection_sha256': runtime.digest((out / 'selection.json').read_bytes()),
                'dataset_sha256': export_selection['dataset_sha256']}
            runtime.write_json(out / 'generation_provenance.json', provenance)
            quality = freeze_quality(quality_dir, out / 'audit/selected_quality', export_selection['dataset_sha256']) if quality_dir else None
            runtime.write_json(out / 'audit/budget/spend.json', calls_all)
            runtime.write_json(out / 'audit/budget/shared_budget_snapshot.json', ledgers[0][1])
            runtime.write_rows(out / 'audit/budget/raw_calls.jsonl', raw_all)
            for archive in billing_archives:
                shutil.copytree(archive, out / 'audit/budget/billing_reconciliations' / archive.name)
            cfg = phases[0]['config']
            name = synth_name(selection['arm'], date=date)
            fields = {
                'title': name,
                'experiment': selection['arm'] + ':716 synthetic conversations selected across immutable recipe phases',
                'date_generated': 'Origin run timestamps: ' + json.dumps({p['phase_id']: p['meta'].get('created_at', 'not recorded') for p in phases}) + '; publication date=' + date,
                'source_repo': 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ ' + commit,
                'constitution': cfg['constitution'] + '; SHA256=' + cfg['constitution_sha256'],
                'models': json.dumps(provenance['models'], ensure_ascii=False),
                'generation_config': 'generation_provenance.json preserves every exact origin recipe and its hash; explicit_selection.json supplies the ordered externally selected716 rows.',
                'provenance': 'origin_phases.json identifies all source roots, frozen configuration hashes and selected candidates. Source inputs, receipts, failed/accepted stages, recoveries, independent audits and raw physical calls are archived under audit/. Dataset SHA256=' + export_selection['dataset_sha256'],
                'schema': 'Only dataset.jsonl is the default training split. messages=[system,user,assistant], with native reasoning_content and final content. Full nonconversation record metadata is retained, with namespaced scenario_id and original_scenario_id. origin binds exact phase/config/candidate/result hashes. Metadata is not injected into the training conversation.',
                'selection': 'Exactly716 externally selected rows with the frozen nine-trait quotas. explicit_selection.json contains the ordered origin manifest; selection.json binds namespaced output IDs and dataset SHA256. No automatic cross-phase selection or pairing is implied.',
                'phases': json.dumps(phase_index, ensure_ascii=False),
                'models_and_recipe': 'All author stages are Sonnet5. Each origin phase retains its exact models, prompts, input snapshots, eligibility/quality/grounding checks, accepted answer receipts and repairs under audit/phases. Where short-draft attempt200 recovery is selected, the failed short draft is untrained input; a fresh final author uses the normal final template/settings PLUS the separately frozen explicit useful-length recovery instruction, followed by fresh normal critics and independent hash-bound adoption. Different explicit prompt/domain phases may coexist; constitution/craft hashes must match. API model IDs are mutable rather than immutable weight identities.',
                'lineage': 'Fresh scenarios are unpaired mechanism inspiration. Original source IDs, revisions, full source-record hashes and mechanically derived source_facts are preserved. Phase namespaces only disambiguate scenario IDs; they do not claim matched pairs. Original full records and every checkpoint remain in flattened provenance archives.',
                'review_limitations': 'Acceptance means all configured checks passed and no bound independent exclusion remains. Model reviews and metadata are not factual guarantees. Prior failures, repairs and rejected rows are retained. No training/evaluation was performed by this workflow.',
                'exclusion_corrections': 'Some independent holds were reversed after a fresh full-conversation reassessment. Each affected row carries exclusion_corrections metadata; exact original exclusion and receipt, reviewed evidence and correction receipt remain in its origin under reconsidered_exclusions/. This explicitly permits ordinary bounded craft/date contexts and accurate visible-prompt references previously excluded by added selection policies. Original author text and automatic reviews are unchanged. All correction histories are verified at export. Shared-topic similarities and minor rhetorical overstatement remain limitations; differing real constraints or remedies are not deduplicated merely because the domain recurs.',
                'selected_quality': ('Hash-bound selected-corpus evidence is preserved under audit/selected_quality; its judgments remain explicit audit judgments, not guarantees.' if quality else 'No separate selected-corpus quality bundle was supplied; origin-arm audit evidence remains archived.'),
                'budget': f'{len(calls_all)} scoped physical calls across selected origin arms; reported USD {sum(e.get("api_reported_cost_usd") or 0 for e in calls_all):.6f}; charged/reserved USD {sum(e["charged_or_reserved_usd"] for e in calls_all):.6f}. Shared ledger snapshot through exclusive index {ledger_end}; raw requests only for these origin root/arms. No transport credentials.',
                'mixing': 'Synthetic only. Planned exact716+9284 mixture uses the identical seed0 replay selection from dougalldeepmind/2026-09-08-nosynth-mix @7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd, yielding7.16% synthetic rows; the existing rounded7 naming suffix remains.'}
            if cfg.get('craft_spec'):
                fields['craft_preference'] = cfg['craft_spec'] + '; SHA256=' + cfg['craft_spec_sha256'] + '. Craft generation uses the qualified operational tensions; ethical constitution is a separate compatibility check. Original specification and exact source provenance remain archived.'
            front = {'tags': training_data_tags('synth', selection['arm'], cfg['constitution']),
                     'configs': [{'config_name': 'default', 'default': True, 'data_files': 'dataset.jsonl'}]}
            runtime.write_json(out / 'card_fields.json', fields)
            runtime.write_json(out / 'card_front_matter.json', front)
            (out / 'README.md').write_text(card_markdown(fields, front), encoding='utf-8')
            runtime.write_json(out / 'publication_manifest.json', {'name': name, 'source_commit': commit,
                'arm': selection['arm'], 'composite': True, 'ledger_end_exclusive': ledger_end,
                'selected_quality': quality,
                'dataset_sha256': export_selection['dataset_sha256'],
                'files': {p.relative_to(out).as_posix(): runtime.digest(p.read_bytes()) for p in out.rglob('*') if p.is_file()}})
        except Exception:
            (out / 'PREPARATION_FAILED').write_text('Incomplete composite snapshot. Do not publish.', encoding='utf-8')
            raise
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'prepare', 'push'])
    parser.add_argument('--selection')
    parser.add_argument('--destination')
    parser.add_argument('--source-commit')
    parser.add_argument('--ledger-end', type=int)
    parser.add_argument('--date')
    parser.add_argument('--quality-dir')
    args = parser.parse_args()
    if args.command == 'validate':
        if not args.selection:
            parser.error('--selection required')
        rows, phases, _ = validate_selection(publication.read_json(args.selection))
        print(json.dumps({'rows': len(rows), 'phases': len(phases)}))
    elif args.command == 'push':
        if not args.destination:
            parser.error('--destination required')
        print(publication.push(args.destination))
    else:
        if any(getattr(args, k) is None for k in ('selection', 'destination', 'source_commit', 'ledger_end', 'date')):
            parser.error('prepare requires --selection --destination --source-commit --ledger-end --date')
        print(prepare(args.selection, args.destination, args.source_commit, args.ledger_end, args.date, args.quality_dir))


if __name__ == '__main__':
    main()
