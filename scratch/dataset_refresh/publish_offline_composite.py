# ABOUTME: Publish the fixed650-row nonmoral baseline plus66 independently adopted saved answers.
# ABOUTME: Preserve distinct review routes, exact716 release gates and full closed-budget provenance without generation.
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

from scratch.dataset_refresh import run as base, per_row, offline_acceptance as offline
from scratch.dataset_refresh import publish_composite as composite, publish_completed as publication
from scratch.dataset_refresh import preserve_partial, preview_release, prepare_mixtures
from scratch.dataset_refresh.billing_evidence import validate_billing_call
from src.infra.huggingface import card_markdown, training_data_tags
from src.naming import synth_name

BASE_SELECTION_SHA = '5e52a18a45dcef2065428c3a9b38d4a1f1c1c79b75a7f34817c572c0c9ebf0aa'
ARM = 'nonmoral-advice'
COMMON = ('constitution', 'constitution_sha256', 'constitution_role', 'craft_spec',
          'craft_spec_sha256', 'original_craft_spec', 'original_craft_spec_sha256')


def validate_base(selection):
    """The original per-row/source/adoption contract, applied to the explicit650-row subset."""
    if selection.get('arm') != ARM or len(selection.get('entries', [])) != 650:
        raise ValueError('Require the exact corrected650-row base selection')
    phases, rows, seen = {}, [], set()
    for entry in selection['entries']:
        if set(entry) != {'root', 'arm', 'candidate_id', 'result_sha256'} or entry['arm'] != ARM:
            raise ValueError('Invalid base origin entry')
        root, cid = Path(entry['root']).resolve(), entry['candidate_id']
        if not re.fullmatch(r't[1-9]_\d+_v\d+', cid) or (str(root), cid) in seen:
            raise ValueError('Invalid or duplicate base source identity')
        seen.add((str(root), cid))
        if str(root) not in phases:
            cfg = base.validate_arm(root, ARM)
            if cfg.get('per_row_regime') is not True or cfg['pipeline'] != ARM:
                raise ValueError('Base is not the frozen per-row arm')
            per_row.assert_models(cfg)
            meta = base.load_checkpoint(root/'run_meta.json')
            active = offline.read(root/'active_phase.json')
            matches = [p for p in (root/'phases').glob('*.json') if not p.name.endswith('.receipt.json') and base.load_checkpoint(p) == active]
            if len(matches) != 1:
                raise ValueError('Original execution phase receipt differs')
            for name, key in [('run.py', 'code_sha256'), ('per_row.py', 'per_row_code_sha256')]:
                if active[key] != offline.sha(Path(__file__).with_name(name)):
                    raise ValueError('Frozen source implementation differs')
            candidates = base.read_rows(root/ARM/'candidates.jsonl')
            by_id = {c['candidate_id']: c for c in candidates}
            if len(candidates) != len(by_id):
                raise ValueError('Duplicate frozen source candidate IDs')
            phase_id = 'phase_'+base.digest({'root_meta_sha256': offline.sha(root/'run_meta.json'), 'config_sha256': base.digest(cfg)})[:16]
            phases[str(root)] = {'root': str(root), 'arm': ARM, 'phase_id': phase_id, 'config': cfg,
                                 'config_sha256': base.digest(cfg), 'meta': meta, 'active_phase': active,
                                 'candidates': by_id, 'selected_ids': []}
        phase = phases[str(root)]
        cfg, candidate = phase['config'], phase['candidates'].get(cid)
        if candidate is None:
            raise ValueError('Selected base candidate missing')
        path = offline.bound(root/ARM/'records'/cid/'result.json', entry['result_sha256'])
        result = base.load_result(path)
        if (result['status'] != 'accepted' or result.get('candidate_id') != cid or
                result.get('trait_id') != candidate['trait_id'] or
                base.load_checkpoint(path.parent/'identity.json') != {'candidate_sha256': base.digest(candidate), 'config_sha256': base.digest(cfg)} or
                not base.acceptance(result['review'], cfg) or base.simple_checks(result['record'])):
            raise ValueError('Base row no longer passes its original acceptance/identity checks')
        per_row.verify_accepted(path, result, cfg)
        composite.validate_adoption(path, result, cfg)
        record = result['record']
        if (record['scenario_id'] != cid or record['trait_id'] != candidate['trait_id'] or
                record['source_id'] != candidate['source_id'] or record['source_record_sha256'] != base.digest(candidate['source']) or
                record['parent_revision'] != cfg['source']['revision']):
            raise ValueError('Base source lineage differs')
        metadata = {k: v for k, v in record.items() if k not in composite.TEXT_KEYS}
        if 'origin' in metadata or 'original_scenario_id' in metadata:
            raise ValueError('Base provenance metadata collision')
        metadata.update(original_scenario_id=cid, scenario_id=phase['phase_id']+'__'+cid,
                        origin={**entry, 'root': str(root), 'phase_id': phase['phase_id'],
                                'config_sha256': phase['config_sha256'], 'candidate_sha256': base.digest(candidate)})
        from scratch.dataset_refresh.reconsider_exclusions import verify_history
        corrections = verify_history(path)
        if corrections:
            metadata['exclusion_corrections'] = corrections
        if 'fenced_review_recovery_manifest_sha256' in result:
            metadata['review_format_recovery'] = {'method': 'Lossless original review parsing; unchanged Sonnet answer.',
                'history_sha256': result['fenced_review_recovery_manifest_sha256'], 'inference_calls': 0}
        rows.append({'messages': offline.messages(record), 'metadata': metadata})
        phase['selected_ids'].append(cid)
    return rows, list(phases.values())


def validate_selection(path):
    selection = offline.read(path)
    if set(selection) != {'arm', 'base_selection', 'offline_entries'} or selection['arm'] != ARM:
        raise ValueError('Explicit base650 plus offline66 selection required')
    ref = selection['base_selection']
    if ref.get('sha256') != BASE_SELECTION_SHA:
        raise ValueError('The corrected canonical650 base must remain exact')
    base_path = offline.bound(ref['path'], BASE_SELECTION_SHA)
    base_selection = offline.read(base_path)
    rows, phases = validate_base(base_selection)
    entries = selection['offline_entries']
    if not isinstance(entries, list) or len(entries) != 66:
        raise ValueError('Exactly66 independently adopted additions required')
    origins = {(str(Path(e['root']).resolve()), e['arm'], e['candidate_id']) for e in base_selection['entries']}
    dossiers = []
    for entry in entries:
        if set(entry) != {'acceptance_path', 'acceptance_sha256'}:
            raise ValueError('Offline entry requires exact acceptance path/hash')
        row, audit = offline.validate_accepted(entry['acceptance_path'], entry['acceptance_sha256'])
        d = audit['dossier']; ref = d['source_ref']
        key = (ref['root'], ref['arm'], ref['candidate_id'])
        if key in origins:
            raise ValueError('Same source selected through two acceptance routes')
        origins.add(key)
        rows.append(row)
        dossiers.append({'path': str(Path(entry['acceptance_path']).resolve().parent),
                         'acceptance_sha256': entry['acceptance_sha256'], **audit})
    offline.validate_release(rows)
    cfg = phases[0]['config']
    for phase in phases:
        if any(phase['config'].get(k) != cfg.get(k) for k in COMMON):
            raise ValueError('Base constitution/craft contracts disagree')
    for dossier in dossiers:
        d, files = offline.validate_dossier(Path(dossier['path'])/'dossier.json')
        review_cfg = offline.read(files['review_config'])
        if any(review_cfg.get(k) != cfg.get(k) for k in COMMON):
            raise ValueError('New offline review uses a different constitution/craft contract')
    return rows, phases, dossiers


def provenance(rows, phases, dossiers, dataset_sha):
    cfg = phases[0]['config']
    requests = {}
    author_counts = Counter()
    for item in dossiers:
        d, files = offline.validate_dossier(Path(item['path'])/'dossier.json')
        request = offline.read(files['author_raw'])['request']
        settings = {k: v for k, v in request.items() if k != 'messages'}
        requests['author_'+base.digest(settings)[:16]] = settings
        author_counts[d['author_kind']] += 1
    offline_cfg = {'pipeline': ARM, **{k: cfg[k] for k in COMMON if k in cfg}, 'models': requests,
        'acceptance_route': offline.ROUTE,
        'author_kinds': dict(author_counts),
        'author_recipe_evidence': 'Every dossier contains the actual request/response, successful physical call receipt, source config and new review contract. Untouched finals retain their real historical author prompt, not a fictitious new generation.',
        'review_procedure': 'Independent Codex agent full-read review plus explicit root adoption, source/author/local/native checks; no automatic model judge pass is asserted for this route.'}
    origins = [{'phase_id': p['phase_id'], 'config_sha256': p['config_sha256'], 'config': p['config'],
                'selected_rows': len(p['selected_ids']), 'acceptance_route': 'original_per_row_with_verified_corrections'} for p in phases]
    origins.append({'phase_id': 'offline_full_read', 'config_sha256': base.digest(offline_cfg), 'config': offline_cfg,
                    'selected_rows': len(dossiers), 'acceptance_route': offline.ROUTE})
    value = {'composite': True, 'mixed_acceptance_routes': True, 'pipeline': ARM,
             **{k: cfg[k] for k in COMMON if k in cfg}, 'dataset_sha256': dataset_sha,
             'origins': origins, 'models': {p['phase_id']: p['config']['models'] for p in origins},
             'review_route_counts': {'original_per_row_with_verified_corrections': 650, offline.ROUTE: 66}}
    return value


def preview(selection_path, output):
    """Write exact publication bytes and local-only census/native diagnostics for adjudication."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Preview destination must be new')
    rows, phases, dossiers = validate_selection(selection_path)
    roots = [Path(p['root']) for p in phases]+[Path(p['path']) for p in dossiers]
    if any(output.is_relative_to(p) for p in roots):
        raise ValueError('Preview cannot be inside a source')
    output.mkdir(parents=True)
    base.write_rows(output/'dataset.jsonl', rows)
    shutil.copyfile(selection_path, output/'selection.json')
    dataset_sha = offline.sha(output/'dataset.jsonl')
    try:
        from transformers import AutoTokenizer
        from src.model_profile import model_profile
        from scratch.dataset_refresh.validate_mixtures import TOKENIZER, token_audit
        from scratch.dataset_refresh import audit_corpus, screen_accepted
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        diagnostics, flags, failures = [], [], []
        for row in rows:
            sid = row['metadata']['scenario_id']
            record = {m['role']: m['content'] for m in row['messages'] if m['role'] != 'assistant'}
            record.update(reasoning=row['messages'][-1]['reasoning_content'], response=row['messages'][-1]['content'])
            found = screen_accepted.screen_record(record)
            flags.append({'id': sid, 'record': record, 'flags': found})
            failures.extend({'scenario_id': sid, **f} for f in found if f['kind'] in ('missing_text', 'replacement_character'))
            try:
                diagnostics.append({'scenario_id': sid, **token_audit(row, tokenizer, model_profile(TOKENIZER), 8192)})
            except (AssertionError, ValueError) as exc:
                failures.append({'scenario_id': sid, 'error': str(exc)})
        quality = output/'quality'
        base.write_json(quality/'token_mask_audit.json', {'dataset_sha256': dataset_sha, 'rows': 716,
            'status': 'failed' if failures else 'passed', 'tokenizer': TOKENIZER, 'max_train_tokens': 8192,
            'untruncated': True, 'supervise': 'all', 'failures': failures, 'diagnostics': diagnostics,
            'tokenizer_snapshot_sha256': base.digest(tokenizer.backend_tokenizer.to_str().encode())})
        base.write_json(quality/'literal_screen.json', {'dataset_sha256': dataset_sha,
            'interpretation': 'Contextual triage flags, not automatic judgments.',
            'flagged_rows': [{'scenario_id': r['id'], 'flags': r['flags']} for r in flags if r['flags']],
            'repeated_final_phrases': screen_accepted.repeated_final_phrases(flags)})
        audit_corpus.audit(output/'dataset.jsonl', quality, local_files_only=True)
        manifest = {'dataset_sha256': dataset_sha, 'selection_sha256': offline.sha(output/'selection.json'),
            'arm': ARM, 'rows': 716, 'quotas': base.quotas(), 'automatic_checks': 'failed' if failures else 'passed',
            'review_route_counts': {'original': 650, offline.ROUTE: 66},
            'independent_adjudication': 'Required separately for final joint corpus.',
            'publisher_sha256': offline.sha(__file__), 'quality_files': preview_release.inventory(quality)}
        base.write_json(output/'preview_manifest.json', manifest)
        return manifest
    except BaseException:
        (output/'PREVIEW_FAILED').write_text('Incomplete preview; do not publish.', encoding='utf-8')
        raise


def validate_quality(directory, rows, dataset_sha):
    directory = Path(directory)
    token = offline.read(directory/'automatic/token_mask_audit.json')
    if (token.get('dataset_sha256') != dataset_sha or token.get('rows') != 716 or token.get('status') != 'passed' or
            token.get('failures') != [] or token.get('untruncated') is not True or
            token.get('tokenizer') != 'Qwen/Qwen3.6-27B' or token.get('max_train_tokens') != 8192):
        raise ValueError('Final native quality audit does not pass for these exact716 rows')
    diag = token['diagnostics']
    if (len(diag) != 716 or {d['scenario_id'] for d in diag} != {r['metadata']['scenario_id'] for r in rows} or
            any(not 0 < d['supervised_tokens'] <= d['training_tokens'] <= 8192 or
                d['raw_reasoning_tokens'] <= 0 or d['raw_assistant_content_tokens'] <= 0 for d in diag)):
        raise ValueError('Native per-row lengths/masks differ')
    corpus = offline.read(directory/'automatic/corpus_audit.json')
    if corpus.get('input_sha256') != dataset_sha or corpus.get('rows') != 716:
        raise ValueError('Final corpus census differs')
    for name in ('full_census.jsonl', 'semantic_pairs.jsonl', 'lexical_pairs.jsonl', 'literal_screen.json'):
        offline.regular(directory/'automatic'/name)
    adjudication = offline.read(directory/'independent/adjudication.json')
    if (adjudication.get('dataset_sha256') != dataset_sha or adjudication.get('release_approved') is not True or
            adjudication.get('unresolved_material_issues') != [] or not adjudication.get('scope')):
        raise ValueError('Final joint independent/root adjudication is required')


def accounting(ledger, cutoff):
    if type(cutoff) is not int or cutoff != len(ledger):
        raise ValueError('Release needs the exact closed shared ledger cutoff')
    for index, entry in enumerate(ledger):
        amount = entry['charged_or_reserved_usd']
        reported = entry.get('api_reported_cost_usd')
        if (entry['call_id'] != index or entry['status'] not in ('settled', 'billing_verified_failure') or
                type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0 or
                reported is not None and (type(reported) not in (int, float) or not math.isfinite(reported) or reported < 0)):
            raise ValueError('Unresolved/active/invalid shared accounting')
    exposure = sum(e['charged_or_reserved_usd'] for e in ledger)
    if exposure > 270:
        raise ValueError('Shared accounting exceeds the user-approved270 ceiling')
    return {'ledger_end_exclusive': cutoff, 'hard_cap_usd': 270, 'charged_usd': exposure,
            'api_reported_usd': sum(e.get('api_reported_cost_usd') or 0 for e in ledger),
            'api_reported_missing_count': sum(e.get('api_reported_cost_usd') is None for e in ledger),
            'statuses': dict(Counter(e['status'] for e in ledger)), 'active_or_uncertain_calls': 0}


def prepare(selection_path, destination, source_commit, ledger_end, date, quality_dir):
    out = Path(destination).resolve()
    if out.exists() or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        raise ValueError('Use a new destination and explicit publication date')
    rows, phases, dossiers = validate_selection(selection_path)
    roots = {Path(p['root']) for p in phases}
    roots.update(Path(d['path']) for d in dossiers)
    if any(out.is_relative_to(p) for p in roots) or out.is_relative_to(Path(quality_dir).resolve()):
        raise ValueError('Publication destination cannot be inside its inputs')
    budget = offline.ORIGINAL_BUDGET_ROOT
    with ExitStack() as locks:
        for root in sorted({p['root'] for p in phases}):
            locks.enter_context(FileLock(str(Path(root)/'execution.lock'), timeout=1))
        locks.enter_context(FileLock(str(budget/'spend.lock'), timeout=1))
        rows, phases, dossiers = validate_selection(selection_path)
        ledger = offline.read(budget/'spend.json')
        totals = accounting(ledger, ledger_end)
        out.mkdir(parents=True)
        try:
            code = preserve_partial.freeze_code(out, source_commit)
            base.write_rows(out/'dataset.jsonl', rows)
            dataset_sha = offline.sha(out/'dataset.jsonl')
            validate_quality(quality_dir, rows, dataset_sha)
            quality = composite.freeze_quality(quality_dir, out/'audit/selected_quality', dataset_sha)
            shutil.copyfile(selection_path, out/'explicit_selection.json')
            ref = offline.read(selection_path)['base_selection']
            shutil.copyfile(offline.bound(ref['path'], BASE_SELECTION_SHA), out/'base650_selection.json')
            selection = {'scenario_ids': [r['metadata']['scenario_id'] for r in rows], 'quotas': base.quotas(),
                         'dataset_sha256': dataset_sha, 'source_selection_sha256': offline.sha(selection_path)}
            base.write_json(out/'selection.json', selection)
            archives, scopes, archived_roots = [], set(), set()
            for phase in phases:
                root = Path(phase['root']); scopes.add((str(root), ARM)); archived_roots.add(str(root))
                archives.append(preserve_partial.archive_tree(root/ARM, out/'audit'/('origin_'+phase['phase_id']+'.tar.gz')))
                phase_evidence = {name: offline.read(root/name) for name in ('run_meta.json', 'run_meta.receipt.json', 'active_phase.json')}
                phase_evidence['execution_phases'] = {p.name: offline.read(p) for p in (root/'phases').glob('*.json')}
                base.write_json(out/'audit'/('phase_'+phase['phase_id']+'.json'), phase_evidence)
            offline_index = []
            for item in dossiers:
                path = Path(item['path']); d = item['dossier']; ident = item['acceptance_sha256']
                target = out/'audit/offline'/ident
                hashes = preview_release.inventory(path)
                preview_release.copy_bound(path, target, hashes)
                offline.validate_accepted(target/'acceptance.json', ident)
                call = ledger[d['physical_receipt']['call_id']]
                if base.digest(call) != d['physical_receipt']['ledger_entry_sha256']:
                    raise ValueError('Offline accepted physical receipt differs from closed shared ledger')
                scopes.add((call['run_root'], call['arm']))
                offline_index.append({'acceptance_sha256': ident, 'path': target.relative_to(out).as_posix(),
                                      'source_ref': d['source_ref'], 'author_kind': d['author_kind']})
            # Preserve complete new execution roots, including unsuccessful outcomes, not only selected answers.
            for root, arm in sorted(scopes):
                if root not in archived_roots:
                    if out.is_relative_to(Path(root)):
                        raise ValueError('Execution archive contains publication destination')
                    archives.append(preserve_partial.archive_tree(Path(root), out/'audit'/('execution_'+base.digest(root)[:16]+'.tar.gz')))
                    archived_roots.add(root)
            calls = [e for e in ledger if (e.get('run_root'), e.get('arm')) in scopes]
            if any(e['model'] != offline.MODEL for e in calls):
                raise ValueError('Selected all-Sonnet author/old-review execution scopes contain an unexpected model')
            call_ids = {e['call_id'] for e in calls}
            raw_calls, billing_archives, cache = [], set(), {}
            for entry in ledger:
                if entry['status'] == 'billing_verified_failure' or entry['call_id'] in call_ids:
                    raw_path = budget/'raw_calls'/f'{entry["call_id"]:06d}.json'
                    raw = offline.read(raw_path)
                    proof = validate_billing_call(budget, entry, raw, cache)
                    if proof:
                        billing_archives.add(proof)
                    if entry['call_id'] in call_ids:
                        publication.safe_audit(raw)
                        if base.digest(raw['request']) != entry['request_sha256']:
                            raise ValueError('Scoped raw request changed')
                        raw_calls.append({'call_id': entry['call_id'], 'sha256': offline.sha(raw_path), 'value': raw})
            for archive in billing_archives:
                archives.append(preserve_partial.archive_tree(archive, out/'audit/billing'/('proof_'+archive.name+'.tar.gz')))
            base.write_json(out/'audit/budget/shared_budget_snapshot.json', ledger)
            base.write_json(out/'audit/budget/scoped_spend.json', calls)
            base.write_rows(out/'audit/budget/scoped_raw_calls.jsonl', raw_calls)
            base.write_json(out/'audit/archives.json', archives)
            base.write_json(out/'offline_acceptance_index.json', offline_index)
            prov = provenance(rows, phases, dossiers, dataset_sha)
            base.write_json(out/'generation_provenance.json', prov)
            prepare_mixtures.synthetic_provenance(out/'generation_provenance.json', ARM)
            cfg = phases[0]['config']; name = synth_name(ARM, date=date)
            fields = {'title': name, 'experiment': '716 nonmoral human-advice conversations:650 original accepted rows plus66 independently adopted saved answers.',
                'date_generated': 'Multiple immutable author phases; publication date '+date,
                'constitution': cfg['constitution']+'; SHA256='+cfg['constitution_sha256']+'; full compatibility review, not an ethical premise injected into the nonmoral author task.',
                'craft_preference': cfg['craft_spec']+'; SHA256='+cfg['craft_spec_sha256']+'. Qualified nine varied craft tensions; original source preferences and historical author prompts are preserved.',
                'models': json.dumps(prov['models'], ensure_ascii=False),
                'review_routes': '650 rows retain verified original automatic reviews and correction/adoption histories.66 additions use independent Codex agent full-read review and explicit root adoption of exact message bytes; these are not human reviews and do not assert automatic Sonnet judge passes. Additions may be one new Sonnet saved-input revision or unchanged historical Sonnet finals; exact counts/settings are in generation_provenance.json.',
                'source_repo': 'https://github.com/Matthew-Bozoukov/Lessons_from_constituitional_AFT @ '+code['commit'],
                'provenance': 'Base650 selection, explicit offline acceptance hashes, all origin checkpoints, unsuccessful new execution outcomes, accepted dossiers, exact scoped raw calls/receipts, source code and closed shared billing ledger are archived. Original excluded or failed rows remain unchanged; new acceptance is a separate route. Source filesystem paths are historical identifiers, not required live locations.',
                'schema': 'Only dataset.jsonl is the default train split. messages=[system,user,assistant], assistant reasoning_content and final content. Other files are untrained provenance; metadata is never injected into model messages.',
                'selection': 'Exactly716 rows with80 each for t1–t5 and79 each for t6–t9. Source prompts/IDs are unique; phase names do not imply matched pairs. Joint native8192, corpus/lexical/semantic triage and independent adjudication bind these exact bytes.',
                'limitations': 'Independent/model judgments and similar-domain screening are not guarantees. Different historical generation and acceptance routes coexist explicitly. No training or evaluation is performed by this publication workflow.',
                'budget': json.dumps(totals)+'; shared accounting covers both arms, prior failures, calibration and corrections. Failed billing-resolved content is never promoted to valid author output.',
                'mixing': 'Synthetic only. Use716 synthetic plus identical9284 seed0 replay rows from '+prepare_mixtures.BASE_REPO+'@'+prepare_mixtures.BASE_REVISION+'. Exactly7.16% of rows; canonical naming retains rounded7. Replay native reasoning/backfill is inherited without new generation; row share is not supervised-token share.'}
            front = {'tags': training_data_tags('synth', ARM, cfg['constitution']),
                     'configs': [{'config_name': 'default', 'default': True, 'data_files': 'dataset.jsonl'}]}
            base.write_json(out/'card_fields.json', fields); base.write_json(out/'card_front_matter.json', front)
            (out/'README.md').write_text(card_markdown(fields, front), encoding='utf-8')
            base.write_json(out/'publication_manifest.json', {'name': name, 'source_commit': source_commit, 'arm': ARM,
                'composite': True, 'mixed_acceptance_routes': True, 'dataset_sha256': dataset_sha,
                'selected_quality': quality, 'budget': totals,
                'files': preview_release.inventory(out)})
        except BaseException:
            (out/'PREPARATION_FAILED').write_text('Incomplete mixed-route snapshot; do not publish.', encoding='utf-8')
            raise
    return str(out)


def mixture_card(config, mixture_dir, audit, synthetic_provenance):
    """Use existing pinned replay validation, then explicitly disclose the new acceptance route."""
    name = prepare_mixtures.prepare_card(config, mixture_dir, audit, synthetic_provenance)
    prov = prepare_mixtures.synthetic_provenance(synthetic_provenance, ARM)
    if prov.get('review_route_counts') != {'original_per_row_with_verified_corrections': 650, offline.ROUTE: 66}:
        raise ValueError('Mixed-route synthetic provenance differs')
    out = Path(mixture_dir); fields = offline.read(out/'card_fields.json')
    fields['review_routes'] = 'Synthetic650 rows use verified original per-row acceptance;66 saved answers use fresh independent Codex agent full-read review and explicit root adoption. No automatic Sonnet judge passes or human reviews are fabricated for those additions. Exact author requests and review evidence remain in the pinned synthetic archive.'
    base.write_json(out/'card_fields.json', fields)
    (out/'README.md').write_text(card_markdown(fields, offline.read(out/'card_front_matter.json')), encoding='utf-8')
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('validate'); p.add_argument('--selection', required=True)
    p = sub.add_parser('preview'); p.add_argument('--selection', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('prepare')
    for key in ('selection', 'destination', 'source-commit', 'date', 'quality-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--ledger-end', type=int, required=True)
    p = sub.add_parser('push'); p.add_argument('--destination', required=True)
    p = sub.add_parser('mixture-card')
    for key in ('config', 'mixture-dir', 'audit', 'synthetic-provenance'):
        p.add_argument('--'+key, required=True)
    args = parser.parse_args()
    if args.command == 'validate':
        rows, _, _ = validate_selection(args.selection); print(offline.validate_release(rows))
    elif args.command == 'preview':
        print(preview(args.selection, args.output))
    elif args.command == 'prepare':
        print(prepare(args.selection, args.destination, args.source_commit, args.ledger_end, args.date, args.quality_dir))
    elif args.command == 'mixture-card':
        print(mixture_card(args.config, args.mixture_dir, args.audit, args.synthetic_provenance))
    else:
        print(publication.push(args.destination))


if __name__ == '__main__':
    main()
