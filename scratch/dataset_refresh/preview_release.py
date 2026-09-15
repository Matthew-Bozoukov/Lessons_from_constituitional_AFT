# ABOUTME: Preview exact composite publication bytes and audit cached tokenizer/masks, corpus overlap and literal flags.
# ABOUTME: Freeze supplied independent evidence into the existing quality_manifest contract; no network, generation or publication.
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

from filelock import FileLock
from transformers import AutoTokenizer

from scratch.dataset_refresh import run as runtime, publish_composite, audit_corpus, screen_accepted
from scratch.dataset_refresh.validate_mixtures import TOKENIZER, token_audit
from src.model_profile import model_profile


def inventory(directory):
    directory = Path(directory).absolute()
    if any(p.is_symlink() or p.is_junction() for p in (directory, *directory.parents)):
        raise ValueError('Artifact directory cannot traverse symlinks/junctions')
    directory = directory.resolve(strict=True)
    result = {}
    for path in directory.rglob('*'):
        if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(directory):
            raise ValueError('Artifact escapes its directory or uses a link')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = runtime.digest(path.read_bytes())
    return result


def copy_bound(source, destination, hashes):
    source, destination = Path(source), Path(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError('Copy destination must be outside its source')
    for name, expected in hashes.items():
        path, target = source / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if runtime.digest(target.read_bytes()) != expected:
            raise ValueError('Artifact changed while copying: ' + name)
    if inventory(source) != hashes:
        raise ValueError('Source artifacts changed while copying')


def preview(selection_path, output):
    selection_path, output = Path(selection_path).resolve(), Path(output).resolve()
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    roots = sorted({str(Path(e['root']).resolve()) for e in selection['entries']})
    if output.exists() or any(output.is_relative_to(Path(root)) for root in roots):
        raise ValueError('Use a new preview directory outside all frozen source roots')
    with ExitStack() as locks:
        for root in roots:
            locks.enter_context(FileLock(str(Path(root) / 'execution.lock'), timeout=1))
        rows, phases, export_selection = publish_composite.validate_selection(selection)
        output.mkdir(parents=True)
        runtime.write_rows(output / 'dataset.jsonl', rows)
        (output / 'selection.json').write_bytes(selection_bytes)
    dataset_sha = runtime.digest((output / 'dataset.jsonl').read_bytes())
    quality = output / 'quality'
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
        profile = model_profile(TOKENIZER)
        diagnostics, triage, failures = [], [], []
        for row in rows:
            sid = row['metadata']['scenario_id']
            record = {m['role']: m['content'] for m in row['messages'] if m['role'] != 'assistant'}
            record.update(reasoning=row['messages'][-1]['reasoning_content'], response=row['messages'][-1]['content'])
            flags = screen_accepted.screen_record(record)
            triage.append({'id': sid, 'record': record, 'flags': flags})
            failures.extend({'scenario_id': sid, **f} for f in flags if f['kind'] in ('missing_text', 'replacement_character'))
            try:
                diagnostics.append({'scenario_id': sid, **token_audit(row, tokenizer, profile, 8192)})
            except (ValueError, AssertionError) as exc:
                failures.append({'scenario_id': sid, 'kind': 'training_stream_or_mask_failure', 'error': str(exc)})
        runtime.write_json(quality / 'token_mask_audit.json', {
            'dataset_sha256': dataset_sha, 'rows': len(rows), 'status': 'failed' if failures else 'passed',
            'tokenizer': TOKENIZER, 'tokenizer_snapshot_sha256': runtime.digest(tokenizer.backend_tokenizer.to_str().encode()),
            'max_train_tokens': 8192, 'supervise': 'all', 'untruncated': True, 'failures': failures,
            'totals': dict(sum((Counter({k: v for k, v in item.items() if isinstance(v, int)}) for item in diagnostics), Counter())),
            'diagnostics': diagnostics})
        runtime.write_json(quality / 'literal_screen.json', {
            'dataset_sha256': dataset_sha, 'rows': len(rows),
            'interpretation': 'Process wording and new quantities are candidates requiring contextual adjudication, not automatic errors.',
            'flag_counts': dict(Counter(f['kind'] for row in triage for f in row['flags'])),
            'flagged_rows': [{'scenario_id': r['id'], 'flags': r['flags']} for r in triage if r['flags']],
            'repeated_final_phrases': screen_accepted.repeated_final_phrases(triage)})
        corpus = audit_corpus.audit(output / 'dataset.jsonl', quality, local_files_only=True)
        if corpus['input_sha256'] != dataset_sha:
            raise ValueError('Corpus audit input changed')
        manifest = {'dataset_sha256': dataset_sha, 'selection_sha256': runtime.digest(selection_bytes),
                    'arm': selection['arm'], 'rows': len(rows), 'quotas': export_selection['quotas'],
                    'phase_ids': [p['phase_id'] for p in phases], 'automatic_checks': 'failed' if failures else 'passed',
                    'independent_adjudication': 'not supplied by automatic preview',
                    'publisher_sha256': runtime.digest(Path(publish_composite.__file__).read_bytes()),
                    'preview_helper_sha256': runtime.digest(Path(__file__).read_bytes()),
                    'quality_files': inventory(quality)}
        runtime.write_json(output / 'preview_manifest.json', manifest)
        return manifest
    except BaseException:
        (output / 'PREVIEW_FAILED').write_text('Incomplete preview; do not seal or publish.', encoding='utf-8')
        raise


def seal(preview_dir, evidence_dir, output):
    # Inspect caller-supplied paths before resolve discards evidence of ancestor links.
    inventory(preview_dir)
    inventory(evidence_dir)
    preview_dir, evidence_dir, output = map(lambda p: Path(p).resolve(), (preview_dir, evidence_dir, output))
    if output.exists() or output.is_relative_to(preview_dir) or output.is_relative_to(evidence_dir):
        raise ValueError('Use a new quality output outside preview and independent evidence')
    if (preview_dir / 'PREVIEW_FAILED').exists():
        raise ValueError('Incomplete preview cannot be sealed')
    manifest = json.loads((preview_dir / 'preview_manifest.json').read_text(encoding='utf-8'))
    if manifest['automatic_checks'] != 'passed':
        raise ValueError('Automatic length/mask/text checks failed; do not seal')
    dataset_sha = runtime.digest((preview_dir / 'dataset.jsonl').read_bytes())
    if dataset_sha != manifest['dataset_sha256'] or runtime.digest((preview_dir / 'selection.json').read_bytes()) != manifest['selection_sha256']:
        raise ValueError('Preview dataset or selection changed')
    quality_hashes = inventory(preview_dir / 'quality')
    if quality_hashes != manifest['quality_files']:
        raise ValueError('Automatic audit artifacts changed after preview')
    evidence_hashes = inventory(evidence_dir)
    if not evidence_hashes:
        raise ValueError('Supply the independent final selection policy/adjudication evidence')
    # One explicit evidence manifest binds the independently supplied files to these exact rows.
    owner = json.loads((evidence_dir / 'adjudication.json').read_text(encoding='utf-8'))
    if owner.get('dataset_sha256') != dataset_sha or not isinstance(owner.get('scope'), str) or not owner['scope'].strip():
        raise ValueError('Independent adjudication must bind the exact dataset and state its scope')
    output.mkdir(parents=True)
    try:
        copy_bound(preview_dir / 'quality', output / 'automatic', quality_hashes)
        copy_bound(evidence_dir, output / 'independent', evidence_hashes)
        runtime.write_json(output / 'preview_provenance.json', manifest)
        files = inventory(output)
        result = {'dataset_sha256': dataset_sha, 'files': files,
                  'interpretation': 'Hashes freeze supplied audit evidence; they do not certify its judgments or override quality exclusions.'}
        runtime.write_json(output / 'quality_manifest.json', result)
        # Exercise the exact publication intake validator without touching any source or publishing.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            publish_composite.freeze_quality(output, Path(temp) / 'checked_quality', dataset_sha)
        return result
    except BaseException:
        (output / 'QUALITY_SEAL_FAILED').write_text('Incomplete quality bundle; do not publish.', encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preview')
    p.add_argument('--selection', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('seal')
    p.add_argument('--preview', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    result = preview(args.selection, args.output) if args.command == 'preview' else seal(args.preview, args.evidence, args.output)
    print(json.dumps(result, indent=2))
    if result.get('automatic_checks') == 'failed':
        raise SystemExit(2)
