# ABOUTME: Preview and apply explicit, hash-bound reversals of independent dataset exclusions without inference.
# ABOUTME: Preserve original verdicts and exclusion receipts; require a reviewed accepted conversation and reversible history.
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil

from filelock import FileLock
from omegaconf import OmegaConf

from scratch.dataset_refresh import run as runtime, per_row, publish_composite


def bound_file(path, sha):
    path = Path(path).resolve(strict=True)
    if runtime.digest(path.read_bytes()) != sha:
        raise ValueError('Changed bound evidence: ' + str(path))
    return path


def validate_evidence(evidence, entry, result_path):
    rows = json.loads(evidence.read_text(encoding='utf-8')).get('rows', [])
    matches = [r for r in rows if r.get('candidate_id') == entry['candidate_id']
               and Path(r.get('result_path', '')).resolve() == result_path]
    if len(matches) != 1:
        raise ValueError('Evidence must identify exactly one reviewed origin row')
    reviewed = matches[0]
    scope = reviewed.get('scope', reviewed.get('coverage'))
    if (reviewed.get('result_sha256') != entry['result_sha256']
            or reviewed.get('exclusion_sha256') != entry['exclusion_sha256']
            or reviewed.get('decision', reviewed.get('recommendation')) != 'restore'
            or scope not in ('full_system_user_reasoning_response_reread',
                             'fresh_full_system_user_reasoning_response')):
        raise ValueError('Evidence does not support full review and restoration of these exact bytes')


def validate_entry(entry, configs):
    root = Path(entry['root']).resolve(strict=True)
    arm, cid = entry['arm'], entry['candidate_id']
    if arm not in publish_composite.ARMS or not re.fullmatch(r't[1-9]_\d+_v\d+', cid):
        raise ValueError('Invalid candidate identity')
    row = root / arm / 'records' / cid
    if row.is_symlink() or row.is_junction() or not row.resolve().is_relative_to(root):
        raise ValueError('Candidate path escapes run root')
    path = bound_file(row / 'result.json', entry['result_sha256'])
    exclusion = bound_file(row / 'independent_exclusion.json', entry['exclusion_sha256'])
    evidence = bound_file(entry['evidence_path'], entry['evidence_sha256'])
    validate_evidence(evidence, entry, path)
    if entry.get('decision') != 'restore' or entry.get('review_scope') != 'full_conversation':
        raise ValueError('Restoration requires an explicit full-conversation decision')
    for key in ('reason', 'decision_owner', 'correction_kind'):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise ValueError('Missing review rationale: ' + key)
    key = (str(root), arm)
    if key not in configs:
        configs[key] = runtime.validate_arm(root, arm)
        per_row.assert_models(configs[key])
    cfg = configs[key]
    original = runtime.load_checkpoint(path)
    note = runtime.load_checkpoint(exclusion)
    if (original['status'] != 'accepted' or note['result_sha256'] != entry['result_sha256']
            or runtime.load_result(path)['status'] != 'rejected'):
        raise ValueError('Only unchanged, originally accepted, currently excluded rows may be reinstated')
    candidate = next((r for r in runtime.read_rows(root / arm / 'candidates.jsonl')
                      if r['candidate_id'] == cid), None)
    if candidate is None or runtime.load_checkpoint(row / 'identity.json') != {
        'candidate_sha256': runtime.digest(candidate), 'config_sha256': runtime.digest(cfg)
    }:
        raise ValueError('Candidate/config identity mismatch')
    if runtime.simple_checks(original['record']) or not runtime.acceptance(original['review'], cfg):
        raise ValueError('Original content fails its frozen mechanical/target gates')
    per_row.verify_accepted(path, original, cfg)
    publish_composite.validate_adoption(path, original, cfg)
    return row, evidence


def archive_and_restore(row, entry, evidence, manifest_sha):
    archive = row / 'reconsidered_exclusions' / manifest_sha
    if archive.exists():
        raise ValueError('Correction archive already exists; inspect it before retrying')
    archive.mkdir(parents=True)
    names = ('independent_exclusion.json', 'independent_exclusion.receipt.json')
    for name in names:
        shutil.copyfile(row / name, archive / name)
        if (archive / name).read_bytes() != (row / name).read_bytes():
            raise ValueError('Exclusion preservation failed')
    shutil.copyfile(evidence, archive / 'review_evidence.json')
    bound_file(archive / 'review_evidence.json', entry['evidence_sha256'])
    runtime.save_checkpoint(archive / 'correction.json', {
        'manifest_sha256': manifest_sha, 'entry': entry,
        'scope': 'Reverses independent exclusion only; original author text and automatic verdict unchanged.',
        'helper_sha256': runtime.digest(Path(__file__).read_bytes()),
    })
    try:
        for name in names:
            (row / name).unlink()
        if (runtime.digest((row / 'result.json').read_bytes()) != entry['result_sha256']
                or runtime.load_result(row / 'result.json')['status'] != 'accepted'):
            raise ValueError('Restoration changed content or failed to expose original verdict')
    except BaseException:
        for name in names:
            shutil.copyfile(archive / name, row / name)
        raise
    return archive


def verify_history(result_path):
    """Revalidate every correction attached to an origin before export; return provenance."""
    result_path = Path(result_path)
    histories = []
    for archive in sorted((result_path.parent / 'reconsidered_exclusions').glob('*')):
        note = runtime.load_checkpoint(archive / 'correction.json')
        entry = note['entry']
        if (note['manifest_sha256'] != archive.name
                or note['helper_sha256'] != runtime.digest(Path(__file__).read_bytes())
                or entry.get('decision') != 'restore'
                or entry.get('review_scope') != 'full_conversation'):
            raise ValueError('Changed correction implementation or decision')
        bound_file(result_path, entry['result_sha256'])
        bound_file(archive / 'independent_exclusion.json', entry['exclusion_sha256'])
        exclusion = runtime.load_checkpoint(archive / 'independent_exclusion.json')
        if exclusion['result_sha256'] != entry['result_sha256']:
            raise ValueError('Historical exclusion bound a different answer')
        evidence = bound_file(archive / 'review_evidence.json', entry['evidence_sha256'])
        validate_evidence(evidence, entry, result_path.resolve())
        histories.append({'manifest_sha256': archive.name,
                          'correction_sha256': runtime.digest((archive / 'correction.json').read_bytes()),
                          'decision_owner': entry['decision_owner'], 'correction_kind': entry['correction_kind'],
                          'reason': entry['reason'], 'remaining_caveat': entry.get('remaining_caveat')})
    return histories


def reconsider(config_path, apply=False):
    config_path = Path(config_path).resolve(strict=True)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    manifest = bound_file(cfg['manifest'], cfg['manifest_sha256'])
    entries = json.loads(manifest.read_text(encoding='utf-8'))['entries']
    if not entries:
        raise ValueError('Empty correction manifest')
    ids = [(str(Path(e['root']).resolve()), e['arm'], e['candidate_id']) for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate correction entry')
    output = Path(cfg['output']).resolve()
    if output.exists() or any(output.is_relative_to(Path(root)) for root, _, _ in ids):
        raise ValueError('Use a new receipt path outside all origins')
    configs, archives = {}, []
    with ExitStack() as locks:
        for root in sorted({key[0] for key in ids}):
            locks.enter_context(FileLock(str(Path(root) / 'execution.lock'), timeout=1))
        validated = [validate_entry(entry, configs) for entry in entries]
        result = {'applied': apply, 'rows': len(entries), 'manifest_sha256': cfg['manifest_sha256'],
                  'config_sha256': runtime.digest(config_path.read_bytes()), 'archives': archives,
                  'train_ready': False, 'mixture_ready': False, 'inference_calls': 0}
        if apply:
            try:
                for entry, (row, evidence) in zip(entries, validated, strict=True):
                    archives.append(str(archive_and_restore(row, entry, evidence, cfg['manifest_sha256'])))
                runtime.save_checkpoint(output, result)
            except BaseException:
                # Roll back every successful prior row, retaining the audit archives.
                for archive_name in archives:
                    archive = Path(archive_name)
                    row = archive.parents[1]
                    for name in ('independent_exclusion.json', 'independent_exclusion.receipt.json'):
                        shutil.copyfile(archive / name, row / name)
                runtime.save_checkpoint(output, {**result, 'applied': False, 'rolled_back': True})
                raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--apply', action='store_true', help='Default is read-only validation')
    args = parser.parse_args()
    print(reconsider(args.config, args.apply))
