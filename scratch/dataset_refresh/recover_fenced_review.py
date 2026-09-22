# ABOUTME: Recover one already billed complete target review from its unique terminal JSON fence, without inference.
# ABOUTME: Preserve failed evidence, reproduce cached stages offline, and bind reversible recovery history for export.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace

from filelock import FileLock
from omegaconf import OmegaConf

from scratch.dataset_refresh import run, per_row
from scratch.dataset_refresh.billing_evidence import validate_billing_call

ARM = 'nonmoral-advice'
CANDIDATE = 't2_007_v1'
CALL_ID = 11275
HISTORY = 'fenced_review_recovery_11275.json'
ARCHIVE = 'recovered_failures/fenced_review_11275'
MARKER = 'fenced_review_recovery_manifest_sha256'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    return run.digest(Path(path).read_bytes())


def bound(path, expected):
    path = Path(path).resolve(strict=True)
    if file_sha(path) != expected:
        raise ValueError('Changed bound file: ' + str(path))
    return path


def extract_final_json(content):
    if not isinstance(content, str) or content.count('```') != 2:
        raise ValueError('Require exactly one JSON fence')
    match = re.fullmatch(r'(?P<preamble>.*?)```json\r?\n(?P<json>.*?)\r?\n```\s*', content, re.S)
    if not match or any(c in match['preamble'] for c in '{}'):
        raise ValueError('Require terminal JSON block with prose-only preamble and no trailing prose')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError('Nonfinite JSON: ' + value)

    value = json.loads(match['json'], object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError('Review must be an object')
    return value, {'content': content, 'preamble': match['preamble'],
                   'json_text': match['json'], 'json_span': list(match.span('json'))}


def expected_request(record, cfg):
    conversation = json.dumps(per_row.conversation(record), ensure_ascii=False)
    fields = dict(record, style_guidance=cfg.get('style_guidance', ''),
                  conversation_json=conversation, record_json=conversation,
                  metadata_json='{}', eligibility_json='{}', constitution=cfg['review_constitution_text'])
    return {'messages': [{'role': role, 'content': run.render(cfg['prompts']['review_' + role], fields)}
                         for role in ('system', 'user')], **run.request_options(cfg['models']['review'])}


class NoNetworkClient:
    def __init__(self):
        self.local = SimpleNamespace()
        self.attempted_calls = 0

    def chat(self, **kwargs):
        self.attempted_calls += 1
        raise run.BudgetStop('No network is permitted during formatting recovery')


def inventory(row):
    values = {}
    for path in row.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError('No linked row evidence')
        if path.is_file():
            values[path.relative_to(row).as_posix()] = file_sha(path)
    return values


def validate_call(raw, call, budget, root, record, cfg):
    if (call.get('call_id') != CALL_ID or call.get('stage') != 'review_0'
            or call.get('candidate_id') != CANDIDATE or call.get('arm') != ARM
            or Path(call.get('run_root', '')).resolve() != root.resolve()
            or call.get('model') != cfg['models']['review']['model']
            or call.get('status') not in ('settled', 'billing_verified_failure')):
        raise ValueError('Wrong scoped/billed review call')
    validate_billing_call(budget, call, raw)
    request = expected_request(record, cfg)
    response = raw.get('response', {})
    if (raw.get('request') != request or call.get('request_sha256') != run.digest(request)
            or response.get('finish_reason') != 'stop' or response.get('tool_calls')
            or response.get('response_model') != request['model']):
        raise ValueError('Review request/output does not exactly match the frozen conversation')
    review, extraction = extract_final_json(response['content'])
    if not run.acceptance(review, cfg) or review.get('issues') != []:
        raise ValueError('Extracted verdict does not pass unchanged target gates')
    return review, extraction


def replay_cached(root, row, cfg, candidate, original, review):
    with tempfile.TemporaryDirectory(prefix='fenced-review-preview-') as temp:
        staged_root = Path(temp)
        staged_arm = staged_root / ARM
        staged_arm.mkdir()
        shutil.copyfile(root / ARM / 'config.json', staged_arm / 'config.json')
        staged_row = staged_arm / 'records' / CANDIDATE
        shutil.copytree(row, staged_row)
        for name in ('result.json', 'result.receipt.json'):
            (staged_row / name).unlink()
        run.save_checkpoint(staged_row / 'review_0.json', review)
        client = NoNetworkClient()
        result = per_row.generate_one(staged_root, ARM, candidate, client)
        if client.attempted_calls or result.get('status') != 'accepted' or result.get('accepted_attempt') != 0:
            raise ValueError('Cached replay did not complete with zero calls and original acceptance gates')
        if result['record'] != dict(original['record'], response_repair_count=0):
            raise ValueError('Cached replay changed original record beyond derived repair count')
        per_row.verify_accepted(staged_row / 'result.json', result, cfg)
        before = inventory(row)
        after = inventory(staged_row)
        allowed = {'result.json', 'result.receipt.json', 'review_0.json', 'review_0.receipt.json',
                   'answer_0.json', 'answer_0.receipt.json'}
        if any(after.get(name) != digest for name, digest in before.items() if name not in allowed):
            raise ValueError('Replay changed existing source, author or stage evidence')
        if set(after) - set(before) - allowed:
            raise ValueError('Replay produced unexpected stages')
        return result, per_row.conversation(result['record'])


def require_pristine_destination(row):
    names = ['review_0.json', 'answer_0.json', HISTORY, 'independent_exclusion.json']
    names += [Path(name).with_suffix('.receipt.json').name for name in names]
    if any((row / name).exists() for name in [*names, ARCHIVE]):
        raise ValueError('Unexpected completed review, orphan receipt, exclusion or recovery history')


def prepare(cfg):
    root = Path(cfg['root']).resolve(strict=True)
    budget = Path(cfg['budget_root']).resolve(strict=True)
    if (root.name != '2026-09-15_dataset_refresh_sonnet_qualified'
            or cfg.get('arm') != ARM or cfg.get('candidate_id') != CANDIDATE or cfg.get('call_id') != CALL_ID):
        raise ValueError('Helper is authorized for one exact origin and raw call only')
    row = root / ARM / 'records' / CANDIDATE
    recipe = run.validate_arm(root, ARM)
    per_row.assert_models(recipe)
    if recipe.get('per_row_regime') is not True:
        raise ValueError('Per-row frozen regime required')
    original_path = bound(row / 'result.json', cfg['original_result_sha256'])
    original = run.load_checkpoint(original_path)
    if (original.get('status') != 'failed' or original.get('error_type') != 'JSONDecodeError'
            or original.get('error') != 'Expecting value: line 1 column 2 (char 1)'
            or original.get('candidate_id') != CANDIDATE or original.get('trait_id') != 't2'):
        raise ValueError('Only the original target-review JSON parse failure is eligible')
    require_pristine_destination(row)
    candidate = [x for x in run.read_rows(root / ARM / 'candidates.jsonl') if x['candidate_id'] == CANDIDATE]
    if len(candidate) != 1:
        raise ValueError('Candidate identity not unique')
    candidate = candidate[0]
    if run.load_checkpoint(row / 'identity.json') != {
            'candidate_sha256': run.digest(candidate), 'config_sha256': run.digest(recipe)}:
        raise ValueError('Candidate identity changed')
    raw_path = bound(budget / 'raw_calls' / f'{CALL_ID:06d}.json', cfg['raw_call_sha256'])
    ledger = read(budget / 'spend.json')
    calls = [c for c in ledger if c.get('call_id') == CALL_ID]
    if len(calls) != 1 or run.digest(calls[0]) != cfg['ledger_entry_sha256']:
        raise ValueError('Ledger call is missing, duplicate or changed')
    review, extraction = validate_call(read(raw_path), calls[0], budget, root, original['record'], recipe)
    # Any missing cached stage would cause the offline replay to raise, never to query a model.
    result, answer = replay_cached(root, row, recipe, candidate, original, review)
    return {'root': root, 'row': row, 'cfg': recipe, 'original': original, 'review': review,
            'answer': answer, 'result': result, 'extraction': extraction, 'call': calls[0],
            'raw_path': raw_path, 'budget': budget, 'candidate': candidate,
            'bound_files': inventory(row)}


def validate_independent_evidence(cfg, prepared):
    path = bound(cfg['review_evidence_path'], cfg['review_evidence_sha256'])
    value = read(path)
    matches = [r for r in value.get('rows', []) if r.get('candidate_id') == CANDIDATE
               and Path(r.get('result_path', '')).resolve() == prepared['row'] / 'result.json']
    if len(matches) != 1:
        raise ValueError('Require exactly one independent full-conversation review')
    r = matches[0]
    approved = (r.get('decision') == 'recover_formatting'
                and r.get('scope') == 'full_system_user_reasoning_response_reread') or (
                    r.get('decision') == 'restore' and r.get('review_scope') == 'full_conversation'
                    and r.get('decision_owner') == 'root')
    if (r.get('result_sha256') != cfg['original_result_sha256'] or not approved
            or not str(r.get('reason', '')).strip()):
        raise ValueError('Independent evidence does not approve exact unchanged conversation')
    return path


def apply_prepared(prepared, cfg, evidence, output):
    row = prepared['row']
    before = prepared['bound_files']
    if inventory(row) != before:
        raise ValueError('Origin changed after preview')
    archive = row / ARCHIVE
    archive.mkdir(parents=True)
    for name in ('result.json', 'result.receipt.json'):
        shutil.copyfile(row / name, archive / name)
        if file_sha(archive / name) != before[name]:
            raise ValueError('Failed terminal preservation failed')
    shutil.copyfile(prepared['raw_path'], archive / 'raw_call_011275.json')
    bound(archive / 'raw_call_011275.json', cfg['raw_call_sha256'])
    shutil.copyfile(evidence, archive / 'independent_review.json')
    bound(archive / 'independent_review.json', cfg['review_evidence_sha256'])
    run.save_checkpoint(archive / 'ledger_call.json', prepared['call'])
    history = {'operation': 'lossless_terminal_fenced_json_review_recovery', 'config': cfg,
               'implementation_sha256': file_sha(__file__), 'recipe_sha256': run.digest(prepared['cfg']),
               'bound_files': before, 'extraction': prepared['extraction'], 'inference_calls': 0,
               'expected_result': prepared['result'], 'independent_review_sha256': cfg['review_evidence_sha256']}
    history_path = row / HISTORY
    allowed = ['review_0.json', 'review_0.receipt.json', 'answer_0.json', 'answer_0.receipt.json',
               HISTORY, Path(HISTORY).with_suffix('.receipt.json').name]
    try:
        run.save_checkpoint(history_path, history)
        run.save_checkpoint(row / 'review_0.json', prepared['review'])
        run.save_checkpoint(row / 'answer_0.json', prepared['answer'])
        result = dict(prepared['result'], **{MARKER: file_sha(history_path)})
        run.save_checkpoint(row / 'result.json', result)
        verify_history(row / 'result.json', result, prepared['cfg'])
        report = {'applied': True, 'candidate_id': CANDIDATE, 'inference_calls': 0,
                  'result_sha256': file_sha(row / 'result.json'), 'history_sha256': file_sha(history_path)}
        run.save_checkpoint(output, report)
        return report
    except BaseException:
        for name in ('result.json', 'result.receipt.json'):
            shutil.copyfile(archive / name, row / name)
        for name in allowed:
            path = row / name
            if path.exists():
                path.unlink()
        if any(file_sha(row / name) != digest for name, digest in before.items()):
            raise RuntimeError('Rollback did not restore original evidence')
        run.save_checkpoint(output, {'applied': False, 'rolled_back': True, 'inference_calls': 0})
        raise


def verify_history(path, result, cfg=None):
    path = Path(path).resolve()
    row, root = path.parent, path.parent.parents[2]
    cfg = run.validate_arm(root, ARM) if cfg is None else cfg
    history = run.load_checkpoint(row / HISTORY)
    settings = history['config']
    archive = row / ARCHIVE
    original = run.load_checkpoint(archive / 'result.json')
    if (row.name != CANDIDATE or row.parent.parent.name != ARM
            or result.get(MARKER) != file_sha(row / HISTORY)
            or history['implementation_sha256'] != file_sha(__file__)
            or history['recipe_sha256'] != run.digest(cfg)
            or history['inference_calls'] != 0
            or original.get('status') != 'failed' or original.get('error_type') != 'JSONDecodeError'
            or file_sha(archive / 'result.json') != settings['original_result_sha256']
            or {k: v for k, v in result.items() if k != MARKER} != history['expected_result']
            or result['record'] != dict(original['record'], response_repair_count=0)):
        raise ValueError('Formatting recovery does not bind original and unchanged accepted result')
    for name, digest in history['bound_files'].items():
        source = archive / name if name in ('result.json', 'result.receipt.json') else row / name
        bound(source, digest)
    raw_path = bound(archive / 'raw_call_011275.json', settings['raw_call_sha256'])
    call = run.load_checkpoint(archive / 'ledger_call.json')
    if run.digest(call) != settings['ledger_entry_sha256']:
        raise ValueError('Archived billing call changed')
    review, extraction = validate_call(read(raw_path), call, Path(settings['budget_root']),
                                       Path(settings['root']), original['record'], cfg)
    if (extraction != history['extraction'] or review != run.load_checkpoint(row / 'review_0.json')
            or review != result['review']):
        raise ValueError('Recovered review is not the lossless extracted original verdict')
    bound(archive / 'independent_review.json', settings['review_evidence_sha256'])
    evidence_cfg = dict(settings, review_evidence_path=str(archive / 'independent_review.json'))
    validate_independent_evidence(evidence_cfg, {'row': Path(settings['root']) / ARM / 'records' / CANDIDATE})
    per_row.verify_accepted(path, result, cfg)
    return True


def recover(config_path, apply=False):
    config_path = Path(config_path).resolve(strict=True)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    root = Path(cfg['root']).resolve(strict=True)
    output = Path(cfg['output']).resolve()
    if output.exists() or output.with_suffix('.receipt.json').exists() or output.is_relative_to(root):
        raise ValueError('Use a new output receipt outside the frozen origin')
    with FileLock(str(root / 'execution.lock'), timeout=1):
        prepared = prepare(cfg)
        if apply:
            evidence = validate_independent_evidence(cfg, prepared)
            return apply_prepared(prepared, cfg, evidence, output)
        evidence = validate_independent_evidence(cfg, prepared) if cfg.get('review_evidence_path') else None
        return {'applied': False, 'preview_passed': True, 'candidate_id': CANDIDATE,
                'original_result_sha256': cfg['original_result_sha256'], 'inference_calls': 0,
                'review_sha256': run.digest(prepared['review']),
                'independent_evidence_required_for_apply': True,
                'independent_evidence_validated': evidence is not None,
                'independent_evidence_sha256': cfg.get('review_evidence_sha256') if evidence else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recover(args.config, args.apply), indent=2))
