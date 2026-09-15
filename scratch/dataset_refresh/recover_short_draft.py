# ABOUTME: Explicit Sonnet final revision from a complete but length-failed untrained draft.
# ABOUTME: Preserves failed inputs, requires source review, and adopts only freshly reviewed hash-bound answers.
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from filelock import FileLock
from scratch.dataset_refresh import run as base
from scratch.dataset_refresh.per_row import assert_models, conversation, grounded, verify_accepted
from scratch.dataset_refresh.recover_missing_changes import parse_complete_answer
from scratch.dataset_refresh.reviewer_probe import validate_verdict

ATTEMPT = 200
AUTHOR = 'short_draft_rewrite_200'
MANIFEST = 'short_draft_recovery_200.json'
PROPOSAL = 'short_draft_candidate_200.json'
SOURCE_REVIEW = 'short_draft_source_review.json'
RECOVERY_INSTRUCTION = ('\n\nThe supplied draft is untrained input and is too short for this dataset. '
    'Write a complete revised answer with at least 700 characters in EACH reasoning and response block. '
    'Add only useful situation-specific deliberation, competing considerations and actionable advice; never pad. '
    'Respect the actual user and system constraints, including concise style where requested. '
    'Keep source facts fixed; do not invent facts, permissions or guarantees. '
    'Do not mention drafting, revision, this instruction, character limits, reviewers or training in either block. '
    'Include the usual changes block separately, outside training text.')


def fields(record, cfg):
    return {**record, 'style_guidance': cfg.get('style_guidance', ''),
            'constitution': cfg['review_constitution_text'], 'metadata_json': '{}', 'eligibility_json': '{}',
            'conversation_json': json.dumps(conversation(record), ensure_ascii=False),
            'record_json': json.dumps(conversation(record), ensure_ascii=False)}


def inputs(root, arm, cid, require_source_review=True):
    root = Path(root).resolve()
    row = (root / arm / 'records' / cid).resolve()
    if not row.is_relative_to(root / arm / 'records') or row.name != cid:
        raise ValueError('Candidate path outside arm')
    cfg = base.validate_arm(root, arm)
    assert_models(cfg)
    hashes = {}

    def checked(path):
        value = base.load_checkpoint(path)
        for p in (path, path.with_suffix('.receipt.json')):
            hashes[str(p)] = base.digest(p.read_bytes())
        return value

    original = checked(row / 'result.json')
    if original.get('status') != 'failed' or original.get('error_type') != 'ValueError':
        raise ValueError('Requires an actual failed draft terminal, never accepted or quality-rejected')
    if original.get('candidate_id') != cid:
        raise ValueError('Terminal identity mismatch')
    if (row / 'independent_exclusion.json').exists():
        raise ValueError('Independent source/content exclusion blocks recovery')
    stages = cfg['response_stages']
    if len(stages) != 2 or [s['name'] for s in stages] != ['draft_responses', 'revise_responses']:
        raise ValueError('Requires exactly the known draft and final author stages')
    if any((row / name).exists() for name in ['revise_responses.json', 'repair_1.json', 'grounding_0.json', 'review_0.json', 'answer_0.json']):
        raise ValueError('Downstream stage exists; not a failed initial draft')
    for stage in stages:
        if stage.get('lint', {}).get('min_chars') != 700 or set(stage['lint'].get('fields', [])) != {'reasoning', 'response'}:
            raise ValueError('Both unchanged author lint specifications must retain the 700 minimum')
    if stages[1]['save'] != {'reasoning': 'reasoning', 'response': 'response', 'rewrite_changes': 'changes'}:
        raise ValueError('Unexpected final author schema')
    record = original['record']
    scenario = checked(row / 'scenario.json')
    if conversation(record, False) != {k: scenario[k] for k in ('system', 'user')}:
        raise ValueError('Source conversation differs from scenario checkpoint')
    if not base.acceptance(checked(row / 'preflight.json'), cfg['preflight']):
        raise ValueError('Source failed normal eligibility')
    if record.get('source_facts') != [record['user']] or record.get('paired_counterfactual') is not False:
        raise ValueError('Derived provenance differs from frozen invariants')
    draft = checked(row / 'draft_responses.json')
    if set(draft) != {'draft_reasoning', 'draft_response'} or any(not isinstance(v, str) or not v.strip() for v in draft.values()):
        raise ValueError('Require complete nonempty draft fields')
    if any(record.get(k) != v for k, v in draft.items()):
        raise ValueError('Terminal draft differs from checkpoint')
    parsed = {'reasoning': draft['draft_reasoning'], 'response': draft['draft_response']}
    lint = base.lint_problems(parsed, stages[0]['lint'], record)
    if not lint or any(not re.fullmatch(r'<(?:reasoning|response)> is \d+ chars, under the 700 minimum', s) for s in lint):
        raise ValueError('Only pure initial-draft minimum-length failures qualify')
    if original.get('error') != 'Response lint: ' + '; '.join(lint):
        raise ValueError('Exact original length lint did not reproduce')
    identity = checked(row / 'identity.json')
    candidates = [c for c in base.read_rows(root / arm / 'candidates.jsonl') if c['candidate_id'] == cid]
    if len(candidates) != 1 or identity != {'candidate_sha256': base.digest(candidates[0]), 'config_sha256': base.digest(cfg)}:
        raise ValueError('Frozen candidate/config identity mismatch')
    meta = checked(root / 'run_meta.json')
    hashes[str(root / arm / 'config.json')] = base.digest((root / arm / 'config.json').read_bytes())
    budget = Path(meta['budget_root'])
    with FileLock(str(budget / 'spend.lock'), timeout=30):
        entries = json.loads((budget / 'spend.json').read_text(encoding='utf-8'))
    calls = [e for e in entries if e.get('run_root') and Path(e['run_root']).resolve() == root
             and e.get('arm') == arm and e.get('candidate_id') == cid and e.get('stage') == 'draft_responses']
    if len(calls) != 1:
        raise ValueError('Exactly one original physical draft call required')
    entry = calls[0]
    cost = entry.get('api_reported_cost_usd')
    if entry.get('status') != 'settled' or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Unknown outcome or billing cannot be recovered')
    raw_path = budget / 'raw_calls' / f"{entry['call_id']:06d}.json"
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    hashes[str(raw_path)] = base.digest(raw_path.read_bytes())
    if raw.get('accounting') != entry or base.digest(raw['request']) != entry['request_sha256']:
        raise ValueError('Raw call/accounting differs from ledger')
    expected = base.request_options(cfg['models'][stages[0]['model']])
    expected['messages'] = [{'role': role, 'content': base.render(stages[0]['prompts'][role], fields(record, cfg))}
                            for role in ('system', 'user')]
    if raw['request'] != expected or expected['model'] != 'anthropic/claude-sonnet-5':
        raise ValueError('Original draft was not the exact frozen Sonnet request')
    if raw.get('response', {}).get('finish_reason') != 'stop' or parse_complete_answer(raw['response']['content']) != parsed:
        raise ValueError('Incomplete or different physical draft')
    source_review = None
    if require_source_review:
        source_review = checked(row / SOURCE_REVIEW)
        if (source_review.get('eligible') is not True or source_review.get('result_sha256') != hashes[str(row / 'result.json')]
                or not isinstance(source_review.get('reason'), str) or not source_review['reason'].strip()):
            raise ValueError('Independent source review must approve this exact failed terminal')
    return root, cfg, row, original, {
        'source_result_sha256': hashes[str(row / 'result.json')], 'bound_file_sha256': hashes,
        'draft_call_id': entry['call_id'], 'draft_request_sha256': entry['request_sha256'],
        'draft_raw_sha256': hashes[str(raw_path)], 'draft_ledger_entry_sha256': base.digest(entry),
        'source_review': source_review, 'original_lint': lint,
        'input_policy': 'Original length-failed draft is preserved as UNTRAINED INPUT only; no final lint is waived.'}


def implementation_hashes():
    return {str(p.resolve()): base.digest(p.read_bytes()) for p in [Path(__file__),
        Path(base.__file__), Path(__file__).with_name('per_row.py'),
        Path(__file__).with_name('reviewer_probe.py'), Path(__file__).with_name('recover_missing_changes.py')]}


def physical_receipt(root, row, name, value, expected_request, tagged=None):
    """Bind each parsed checkpoint to one stopped, settled physical call and its started marker."""
    meta = base.load_checkpoint(root / 'run_meta.json')
    budget = Path(meta['budget_root'])
    with FileLock(str(budget / 'spend.lock'), timeout=30):
        entries = json.loads((budget / 'spend.json').read_text(encoding='utf-8'))
    matches = [e for e in entries if e.get('run_root') and Path(e['run_root']).resolve() == root
               and e.get('arm') == row.parent.parent.name and e.get('candidate_id') == row.name and e.get('stage') == name]
    if len(matches) != 1:
        raise ValueError('Requires exactly one physical recovery call: ' + name)
    entry = matches[0]
    cost = entry.get('api_reported_cost_usd')
    if entry.get('status') != 'settled' or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Recovery call has unknown outcome/billing: ' + name)
    raw_path = budget / 'raw_calls' / f"{entry['call_id']:06d}.json"
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    marker = base.load_checkpoint(row / (name + '.started.json'))
    if (raw.get('accounting') != entry or raw['request'] != expected_request or base.digest(raw['request']) != entry['request_sha256']
            or marker['request_sha256'] != entry['request_sha256']
            or marker['messages_sha256'] != base.digest(raw['request']['messages'])
            or raw.get('response', {}).get('finish_reason') != 'stop'):
        raise ValueError('Recovery physical request/response differs from recorded evidence')
    if tagged:
        parsed = base._parse_tagged(raw['response']['content'], tuple(tagged['tags']))
        observed = {dest: parsed[tag] for dest, tag in tagged['save'].items()}
    else:
        observed = base._parse_json(raw['response']['content'])
    if observed != value:
        raise ValueError('Recovery checkpoint differs from actual physical output')
    return {'call_id': entry['call_id'], 'raw_call_path': str(raw_path),
            'raw_call_sha256': base.digest(raw_path.read_bytes()), 'request_sha256': entry['request_sha256'],
            'ledger_entry_sha256': base.digest(entry), 'checkpoint_sha256': base.digest((row / (name + '.json')).read_bytes())}


def expected_recovery_requests(original_record, final_record, cfg):
    stage = cfg['response_stages'][-1]
    author_messages = [{'role': role, 'content': base.render(stage['prompts'][role], fields(original_record, cfg))}
                       for role in ('system', 'user')]
    author_messages[-1]['content'] += RECOVERY_INSTRUCTION
    requests = {AUTHOR: dict(base.request_options(cfg['models'][stage['model']]), messages=author_messages)}
    if final_record is not None:
        narrow = conversation(final_record); narrow['final'] = narrow.pop('response')
        spec = cfg['grounding_review']
        requests['grounding_200'] = dict(base.request_options(cfg['models'][spec['model']]), messages=[
            {'role': role, 'content': base.render(spec['prompts'][role], {'conversation_json': json.dumps(narrow, ensure_ascii=False)})}
            for role in ('system', 'user')])
        requests['review_200'] = dict(base.request_options(cfg['models']['review']), messages=[
            {'role': role, 'content': base.render(cfg['prompts']['review_' + role], fields(final_record, cfg))}
            for role in ('system', 'user')])
    return requests


def propose(root, arm, cid, ceiling, send=None):
    root, cfg, row, original, binding = inputs(root, arm, cid)
    with FileLock(str(row / 'short_draft_recovery.lock'), timeout=1):
        manifest_path = row / MANIFEST
        if manifest_path.exists():
            manifest = base.load_checkpoint(manifest_path)
            if manifest['binding'] != binding or manifest['implementation_files'] != implementation_hashes():
                raise ValueError('Frozen recovery inputs or implementation changed')
        else:
            if any(row.glob('*_200*')):
                raise ValueError('Attempt200 namespace already occupied')
            manifest = {'attempt': ATTEMPT, 'binding': binding, 'implementation_files': implementation_hashes(),
                'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'authorization': 'Explicit user-authorized iterative row repair; root approved completed short draft as untrained input.',
                'author_stage_mapping': {'input': 'draft_responses', 'actual_final': AUTHOR,
                                         'frozen_template_and_settings': 'revise_responses',
                                         'appended_instruction_sha256': base.digest(RECOVERY_INSTRUCTION)},
                'appended_recovery_instruction': RECOVERY_INSTRUCTION,
                'appended_recovery_instruction_sha256': base.digest(RECOVERY_INSTRUCTION),
                'policy': 'One fresh Sonnet final author call; unchanged final700/lint/local checks, fresh normal grounding and target review; separate independent adoption.'}
            base.save_checkpoint(manifest_path, manifest)
        dest = row / PROPOSAL
        if dest.exists():
            return base.load_checkpoint(dest)
        meta = base.load_checkpoint(root / 'run_meta.json')
        client = base.BudgetClient(meta['budget_root'], ceiling, {m['model'] for m in cfg['models'].values()}, send=send)
        client.local.arm, client.local.candidate_id, client.local.run_root = arm, cid, str(root)
        record = dict(original['record'])

        def call(name, messages, role, tagged=None):
            path = row / (name + '.json')
            if path.exists():
                value = base.load_checkpoint(path)
            else:
                marker = row / (name + '.started.json')
                if marker.exists():
                    raise base.BudgetStop('Stage already started without settled checkpoint: ' + name)
                options = base.request_options(cfg['models'][role])
                base.save_checkpoint(marker, {'messages_sha256': base.digest(messages), 'model_role': role,
                                             'request_sha256': base.digest(dict(options, messages=messages))})
                client.local.stage = name
                raw = client.chat(messages=messages, **options)
                if tagged:
                    parsed = base._parse_tagged(raw.content, tuple(tagged['tags']))
                    value = {dest: parsed[tag] for dest, tag in tagged['save'].items()}
                else:
                    value = base._parse_json(raw.content)
                base.save_checkpoint(path, value)
            evidence = physical_receipt(root, row, name, value,
                dict(base.request_options(cfg['models'][role]), messages=messages), tagged)
            physical_path = row / (name + '.physical.json')
            if physical_path.exists() and base.load_checkpoint(physical_path) != evidence:
                raise ValueError('Physical recovery evidence changed')
            if not physical_path.exists():
                base.save_checkpoint(physical_path, evidence)
            return value

        try:
            stage = cfg['response_stages'][-1]
            messages = expected_recovery_requests(record, None, cfg)[AUTHOR]['messages']
            saved = call(AUTHOR, messages, stage['model'], stage)
            record.update(saved)
            lint = base.lint_problems({tag: saved[dest] for dest, tag in stage['save'].items()}, stage['lint'], record)
            if lint or base.simple_checks(record):
                raise ValueError('Fresh final failed unchanged local checks: ' + str(lint or base.simple_checks(record)))
            narrow = conversation(record); narrow['final'] = narrow.pop('response')
            spec = cfg['grounding_review']
            g = call('grounding_200', [{'role': role, 'content': base.render(spec['prompts'][role],
                {'conversation_json': json.dumps(narrow, ensure_ascii=False)})} for role in ('system', 'user')], spec['model'])
            validate_verdict(g, narrow)
            r = call('review_200', [{'role': role, 'content': base.render(cfg['prompts']['review_' + role], fields(record, cfg))}
                                   for role in ('system', 'user')], 'review')
            base.save_checkpoint(row / 'answer_200.json', conversation(record))
            record.update(response_repair_count=1, short_draft_input_recovery=True)
            candidate = {'candidate_id': cid, 'trait_id': original['trait_id'],
                'status': 'accepted' if grounded(g) and base.acceptance(r, cfg) else 'rejected',
                'record': record, 'review': r, 'grounding_review': g, 'accepted_attempt': ATTEMPT,
                'short_draft_recovery_manifest_sha256': base.digest(manifest_path.read_bytes()),
                'source_result_sha256': binding['source_result_sha256'], 'author_stage_mapping': manifest['author_stage_mapping'],
                'physical_receipt_sha256': {name: base.digest((row / (name + '.physical.json')).read_bytes())
                                          for name in (AUTHOR, 'grounding_200', 'review_200')}}
        except base.BudgetStop:
            raise
        except Exception as exc:
            candidate = {'candidate_id': cid, 'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                         'source_result_sha256': binding['source_result_sha256'], 'record': record}
        base.save_checkpoint(dest, candidate)
        return candidate


def adopt(root, arm, cid, approved_sha, reason):
    if not reason.strip():
        raise ValueError('Independent full-answer audit reason required')
    root = Path(root).resolve()
    with FileLock(str(root / 'execution.lock'), timeout=1):
        root, cfg, row, original, binding = inputs(root, arm, cid)
        with FileLock(str(row / 'short_draft_recovery.lock'), timeout=1):
            manifest = base.load_checkpoint(row / MANIFEST)
            if manifest['binding'] != binding or manifest['implementation_files'] != implementation_hashes():
                raise ValueError('Recovery manifest source or implementation changed')
            if (manifest['appended_recovery_instruction'] != RECOVERY_INSTRUCTION
                    or manifest['appended_recovery_instruction_sha256'] != base.digest(RECOVERY_INSTRUCTION)):
                raise ValueError('Recovery instruction changed')
            path = row / PROPOSAL
            candidate = base.load_checkpoint(path)
            if base.digest(path.read_bytes()) != approved_sha or candidate['status'] != 'accepted':
                raise ValueError('Approval must bind the exact successful proposal')
            if (candidate['source_result_sha256'] != binding['source_result_sha256']
                    or candidate['short_draft_recovery_manifest_sha256'] != base.digest((row / MANIFEST).read_bytes())
                    or conversation(candidate['record'], False) != conversation(original['record'], False)):
                raise ValueError('Proposal changed source or manifest binding')
            stage = cfg['response_stages'][-1]
            saved = base.load_checkpoint(row / (AUTHOR + '.json'))
            if any(candidate['record'].get(k) != v for k, v in saved.items()):
                raise ValueError('Candidate differs from actual author checkpoint')
            expected_requests = expected_recovery_requests(original['record'], candidate['record'], cfg)
            for name in (AUTHOR, 'grounding_200', 'review_200'):
                value = base.load_checkpoint(row / (name + '.json'))
                receipt_path = row / (name + '.physical.json')
                if (candidate['physical_receipt_sha256'][name] != base.digest(receipt_path.read_bytes())
                        or base.load_checkpoint(receipt_path) != physical_receipt(root, row, name, value,
                            expected_requests[name], stage if name == AUTHOR else None)):
                    raise ValueError('Physical recovery chain changed')
            if base.lint_problems({tag: saved[dest] for dest, tag in stage['save'].items()}, stage['lint'], candidate['record']) or base.simple_checks(candidate['record']):
                raise ValueError('Final fails unchanged local checks')
            verify_accepted(path, candidate, cfg)
            archive = row / 'recovered_failures' / 'short_draft_recovery_200'
            if archive.exists():
                raise ValueError('Adoption already started; inspect preserved evidence')
            base.save_checkpoint(row / 'short_draft_adoption_200.json', {
                'candidate_sha256': approved_sha, 'source_result_sha256': binding['source_result_sha256'],
                'manifest_sha256': candidate['short_draft_recovery_manifest_sha256'],
                'independent_audit_reason': reason, 'source_review': binding['source_review']})
            archive.mkdir(parents=True)
            for name in ('result.json', 'result.receipt.json'):
                shutil.copy2(row / name, archive / name)
            base.save_checkpoint(row / 'result.json', candidate)
            return {'candidate_id': cid, 'status': 'adopted', 'result_sha256': base.digest((row / 'result.json').read_bytes())}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['inspect', 'propose', 'adopt'])
    p.add_argument('--root', required=True); p.add_argument('--arm', required=True)
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--ceiling', type=float, default=245); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--approved-sha'); p.add_argument('--reason')
    args = p.parse_args()
    if args.command == 'adopt':
        if len(args.candidates) != 1 or not args.approved_sha or not args.reason:
            p.error('Adoption requires one candidate, approved SHA and independent audit reason')
        print(json.dumps(adopt(args.root, args.arm, args.candidates[0], args.approved_sha, args.reason)))
    else:
        def task(cid):
            if args.command == 'inspect':
                try:
                    *_, binding = inputs(args.root, args.arm, cid, require_source_review=False)
                    result = {'candidate_id': cid, 'status': 'technical_candidate', 'binding': binding}
                except Exception as exc:
                    result = {'candidate_id': cid, 'status': 'refused', 'error': str(exc)}
            else:
                result = propose(args.root, args.arm, cid, args.ceiling)
            print(json.dumps(result if args.command == 'inspect' else {k: result[k] for k in ('candidate_id', 'status', 'error') if k in result}), flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(task, args.candidates))
