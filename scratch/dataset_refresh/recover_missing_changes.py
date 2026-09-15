# ABOUTME: Recover complete author answers missing only the nontraining changes audit block.
# ABOUTME: Preserves original failures and raw evidence; adds no prose and leaves normal lint/review mandatory.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from filelock import FileLock

from scratch.dataset_refresh import run as runtime
from scratch.dataset_refresh import per_row
from scratch.dataset_refresh.recover_scenario_json import byte_sha, within
from scratch.dataset_refresh.retry_technical_review import checked

ERROR = 'missing <changes> block'


def parse_complete_answer(text):
    """Accept exactly one closed reasoning and response block; use ordinary parser whitespace semantics."""
    if not isinstance(text, str) or re.search(r'</?changes\b', text, re.I):
        raise ValueError('Audit tag must be entirely absent, not partial or malformed')
    body = text.strip()
    if body.startswith('```'):
        fence = re.fullmatch(r'```(?:xml)?[ \t]*\r?\n([\s\S]*)\r?\n```[ \t]*', body, re.I)
        if not fence:
            raise ValueError('Unrecognized outer fence or extraneous text')
        body = fence.group(1).strip()
    tokens = re.findall(r'</?(?:reasoning|response)\b[^>]*>', body, re.I)
    allowed = [['<reasoning>', '</reasoning>', '<response>', '</response>'],
               ['<response>', '</response>', '<reasoning>', '</reasoning>']]
    if tokens not in allowed:
        raise ValueError('Require exactly two complete unambiguous training blocks')
    pattern = r'\A\s*<(reasoning|response)>(.*?)</\1>\s*<(reasoning|response)>(.*?)</\3>\s*\Z'
    match = re.fullmatch(pattern, body, re.S)
    if not match or match.group(1) == match.group(3):
        raise ValueError('Extraneous output or ambiguous block boundaries')
    parsed = runtime._parse_tagged(text, ('reasoning', 'response'))
    if not all(parsed[k] for k in ('reasoning', 'response')):
        raise ValueError('Training blocks must be nonempty')
    return parsed


def plan_row(root, terminal, entries):
    root, terminal = Path(root).resolve(), Path(terminal).resolve()
    row = within(terminal.parent, root)
    hashes = {}
    failed = checked(terminal, hashes)
    cid, arm = row.name, row.parent.parent.name
    if row.parent.name != 'records' or failed.get('candidate_id') != cid:
        raise ValueError('Unexpected candidate layout or identity')
    if failed.get('status') != 'failed' or failed.get('error_type') != 'ValueError' or failed.get('error') != ERROR:
        raise ValueError('Only the exact missing changes parser failure is eligible')
    external = terminal.name == 'independent_candidate_100.json'
    if terminal.name not in ('result.json', 'independent_candidate_100.json'):
        raise ValueError('Unsupported failed terminal')
    cfg = runtime.validate_arm(root, arm)
    per_row.assert_models(cfg)
    cfg_path = root / arm / 'config.json'
    hashes[str(cfg_path)] = byte_sha(cfg_path.read_bytes())
    identity = checked(row / 'identity.json', hashes)
    candidates = runtime.read_rows(root / arm / 'candidates.jsonl')
    candidates = [c for c in candidates if c['candidate_id'] == cid]
    if len(candidates) != 1 or identity != {'candidate_sha256': runtime.digest(candidates[0]), 'config_sha256': runtime.digest(cfg)}:
        raise ValueError('Candidate identity differs from frozen inputs')
    meta = checked(root / 'run_meta.json', hashes)
    calls = [e for e in entries if e.get('run_root') and Path(e['run_root']).resolve() == root
             and e.get('arm') == arm and e.get('candidate_id') == cid]
    if not calls:
        raise ValueError('Missing physical author call')
    entry = max(calls, key=lambda e: e['call_id'])
    stage = entry['stage']
    if sum(e['stage'] == stage for e in calls) != 1:
        raise ValueError('Require exactly one physical call for the failed stage')
    if external:
        if stage != 'independent_rewrite_100':
            raise ValueError('External candidate failure must be the independent rewrite stage')
        spec = cfg['response_stages'][-1]
        original = checked(row / 'result.json', hashes)
        exclusion = checked(row / 'independent_exclusion.json', hashes)
        manifest = checked(row / 'independent_repair_100.json', hashes)
        marker = checked(row / (stage + '.started.json'), hashes)
        original_sha = runtime.digest((row / 'result.json').read_bytes())
        if (original['status'] != 'accepted' or exclusion['result_sha256'] != original_sha
                or manifest['source_result_sha256'] != original_sha or manifest['exclusion'] != exclusion
                or failed['source_result_sha256'] != original_sha
                or manifest['script_sha256'] != runtime.digest(Path(__file__).with_name('repair_independent.py').read_bytes())):
            raise ValueError('External repair no longer binds its excluded original and implementation')
        if per_row.conversation(failed['record']) != per_row.conversation(original['record']):
            raise ValueError('Failed external repair changed the original before parsing')
    else:
        if (row / 'independent_exclusion.json').exists():
            raise ValueError('An excluded substantive terminal cannot be format-recovered')
        matching = [s for s in cfg['response_stages'] if s['name'] == stage]
        repair = re.fullmatch(r'repair_(\d+)', stage)
        if matching:
            if len(matching) != 1:
                raise ValueError('Ambiguous configured stage')
            spec = matching[0]
        elif repair and 1 <= int(repair.group(1)) <= cfg.get('max_response_repairs', 1):
            spec = cfg['response_stages'][-1]
        else:
            raise ValueError('Failure is not a configured author rewrite stage')
    if spec['tags'] != ['reasoning', 'response', 'changes'] or spec['save'].get('rewrite_changes') != 'changes' or set(spec['save'].values()) != {'reasoning', 'response', 'changes'}:
        raise ValueError('Frozen schema must identify changes only as nontraining rewrite_changes')
    if not runtime.acceptance(checked(row / 'preflight.json', hashes), cfg['preflight']):
        raise ValueError('Ineligible scenario cannot be rescued by output formatting')
    checkpoint = row / (stage + '.json')
    note = row / (stage + '_changes_recovery.json')
    archive = within(row / 'recovered_failures' / ('missing_changes_' + stage + '_0'), row)
    if checkpoint.exists() or checkpoint.with_suffix('.receipt.json').exists() or note.exists() or archive.exists():
        raise ValueError('Recovery already started or author checkpoint already exists')
    cost = entry.get('api_reported_cost_usd')
    if entry.get('status') != 'settled' or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Only a settled known-cost physical response qualifies')
    budget = Path(meta['budget_root']).resolve()
    raw_path = within(budget / 'raw_calls' / f'{entry["call_id"]:06d}.json', budget)
    raw = json.loads(raw_path.read_bytes())
    hashes[str(raw_path)] = byte_sha(raw_path.read_bytes())
    if raw.get('accounting') != entry or runtime.digest(raw['request']) != entry['request_sha256']:
        raise ValueError('Raw request/accounting does not match ledger')
    expected_options = runtime.request_options(cfg['models'][spec['model']])
    if expected_options != {k: v for k, v in raw['request'].items() if k != 'messages'} or raw['request']['model'] != 'anthropic/claude-sonnet-5':
        raise ValueError('Physical call does not use the frozen Sonnet author settings')
    if external and (marker['messages_sha256'] != runtime.digest(raw['request']['messages']) or marker['model'] != spec['model']):
        raise ValueError('External started marker does not bind the exact physical request')
    response = raw.get('response', {})
    if response.get('finish_reason') != 'stop':
        raise ValueError('Never recover truncated or filtered author output')
    try:
        runtime._parse_tagged(response['content'], tuple(spec['tags']))
    except ValueError as exc:
        if str(exc) != ERROR:
            raise ValueError('Exact missing changes parser failure not reproduced') from exc
    else:
        raise ValueError('Original parser did not fail on the missing audit block')
    parsed = parse_complete_answer(response['content'])
    payload = {dest: ('' if tag == 'changes' else parsed[tag]) for dest, tag in spec['save'].items()}
    # Bind every other stage receipt that will be reused after the failed terminal is archived.
    for path in row.glob('*.json'):
        if path.name.endswith('.receipt.json'):
            continue
        checked(path, hashes)
    return {'root': str(root), 'row': str(row), 'arm': arm, 'candidate_id': cid, 'stage': stage,
            'external_repair': external, 'terminal_name': terminal.name, 'call_id': entry['call_id'],
            'request_sha256': entry['request_sha256'], 'raw_call_path': str(raw_path),
            'raw_call_sha256': hashes[str(raw_path)], 'ledger_entry_sha256': runtime.digest(entry),
            'original_content_sha256': byte_sha(response['content'].encode('utf-8')),
            'payload': payload, 'parsed_training_text_sha256': runtime.digest(parsed),
            'bound_file_sha256': hashes, 'archive': str(archive),
            'method': 'Recover complete training blocks; missing nontraining rewrite_changes is empty, not an invented author explanation.',
            'next_step': 'Resume the unchanged frozen pipeline; all existing lint, quality and grounding checks still apply.'}


def apply_plan(plan):
    row = Path(plan['row']).resolve()
    checkpoint = within(row / (plan['stage'] + '.json'), row)
    note = within(row / (plan['stage'] + '_changes_recovery.json'), row)
    archive = within(plan['archive'], row)
    if checkpoint.exists() or note.exists() or archive.exists():
        raise ValueError('Recovery already started')
    for name, expected in plan['bound_file_sha256'].items():
        if byte_sha(Path(name).read_bytes()) != expected:
            raise ValueError('Bound evidence changed since planning: ' + name)
    runtime.save_checkpoint(checkpoint, plan['payload'])
    runtime.save_checkpoint(note, {**plan, 'audit_tag_absent': True, 'author_explanation_synthesized': False})
    archive.mkdir(parents=True)
    terminal = within(row / plan['terminal_name'], row)
    for path in (terminal.with_suffix('.receipt.json'), terminal):
        path.rename(within(archive / path.name, row))
    runtime.load_checkpoint(archive / plan['terminal_name'])
    if runtime.load_checkpoint(checkpoint) != plan['payload']:
        raise ValueError('Recovered author payload changed')


def inspect(root, apply=False):
    root = Path(root).resolve()
    meta = runtime.load_checkpoint(root / 'run_meta.json')
    entries = json.loads((Path(meta['budget_root']) / 'spend.json').read_text(encoding='utf-8'))
    report = {'root': str(root), 'mode': 'apply' if apply else 'dry_run', 'recoverable': [], 'refused': []}
    terminals = sorted(list(root.glob('*/records/*/result.json')) + list(root.glob('*/records/*/independent_candidate_100.json')))
    for terminal in terminals:
        saved = runtime.load_checkpoint(terminal)
        if saved.get('status') != 'failed' or saved.get('error') != ERROR:
            continue
        try:
            if apply:
                with FileLock(str(terminal.parent / 'independent_repair.lock'), timeout=1):
                    plan = plan_row(root, terminal, entries)
                    apply_plan(plan)
            else:
                plan = plan_row(root, terminal, entries)
            report['recoverable'].append(plan)
        except (ValueError, KeyError, FileNotFoundError, runtime.BudgetStop) as exc:
            report['refused'].append({'path': str(terminal), 'reason': str(exc)})
    return report


def recover(root, apply=False):
    if not apply:
        return inspect(root)
    with FileLock(str(Path(root) / 'execution.lock'), timeout=1):
        return inspect(root, apply=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.apply), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
