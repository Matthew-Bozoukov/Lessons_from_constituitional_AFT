# ABOUTME: Deterministic formatting-only recovery of failed two-string scenario JSON responses.
# ABOUTME: Binds raw-call evidence, refuses ambiguous structure, and retains original failed checkpoints.
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from filelock import FileLock

from scratch.dataset_refresh import run as runtime

SCHEMA_INSTRUCTION = 'Return exactly two keys: system and user, both nonempty strings.'


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def unescaped(text, index):
    count = 0
    while index > 0 and text[index - 1] == '\\':
        index -= 1
        count += 1
    return count % 2 == 0


def decode_string_lexeme(value):
    """Keep literal characters; decode existing valid JSON escapes exactly once."""
    encoded, quote_count, controls, index = [], 0, 0, 0
    while index < len(value):
        char = value[index]
        if char == '\\':
            if index + 1 >= len(value):
                raise ValueError('Dangling backslash is ambiguous')
            following = value[index + 1]
            if following == 'u':
                escape = value[index:index + 6]
                if not re.fullmatch(r'\\u[0-9a-fA-F]{4}', escape):
                    raise ValueError('Invalid Unicode escape; cannot repair without guessing')
                encoded.append(escape)
                index += 6
            elif following in '"\\/bfnrt':
                encoded.append(value[index:index + 2])
                index += 2
            else:
                raise ValueError('Invalid JSON escape; cannot reinterpret a backslash')
            continue
        if char == '"':
            encoded.append('\\"')
            quote_count += 1
        elif ord(char) < 32:
            encoded.append(json.dumps(char)[1:-1])
            controls += 1
        else:
            encoded.append(char)
        index += 1
    decoded = json.loads('"' + ''.join(encoded) + '"')
    # Reject unpaired Unicode surrogates rather than silently replacing text.
    decoded.encode('utf-8')
    return decoded, {'escaped_literal_quotes': quote_count, 'escaped_literal_controls': controls,
                     'source_lexeme_sha256': byte_sha(value.encode('utf-8')),
                     'decoded_string_sha256': byte_sha(decoded.encode('utf-8'))}


def repair_two_string_json(raw):
    if not isinstance(raw, str):
        raise ValueError('Response content must be text')
    text = raw.strip()
    fenced = text.startswith('```')
    if fenced:
        match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*)\r?\n```[ \t]*', text, flags=re.I)
        if not match:
            raise ValueError('Only an exact outer JSON markdown fence may be removed')
        text = match.group(1).strip()
    first = re.match(r'\A\{\s*"(system|user)"\s*:\s*"', text)
    last = re.search(r'"\s*\}\Z', text)
    if not first or not last or not unescaped(text, last.start()):
        raise ValueError('Expected anchored two-string object and unambiguous closing quote')
    property_tokens = [m for m in re.finditer(r'"((?:\\.|[^"\\])*)"\s*:', text)
                       if unescaped(text, m.start())]
    if len(property_tokens) != 2 or {m.group(1) for m in property_tokens} != {'system', 'user'}:
        raise ValueError('Extra or ambiguous key-like text; exactly the two known properties required')
    if any(unescaped(text, m.start()) for m in re.finditer(r'"\s*,\s*[A-Za-z_]\w*\s*:', text)):
        raise ValueError('Unquoted extra property is not an inner-quote formatting repair')
    # Every apparent unescaped property boundary is significant, even inside prose.
    # Extra keys and nested JSON-like quoted examples are refused, not reinterpreted.
    boundaries = []
    pattern = r'"\s*,\s*"(?P<key>(?:\\.|[^"\\])*)"\s*:\s*'
    for match in re.finditer(pattern, text):
        key_quote = text.index('"', match.start() + 1)
        if unescaped(text, match.start()) and unescaped(text, key_quote):
            boundaries.append(match)
    if len(boundaries) != 1:
        raise ValueError('Extra or ambiguous property boundaries; exactly one transition required')
    boundary = boundaries[0]
    other = 'user' if first.group(1) == 'system' else 'system'
    if boundary.group('key') != other or text[boundary.end():boundary.end() + 1] != '"':
        raise ValueError('Second property must be the other known key with a string value')
    if not first.end() <= boundary.start() < boundary.end() < last.start():
        raise ValueError('Overlapping or empty structural boundaries')
    lexemes = {first.group(1): text[first.end():boundary.start()],
               other: text[boundary.end() + 1:last.start()]}
    decoded, changes = {}, {}
    for key, value in lexemes.items():
        decoded[key], changes[key] = decode_string_lexeme(value)
        if not decoded[key].strip():
            raise ValueError('System/user must be nonempty strings')
    return decoded, {'removed_outer_markdown_fence': fenced, 'fields': changes,
                     'decoded_payload_sha256': runtime.digest(decoded),
                     'interpretation': 'Existing valid JSON escapes decoded exactly once; unescaped inner quotes and controls preserved literally. No prose edits.'}


def within(path, directory):
    resolved, parent = Path(path).resolve(), Path(directory).resolve()
    if not resolved.is_relative_to(parent) or resolved == parent:
        raise ValueError('Recovery path escapes intended directory: ' + str(path))
    return resolved


def plan_row(root, terminal, entries):
    root, terminal = Path(root).resolve(), Path(terminal).resolve()
    row = within(terminal.parent, root)
    result = runtime.load_checkpoint(terminal)
    if result.get('status') != 'failed' or result.get('error_type') != 'JSONDecodeError':
        raise ValueError('Only a terminal JSONDecodeError is eligible, never a quality rejection')
    arm, cid = row.parent.parent.name, row.name
    if row.parent.name != 'records' or result.get('candidate_id') != cid:
        raise ValueError('Unexpected row layout/identity')
    allowed = {'identity.json', 'identity.receipt.json', 'result.json', 'result.receipt.json'}
    if {p.name for p in row.iterdir()} != allowed:
        raise ValueError('Row already has scenario/downstream/recovery artifacts; manual review required')
    cfg = json.loads((root / arm / 'config.json').read_text(encoding='utf-8'))
    identity = runtime.load_checkpoint(row / 'identity.json')
    if identity.get('config_sha256') != runtime.digest(cfg) or SCHEMA_INSTRUCTION not in cfg['prompts']['scenario_user']:
        raise ValueError('Frozen config does not establish this exact two-string schema')
    matches = [entry for entry in entries if entry.get('run_root') and Path(entry['run_root']).resolve() == root
               and entry.get('arm') == arm and entry.get('candidate_id') == cid and entry.get('stage') == 'scenario']
    if len(matches) != 1:
        raise ValueError('Exactly one matching physical scenario call is required')
    entry = matches[0]
    if entry.get('status') != 'settled' or type(entry.get('call_id')) is not int:
        raise ValueError('Unknown billing/provider outcome is not a completed formatting failure')
    metadata = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    budget = Path(metadata.get('budget_root', root / 'budget')).resolve()
    raw_path = within(budget / 'raw_calls' / f'{entry["call_id"]:06d}.json', budget)
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get('accounting') != entry or runtime.digest(raw['request']) != entry['request_sha256']:
        raise ValueError('Raw request/accounting does not match ledger')
    response = raw.get('response', {})
    if response.get('finish_reason') != 'stop':
        raise ValueError('Incomplete or filtered response cannot be recovered')
    try:
        runtime._parse_json(response['content'])
    except json.JSONDecodeError as exc:
        if str(exc) != result.get('error'):
            raise ValueError('Original parser error does not match this response') from exc
    else:
        raise ValueError('Original parser does not reproduce the saved JSONDecodeError')
    payload, changes = repair_two_string_json(response['content'])
    return {'arm': arm, 'candidate_id': cid, 'row': str(row), 'payload': payload,
            'raw_call_path': str(raw_path), 'raw_call_sha256': byte_sha(raw_bytes),
            'request_sha256': entry['request_sha256'], 'call_id': entry['call_id'],
            'ledger_entry_sha256': runtime.digest(entry), 'original_content_sha256': byte_sha(response['content'].encode('utf-8')),
            'original_terminal_sha256': byte_sha(terminal.read_bytes()),
            'original_terminal_receipt_sha256': byte_sha(terminal.with_suffix('.receipt.json').read_bytes()),
            'identity_sha256': byte_sha((row / 'identity.json').read_bytes()),
            'format_recovery': changes}


def apply_plan(plan):
    row = Path(plan['row']).resolve()
    terminal = within(row / 'result.json', row)
    receipt = within(row / 'result.receipt.json', row)
    archive = within(row / 'recovered_failures' / 'scenario_json_0', row)
    if archive.exists() or (row / 'scenario.json').exists() or (row / 'scenario_recovery.json').exists():
        raise ValueError('Recovery already started; refusing to overwrite evidence')
    if byte_sha(terminal.read_bytes()) != plan['original_terminal_sha256'] or byte_sha(receipt.read_bytes()) != plan['original_terminal_receipt_sha256']:
        raise ValueError('Terminal changed after planning')
    if byte_sha(Path(plan['raw_call_path']).read_bytes()) != plan['raw_call_sha256']:
        raise ValueError('Raw call changed after planning')
    scenario = within(row / 'scenario.json', row)
    recovery = within(row / 'scenario_recovery.json', row)
    # Write the resumable checkpoint while the original terminal still blocks execution.
    # A crash cannot remove the terminal before the recovered checkpoint is durable.
    runtime.save_checkpoint(scenario, plan['payload'])
    runtime.save_checkpoint(recovery, {**plan, 'method': 'lossless_two_string_json_format_recovery',
                                       'archived_failure_directory': str(archive)})
    archive.mkdir(parents=True)
    terminal.rename(within(archive / 'result.json', row))
    receipt.rename(within(archive / 'result.receipt.json', row))
    runtime.load_checkpoint(archive / 'result.json')
    if runtime.load_checkpoint(scenario) != plan['payload']:
        raise ValueError('Recovered scenario changed on disk')


def inspect(root, apply=False):
    root = Path(root).resolve()
    metadata = json.loads((root / 'run_meta.json').read_text(encoding='utf-8'))
    budget = Path(metadata.get('budget_root', root / 'budget')).resolve()
    entries = json.loads((budget / 'spend.json').read_text(encoding='utf-8'))
    result = {'root': str(root), 'mode': 'apply' if apply else 'dry_run', 'recoverable': [], 'refused': []}
    for terminal in sorted(root.glob('*/records/*/result.json')):
        saved = runtime.load_checkpoint(terminal)
        if saved.get('status') != 'failed' or saved.get('error_type') != 'JSONDecodeError':
            continue
        try:
            plan = plan_row(root, terminal, entries)
            if apply:
                apply_plan(plan)
            result['recoverable'].append(plan)
        except (ValueError, runtime.BudgetStop) as exc:
            result['refused'].append({'path': str(terminal), 'reason': str(exc)})
    return result


def recover(root, apply=False):
    if not apply:
        return inspect(root)
    # The production executor holds this same lock. Do not edit a running batch.
    with FileLock(str(Path(root) / 'execution.lock'), timeout=1):
        return inspect(root, apply=True)


def main():
    parser = argparse.ArgumentParser(description='Lossless two-key JSON recovery; default only inspects.')
    parser.add_argument('--root', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.apply), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
