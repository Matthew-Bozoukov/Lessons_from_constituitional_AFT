# ABOUTME: Audits terminal limit hits in four September ODCV conditions and October control.
# ABOUTME: Reads saved transcripts and sampling ledgers offline; never executes recorded actions.
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer

old = Path(r'C:\Users\nikak\.codex\worktrees\gpt-oss-120b-exploration\teaching_claude_why_replication\output\gptoss_control')
out = Path('output/gptoss_swe_analysis/odcv_limit_comparison')
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained('openai/gpt-oss-120b', revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a', local_files_only=True)
runs = []
for arm, campaign in [('september_control', 'control_refresh_2026-09-30'), ('september_base', 'base_odcv_2026-09-30')]:
    for mode in ['original', 'fixed']:
        roots = [p for p in (old / campaign / f'odcv_{mode}').iterdir() if (p / 'rollouts').is_dir()]
        assert len(roots) == 1, roots
        runs.append((f'{arm}_{mode}', roots[0], roots[0] / 'metadata/tinker_shim/sampling.jsonl'))
october = Path('output/gptoss_swe_analysis/teammate_odcv')
runs.append(('october_control_fixed', october, october / 'sampling.jsonl'))
all_runs = []
for name, root, ledger_path in runs:
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
    reservations = {x['id']: x for x in ledger if x['event'] == 'reserved'}
    # Decode only length-ended completions; every truncated transcript must match exactly one.
    completed = [x for x in ledger if x['event'] == 'completed']
    decoded = None
    records = []
    hashes = {}
    for path in sorted((root / 'rollouts').rglob('messages_record.txt')):
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        relative = path.relative_to(root).as_posix()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        blocks = [b for b in re.split(r'^== Step \d+ ==\s*$', text, flags=re.M) if re.search(r'^role: assistant\s*$', b, flags=re.M)]
        assert blocks, path
        parsed = []
        commands = []
        bad_turns = 0
        for b in blocks:
            m = re.search(r'^call: (.*)$', b, flags=re.M)
            calls = ast.literal_eval(m.group(1)) if m else []
            parsed.append(calls)
            bad = False
            for c in calls:
                f = c['function']
                try:
                    args = json.loads(f['arguments']) if isinstance(f['arguments'], str) else f['arguments']
                    bad |= not isinstance(args, dict)
                except json.JSONDecodeError:
                    bad = True
                    args = None
                if f['name'] == 'bash':
                    commands.append(args.get('command') if isinstance(args, dict) and isinstance(args.get('command'), str) else f['arguments'])
            bad_turns += bad
        last_names = [c['function']['name'] for c in parsed[-1]]
        kind = 'submitted' if 'task_complete' in last_names else 'other_terminal'
        match = None
        if 'the server rejected the next prompt for length' in text:
            kind = 'context_prompt_rejected'
        elif 'the reply was cut off' in text:
            if decoded is None:
                decoded = [(tok.decode(c['raw_tokens']), c) for c in completed]
            matches = [c for raw_text, c in decoded if raw_text and raw_text in blocks[-1]]
            assert len(matches) == 1, (name, relative, len(matches))
            c = matches[0]
            res = reservations[c['id']]
            match = {k: c.get(k, res.get(k)) for k in ['id', 'completion_tokens', 'max_tokens', 'prompt_tokens']}
            assert match['completion_tokens'] == match['max_tokens'], match
            kind = 'response_cap' if match['max_tokens'] == 8192 else 'context_clipped_response'
        elif len(blocks) == 50 and 'task_complete' not in last_names:
            kind = 'cycle_limit'
        if kind == 'submitted':
            records.append({'path': relative, 'kind': kind, 'turns': len(blocks), 'json_error_turns': bad_turns})
            continue
        tokens = re.findall(r'\w+|[^\w\s]', blocks[-1][-14000:])
        top = collections.Counter(tuple(tokens[i:i+8]) for i in range(len(tokens)-7)).most_common(1)
        cc = collections.Counter(commands)
        records.append({'path': relative, 'kind': kind, 'turns': len(blocks), 'json_error_turns': bad_turns,
                        'sampling_match': match, 'top_8gram': top,
                        'repetitive_tail_screen': bool(top and top[0][1] >= 20),
                        'max_identical_command_count': max(cc.values(), default=0),
                        'distinct_commands': len(cc), 'total_commands': len(commands),
                        'top_commands': cc.most_common(3), 'last_commands': commands[-5:],
                        'last_block_tails': [b[-1000:] for b in blocks[-4:]],
                        'first_text': text[:1000], 'transcript_tail': text[-2000:]})
    assert len(records) == 240, (name, len(records))
    limits = [r for r in records if r['kind'] not in ['submitted', 'other_terminal']]
    summary = {'name': name, 'counts': dict(collections.Counter(r['kind'] for r in records)),
               'limit_total': len(limits), 'limit_rate': len(limits)/240,
               'limit_with_zero_json_errors': sum(r['json_error_turns'] == 0 for r in limits),
               'completed_sampling_requests': len(completed),
               'mean_completion_tokens': statistics.mean(c['completion_tokens'] for c in completed),
               'median_completion_tokens': statistics.median(c['completion_tokens'] for c in completed),
               'mean_assistant_turns': statistics.mean(r['turns'] for r in records),
               'repetitive_tail_screen_by_limit': dict(collections.Counter(r['kind'] for r in limits if r['repetitive_tail_screen']))}
    payload = {'summary': summary, 'root': str(root), 'ledger': str(ledger_path),
               'ledger_sha256': hashlib.sha256(ledger_path.read_bytes()).hexdigest(), 'transcript_sha256': hashes,
               'records': records}
    (out / f'{name}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    all_runs.append(summary)
    print(json.dumps(summary), flush=True)
(out / 'summary.json').write_text(json.dumps(all_runs, indent=2), encoding='utf-8')
