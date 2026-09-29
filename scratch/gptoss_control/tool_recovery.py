# ABOUTME: Count malformed-call episodes and distinguish resumed bash execution from mere submission.
# ABOUTME: Reads SHA-verified original scored transcripts; never executes tools or invokes models.
import ast
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

from omegaconf import OmegaConf


def parse(text):
    pending, turns = deque(), []
    for chunk in re.split(r'^== Step \d+ ==\n', text, flags=re.M)[1:]:
        role, body = chunk.rstrip('\n').split('\ncontent: ', 1)
        role = role.removeprefix('role: ')
        if role == 'assistant':
            calls = []
            if '\ncall: ' in body:
                body, call_text = body.rsplit('\ncall: ', 1)
                for call in ast.literal_eval(call_text):
                    fn = call['function']
                    item = {'name': fn['name'], 'arguments': fn['arguments'], 'error': None}
                    try:
                        args = json.loads(fn['arguments'])
                    except json.JSONDecodeError:
                        item['error'] = 'invalid_json'
                    else:
                        key = {'bash': 'command', 'task_complete': 'reason'}.get(fn['name'])
                        if key is None or not isinstance(args, dict) or set(args) != {key} or not isinstance(args[key], str):
                            item['error'] = 'schema'
                    calls.append(item)
                    pending.append(item)
            turns.append({'turn': len(turns) + 1, 'calls': calls, 'text': body})
        elif role == 'tool':
            item = pending.popleft()
            item['result'] = json.loads(body)
            feedback = item['result'].get('error', '')
            if item['error']:
                assert 'no command was executed' in feedback, (item, feedback)
            else:
                assert 'Tool arguments must be a valid JSON object' not in feedback
    assert not pending
    return turns


def main():
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    receipt = json.loads(Path(cfg.verified_receipt).read_text(encoding='utf-8'))
    assert receipt['verified_payload_revision'] == cfg.revision
    source = Path(cfg.source)
    def read(relative):
        p = source / relative
        assert hashlib.sha256(p.read_bytes()).hexdigest() == receipt['verified_files_sha256'][relative]
        return p.read_text(encoding='utf-8')
    progress = json.loads(read('results/scores_progress_gemini-3-flash-preview.json'))
    misconduct = json.loads(read('results/scores_gemini-3-flash-preview.json'))
    results = json.loads(read('results/results.json'))
    rows, transitions, affected_transcripts = [], Counter(), {}
    for relative in sorted(p for p in receipt['verified_files_sha256'] if p.endswith('/messages_record.txt')):
        text = read(relative)
        turns = parse(text)
        bad = [t for t in turns if any(c['error'] == 'invalid_json' for c in t['calls'])]
        all_bad = [t for t in turns if any(c['error'] for c in t['calls'])]
        _, variant, scenario, pass_name, _ = relative.split('/')
        key = f'{variant}/{scenario}/rollout_{int(pass_name[4:])-1:03d}'
        row = {'path': relative, 'score_key': key, 'assistant_turns': len(turns),
               'json_error_turns': [t['turn'] for t in bad],
               'all_format_error_turns': [t['turn'] for t in all_bad],
               'progress_score': progress[key]['score'], 'misconduct_score': misconduct[key]['score'],
               'submitted': results['progress']['submitted'][key]}
        if bad:
            affected_transcripts[relative] = text
            first, last = bad[0]['turn'], bad[-1]['turn']
            later_calls = [c for t in turns if t['turn'] > last for c in t['calls']]
            bash_after = [c for c in later_calls if c['name'] == 'bash' and not c['error']]
            bad_bash = any(c['name'] == 'bash' and c['error'] == 'invalid_json' for t in bad for c in t['calls'])
            row.update(first_error_turn=first, last_error_turn=last,
                malformed_bash=bad_bash, valid_bash_after_last_error=len(bash_after),
                zero_exit_bash_after_last_error=sum(c['result'].get('returncode') == 0 for c in bash_after),
                valid_bash_before_first_error=sum(c['name'] == 'bash' and not c['error'] for t in turns if t['turn'] < first for c in t['calls']),
                turns_after_last_error=len(turns)-last,
                subsequent_calls=[{'name': c['name'], 'error': c['error'], 'result': c['result']} for c in later_calls],
                turn_trace=[{'turn': t['turn'], 'calls': [{'name': c['name'], 'error': c['error'],
                    'command_start': c['arguments'][:120], 'result': c['result']} for c in t['calls']]} for t in turns])
            for t in bad:
                if t['turn'] == len(turns):
                    transitions['no_next_response'] += 1
                    continue
                nxt = turns[t['turn']]
                if any(c['error'] == 'invalid_json' for c in nxt['calls']):
                    outcome = 'another_json_error'
                elif any(c['name'] == 'bash' and not c['error'] for c in nxt['calls']):
                    outcome = 'valid_bash'
                elif any(c['name'] == 'task_complete' and not c['error'] for c in nxt['calls']):
                    outcome = 'task_complete_only'
                else:
                    outcome = 'other'
                transitions[outcome] += 1
        rows.append(row)
    affected = [r for r in rows if r['json_error_turns']]
    bash_affected = [r for r in affected if r['malformed_bash']]
    groups = {}
    for label, subset in [('affected', affected), ('unaffected', [r for r in rows if not r['json_error_turns']])]:
        groups[label] = {'n': len(subset), 'mean_progress': statistics.mean(r['progress_score'] for r in subset),
                        'progress_below_3': sum(r['progress_score'] < 3 for r in subset),
                        'submitted': sum(r['submitted'] for r in subset),
                        'misconduct': sum(r['misconduct_score'] >= 3 for r in subset)}
    summary = {'rollouts': len(rows), 'delivered_assistant_turns': sum(r['assistant_turns'] for r in rows),
               'json_error_responses': sum(len(r['json_error_turns']) for r in rows),
               'all_format_error_responses': sum(len(r['all_format_error_turns']) for r in rows),
               'json_affected_rollouts': len(affected),
               'all_format_affected_rollouts': sum(bool(r['all_format_error_turns']) for r in rows),
               'first_error_on_first_assistant_turn': sum(r['first_error_turn'] == 1 for r in affected),
               'first_error_turn_distribution': dict(Counter(r['first_error_turn'] for r in affected)),
               'next_response_after_each_json_error': dict(transitions),
               'malformed_bash_rollouts': len(bash_affected),
               'bash_resumed_after_last_error': sum(r['valid_bash_after_last_error'] > 0 for r in bash_affected),
               'bash_zero_exit_after_last_error': sum(r['zero_exit_bash_after_last_error'] > 0 for r in bash_affected),
               'groups': groups}
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('results.json', summary), ('trajectories.json', rows), ('run_meta.json', {
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'config': OmegaConf.to_container(cfg), 'read_files_verified_against_published_sha256': True,
        'interpretation': 'Valid subsequent bash is format recovery, not necessarily the same command or successful task; scores are observational, not causal.'})]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    for r in affected:
        print(r['path'], 'bad turns', r['json_error_turns'], 'bash resumed', r['valid_bash_after_last_error'],
              'zero exits', r['zero_exit_bash_after_last_error'], 'progress', r['progress_score'])


if __name__ == '__main__':
    main()
