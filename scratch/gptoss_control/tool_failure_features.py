# ABOUTME: Measure reasoning, heredoc, and malformed closing-bracket associations without executing tools.
# ABOUTME: Diagnostic deletion at a JSON error offset identifies extra brackets; never repairs live calls.
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf


def main():
    cfg = OmegaConf.load(Path(__file__).with_suffix('.yaml'))
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, groups = [], []
    reference = None
    for run, directory in cfg.runs.items():
        directory = Path(directory)
        meta = json.loads((directory / 'run_meta.json').read_text(encoding='utf-8'))
        prompts = json.loads((directory / 'prompts.json').read_text(encoding='utf-8'))
        tokens = [p['tokens'] for p in prompts]
        if reference is None:
            reference = tokens
        assert tokens == reference, 'All runs must have identical input tokens'
        source = [json.loads(line) for line in (directory / 'responses.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(source) == 60
        assert len({(r['arm'], r['prompt'], r['seed']) for r in source}) == 60
        for r in source:
            extra_brackets, malformed_calls = 0, 0
            for call in r['calls']:
                arg = call['arguments']
                try:
                    json.loads(arg)
                except json.JSONDecodeError as exc:
                    malformed_calls += 1
                    if arg[exc.pos:exc.pos + 1] == ']':
                        try:
                            json.loads(arg[:exc.pos] + arg[exc.pos + 1:])
                        except json.JSONDecodeError:
                            pass
                        else:
                            extra_brackets += 1
            rows.append({'run': run, 'arm': r['arm'], 'prompt': r['prompt'], 'seed': r['seed'],
                         'temperature': meta['config']['temperature'],
                         'analysis_channel': '<|channel|>analysis' in r['text'],
                         'heredoc': any('<<' in c['arguments'] for c in r['calls']),
                         'output_tokens': r['output_tokens'],
                         'truncated': r['output_tokens'] == meta['config']['max_tokens'] and r['raw_tokens'][-1] not in [200002, 200012],
                         'extra_bracket_calls': extra_brackets, 'malformed_calls': malformed_calls,
                         'errors': r['errors']})
        for arm in ['base', 'lora']:
            rr = [r for r in rows if r['run'] == run and r['arm'] == arm]
            groups.append({'run': run, 'arm': arm, 'n': len(rr),
                'temperature': meta['config']['temperature'], 'seeds': meta['config']['seeds'],
                'analysis_channel': sum(r['analysis_channel'] for r in rr),
                'heredoc': sum(r['heredoc'] for r in rr),
                'heredoc_complete_json_failures': sum(r['heredoc'] and r['malformed_calls'] > 0 and not r['truncated'] for r in rr),
                'completed_json_failures': sum(r['malformed_calls'] > 0 and not r['truncated'] for r in rr),
                'extra_bracket_responses': sum(r['extra_bracket_calls'] > 0 for r in rr),
                'truncated': sum(r['truncated'] for r in rr),
                'any_format_error': sum(bool(r['errors']) for r in rr)})
    result = {'groups': groups, 'features': rows,
              'interpretation': 'Descriptive associations on failure-selected histories, not causal identification; diagnostic bracket deletion never executed.'}
    (out / 'features.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (out / 'run_meta.json').write_text(json.dumps({'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'config': OmegaConf.to_container(cfg)}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
