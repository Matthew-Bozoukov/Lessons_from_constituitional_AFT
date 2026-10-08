# ABOUTME: Derives limit, recovery and cost tables from the offline stopped-run census.
# ABOUTME: Uses recorded token/cache counts and a declared repetitive-tail heuristic; makes no model calls.
import collections
import gzip
import json
import pathlib
import re

p = pathlib.Path('output/gptoss_swe_analysis')
rows = [json.loads(line) for line in (p/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
tasks = json.loads((p/'tasks.json').read_text(encoding='utf-8'))
with gzip.open(p/'length_outputs.json.gz', 'rt', encoding='utf-8') as f:
    length_outputs = json.load(f)
result = {}
for arm in ('base', 'control', 'da15'):
    rr = [r for r in rows if r['arm'] == arm]
    assert len({r['id'] for r in rr}) == len(rr)
    input_cost = output_cost = error_cost = 0
    for r in rr:
        prompt = r['tokens']['prompt_tokens']
        cached = r['cached_prompt_tokens']
        assert 0 <= cached <= prompt
        ic = ((prompt-cached)*.78+cached*.156)/1_000_000
        oc = r['tokens']['completion_tokens']*1.94/1_000_000
        input_cost += ic
        output_cost += oc
        if r['errors']:
            error_cost += ic+oc
    capped = [r for r in length_outputs if r['arm'] == arm and r['tokens']['completion_tokens'] == 16384]
    repetitive = []
    for r in capped:
        tokens = re.findall(r'\w+|[^\w\s]', r['raw'][-14000:])
        counts = collections.Counter(tuple(tokens[i:i+8]) for i in range(len(tokens)-7))
        if max(counts.values(), default=0) >= 20:
            repetitive.append(r['iid'])
    result[arm] = {
        'input_usd': input_cost, 'output_usd': output_cost,
        'format_error_response_usd': error_cost,
        'full_16384_responses': len(capped),
        'repetitive_tail_screen_count': len(repetitive),
        'repetitive_tail_tasks': repetitive,
        'limit_reasons': dict(collections.Counter(
            t['diagnostics']['forced-submission.json']['limit_reason']
            for t in tasks if t['arm'] == arm and 'forced-submission.json' in t['diagnostics'])),
        'later_tool_after_bracket': sum(r.get('later_tool', False) for r in rr if r['brackets']),
        'next_exact_bracket_command': sum(r.get('next_exact_bracket_command', False) for r in rr if r['brackets']),
    }
(p/'derived_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
