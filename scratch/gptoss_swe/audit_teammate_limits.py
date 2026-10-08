# ABOUTME: Separates ODCV output truncation, context exhaustion and cycle-limit endings.
# ABOUTME: Matches saved truncated completions to their original allowance and screens repetition offline.
import ast
import collections
import json
import os
import pathlib
import re
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer

p=pathlib.Path('output/gptoss_swe_analysis/teammate_odcv')
audit=json.loads((p/'streak_audit.json').read_text(encoding='utf-8'))
ledger=[json.loads(l) for l in (p/'sampling.jsonl').read_text(encoding='utf-8').splitlines()]
reserved={r['id']:r for r in ledger if r['event']=='reserved'}
tok=AutoTokenizer.from_pretrained('openai/gpt-oss-120b',revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a',local_files_only=True)
completions=[(tok.decode(r['raw_tokens']),r,reserved[r['id']]) for r in ledger if r['event']=='completed']
out=[]
for r in audit['rollouts']:
    text=(p/r['path']).read_text(encoding='utf-8')
    if 'the server rejected the next prompt for length' in text:kind='context_prompt_rejected'
    elif 'the reply was cut off' in text:kind='response_truncated'
    elif r['turns']==50 and 'task_complete' not in r['last_names']:kind='cycle_limit'
    else:continue
    blocks=[b for b in re.split(r'^== Step \d+ ==\s*$',text,flags=re.M) if re.search(r'^role: assistant\s*$',b,flags=re.M)]
    last=blocks[-1]
    matches=[(c,res) for raw,c,res in completions if raw and raw in last] if kind=='response_truncated' else []
    if kind=='response_truncated':assert len(matches)==1,(r['path'],len(matches))
    tokens=re.findall(r'\w+|[^\w\s]',last[-14000:])
    ngrams=collections.Counter(tuple(tokens[i:i+8]) for i in range(len(tokens)-7))
    top=ngrams.most_common(1)
    commands=[]
    for b in blocks:
        m=re.search(r'^call: (.*)$',b,flags=re.M)
        for c in ast.literal_eval(m.group(1)) if m else []:
            f=c['function']
            if f['name']=='bash':commands.append(f['arguments'])
    cc=collections.Counter(commands)
    item={'path':r['path'],'kind':kind,'turns':r['turns'],'json_errors':r['bad_turns'],
          'completion_tokens':matches[0][0]['completion_tokens'] if matches else None,
          'allowance':matches[0][1]['max_tokens'] if matches else None,
          'prompt_tokens':matches[0][1]['prompt_tokens'] if matches else None,
          'top_ngram':top,'repetitive_tail_screen':bool(top and top[0][1]>=20),
          'max_identical_command_count':max(cc.values(),default=0),'distinct_commands':len(cc),
          'total_commands':len(commands),'last_5_calls':[re.findall(r'^call: (.*)$',b,flags=re.M) for b in blocks[-5:]],
          'last_tail':last[-1400:],'last_5_reasoning_tails':[b[-700:] for b in blocks[-5:]]}
    out.append(item)
summary={'kinds':dict(collections.Counter(r['kind'] for r in out)),
         'truncated_8192':sum(r['completion_tokens']==8192 for r in out),
         'truncated_smaller_allowance':sum(r['allowance'] is not None and r['allowance']<8192 for r in out),
         'truncated_not_reaching_allowance':sum(r['allowance'] is not None and r['completion_tokens']<r['allowance'] for r in out),
         'repetitive_tail_screen':dict(collections.Counter(r['kind'] for r in out if r['repetitive_tail_screen']))}
(p/'limit_behavior_audit.json').write_text(json.dumps({'summary':summary,'rollouts':out},indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
for r in out:
    print(json.dumps({k:r[k] for k in ['path','kind','turns','completion_tokens','allowance','json_errors','repetitive_tail_screen','max_identical_command_count','last_tail']},ensure_ascii=True))
