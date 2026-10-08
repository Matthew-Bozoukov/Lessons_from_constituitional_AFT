# ABOUTME: Offline audit of the teammate's pinned October 6 control ODCV sampling ledger.
# ABOUTME: Decodes saved tokens and counts bracket defects without sampling or executing tools.
import collections
import json
import os
import pathlib
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from transformers import AutoTokenizer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

p=pathlib.Path('output/gptoss_swe_analysis/teammate_odcv')
rows=[json.loads(x) for x in (p/'sampling.jsonl').read_text(encoding='utf-8').splitlines()]
tok=AutoTokenizer.from_pretrained('openai/gpt-oss-120b',revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a',local_files_only=True)
renderer=GptOssRenderer(tok)
counts=collections.Counter()
samples=[]
reservations={r['id']:r for r in rows if r['event']=='reserved'}
for row in rows:
    counts['event_'+row['event']]+=1
    if row['event']=='model_output_error':counts['output_error_'+row.get('type','unknown')]+=1
    if row['event']!='completed':continue
    raw=tok.decode(row['raw_tokens'])
    if row['completion_tokens']==reservations[row['id']]['max_tokens']:counts['allowance_reached']+=1
    if row['completion_tokens']==8192:counts['full_8192']+=1
    calls=renderer._parse_harmony_messages(raw.removesuffix('<|call|>').removesuffix('<|return|>'))
    for call in calls:
        if not (call.get('recipient') or '').startswith('functions.'):continue
        counts['raw_tool_blocks']+=1
        arg=call.get('content') or ''
        try:json.loads(arg)
        except json.JSONDecodeError as e:
            counts['invalid_json_blocks']+=1
            if e.pos<len(arg) and arg[e.pos]==']':
                try:fixed=json.loads(arg[:e.pos]+arg[e.pos+1:])
                except json.JSONDecodeError:continue
                if isinstance(fixed,dict):
                    counts['confirmed_extra_bracket_blocks']+=1
                    samples.append({'id':row['id'],'tail':arg[-180:]})
result={'counts':dict(counts),'checkpoints':sorted({r['checkpoint'] for r in reservations.values()}),'bracket_examples':samples,
        'error_records':[r for r in rows if r['event']=='error']}
(p/'sampling_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('bracket_examples','error_records')},indent=2))
