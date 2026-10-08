# ABOUTME: Offline replay of stock cookbook 0.5.3 rendering on preserved SWE responses.
# ABOUTME: Demonstrates historical raw-tool duplication without sampling or executing commands.
import os
import sys
import pathlib
import gzip
import json
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,'C:/Users/nikak/AppData/Local/uv/cache/archive-v0/QsPURGjZo5FNpCUb')
from transformers import AutoTokenizer
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.renderers.base import ToolCall, RenderContext

p=pathlib.Path('output/gptoss_swe_analysis')
tok=AutoTokenizer.from_pretrained('openai/gpt-oss-120b',revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a',local_files_only=True)
renderer=GptOssRenderer(tok,use_system_prompt=True,reasoning_effort='medium',current_date='2026-10-08')
with gzip.open(p/'renderer_examples.json.gz','rt',encoding='utf-8') as f:examples=json.load(f)
out=[]
for ex in examples:
    m=ex['messages'][-1];reasoning=m.get('reasoning_content') or ''
    parts=([{'type':'thinking','thinking':reasoning}] if reasoning else [])+[{'type':'text','text':m.get('content') or ''}]
    calls=[ToolCall(id=c['id'],function=ToolCall.FunctionBody(**c['function'])) for c in m['tool_calls']]
    item={'role':'assistant','content':parts,'tool_calls':calls}
    ctx=RenderContext(idx=0,is_last=False)
    def render(x):
        rm=renderer.render_message(x,ctx)
        return tok.decode(rm.header.tokens+[t for chunk in rm.output for t in chunk.tokens])
    original=render(item)
    cleaned=render({**item,'content':[x for x in parts if x['type']=='thinking']})
    argument=calls[0].function.arguments
    assert original.count(argument)==2 and cleaned.count(argument)==1
    result={'arm':ex['arm'],'iid':ex['iid'],'argument_occurrences_original':original.count(argument),'argument_occurrences_without_raw_fallback':cleaned.count(argument),'original':original,'cleaned':cleaned}
    out.append(result)
(p/'renderer_replay.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in x.items() if k not in ('original','cleaned')} for x in out]))
