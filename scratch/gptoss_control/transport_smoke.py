# ABOUTME: Live, bounded GPT-OSS tool round trip through the real Docker-to-Tinker bridge.
# ABOUTME: Archives requests, responses and exact token counts; does not score a benchmark.
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from src.infra.endpoints.tinker import tinker_shim

os.chdir(ROOT)
common = Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],text=True).strip()).resolve()
load_dotenv(common.parent/'.env')
checkpoint = sys.argv[1] if len(sys.argv)>1 else 'base'
out = ROOT/'output/gptoss_control'/('base_transport' if checkpoint=='base' else 'adapter_transport')
code = r'''
import json, os, urllib.request
base='http://host.docker.internal:18322'
def post(route,body):
    req=urllib.request.Request(base+route, data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['TINKER_SHIM_API_KEY']})
    with urllib.request.urlopen(req,timeout=300) as r: return json.load(r)
tools=[{'type':'function','function':{'name':'lookup','description':'Returns the stored value for a key.',
    'parameters':{'type':'object','properties':{'key':{'type':'string'}},'required':['key']}}}]
messages=[{'role':'user','content':'Use lookup to read the value for key test_value. Do not guess it. Once the tool replies, tell me the value.'}]
body={'model':'openai/gpt-oss-120b','messages':messages,'tools':tools,'temperature':0,'max_tokens':2048}
count=post('/tokenize',body)
response=post('/v1/chat/completions',body)
print(json.dumps({'request':body,'tokenize':count,'response':response}),flush=True)
assert response['usage']['prompt_tokens']==count['count']
m=response['choices'][0]['message']
assert '<|message|>' not in (m.get('content') or '')
assert response['choices'][0]['finish_reason']=='tool_calls' and len(m['tool_calls'])==1
call=m['tool_calls'][0]
assert call['function']['name']=='lookup' and json.loads(call['function']['arguments'])=={'key':'test_value'}
messages.extend([m,{'role':'tool','tool_call_id':call['id'],'content':'{"value":"cedar-941"}'}])
count=post('/tokenize',body)
response=post('/v1/chat/completions',body)
print(json.dumps({'request':body,'tokenize':count,'response':response}),flush=True)
assert response['usage']['prompt_tokens']==count['count']
assert response['choices'][0]['finish_reason']=='stop'
assert 'cedar-941' in response['choices'][0]['message']['content']
'''
with tinker_shim(checkpoint,port=18322,bind='0.0.0.0',log_dir=out,max_cost_usd=.10):
    result=subprocess.run(['docker','run','--rm','-i','--name','gptoss-control-transport',
        '-e','TINKER_SHIM_API_KEY','--entrypoint','python',
        'nika-da-sep25-prewarm-bfc105179cb2-executor:latest','-'],
        input=code,text=True,encoding='utf-8',capture_output=True,timeout=660)
    (out/'roundtrip.jsonl').write_text(result.stdout,encoding='utf-8')
    (out/'stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Transport smoke failed; see {out}')
    (out/'passed.json').write_text(json.dumps({'checkpoint':checkpoint,'passed':True}),encoding='utf-8')
    print('PASS: Docker bridge, exact counting, real tool call and tool-result continuation')
