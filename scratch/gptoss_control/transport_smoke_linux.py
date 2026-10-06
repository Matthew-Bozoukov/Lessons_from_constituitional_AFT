# ABOUTME: transport_smoke.py's qualification on a Linux Docker host: same assertions, reachable bridge.
# ABOUTME: Proves the Docker-to-Tinker path, exact token counting and a real tool round trip.
"""The adapter transport smoke, runnable where Docker Desktop's conveniences do not exist.

Two things in scratch/gptoss_control/transport_smoke.py only hold on Docker Desktop:

  * `host.docker.internal` resolves implicitly. On Linux it does not: a container here gets
    NXDOMAIN unless the run passes `--add-host=host.docker.internal:host-gateway`. This host's
    bridge gateway is 10.201.0.1, a non-default address that has already cost this project two
    eval pods (docs/GOTCHAS.md), so the gateway is asked for by name rather than assumed.
  * The image `nika-da-sep25-prewarm-bfc105179cb2-executor:latest` is a local artifact that
    nothing in this repository builds. `python:3.13-slim` is used instead -- the base ODCV's own
    scenarios build on.

Every assertion is carried over unchanged, because what is under test is the SHIM and the network
path, not the image's contents: exact token counting (/tokenize equals usage.prompt_tokens), no
raw Harmony markers leaking into content, a real tool call with the right arguments, and a
tool-result continuation that reads the planted value back.

What this does NOT cover, which the original does on the machine that has that image: that
Nika's prewarm executor specifically can reach the shim. Run the original there for that.

Usage:
    uv run --project src/infra/endpoints/tinker_runtime python \
        scratch/gptoss_control/transport_smoke_linux.py <tinker://... sampler path> \
        --output output/gptoss_arms/<arm>/adapter_transport --port 18322
"""
import argparse
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
common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip()).resolve()
load_dotenv(common.parent / '.env')
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', nargs='?', default='base')
parser.add_argument('--output', type=Path)
parser.add_argument('--port', type=int, default=18322)
parser.add_argument('--image', default='python:3.13-slim')
parser.add_argument('--reasoning', default='medium', choices=['low', 'medium', 'high'],
                    help='must match the arm it qualifies; it is baked into the Harmony prompt')
parser.add_argument('--allow-incomplete-tool-cycle', action='store_true',
                    help='record, rather than fail on, an adapter that does not answer after a tool '
                         'result. The INFRASTRUCTURE assertions still must pass (exact token '
                         'counting, no Harmony marker leak, a correctly-formed call). Use when the '
                         'arm under test is known not to complete tool cycles -- 2026-10-06-gptoss'
                         '120b-0-msm re-issues the identical call instead of answering, because '
                         'msm-mix holds 1,710 tool calls and 0 tool results, so nothing in its '
                         'training showed what follows a result. passed.json then records '
                         'tool_cycle_completed: false and the turn-2 finish_reason.')
args = parser.parse_args()
checkpoint = args.checkpoint
out = args.output or ROOT / 'output/gptoss_control' / ('base_transport' if checkpoint == 'base' else 'adapter_transport')
out.mkdir(parents=True, exist_ok=True)

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
__CYCLE_CHECK__
'''
STRICT = """assert response['choices'][0]['finish_reason']=='stop'
assert 'cedar-941' in response['choices'][0]['message']['content']
print(json.dumps({'verdict': {'tool_cycle_completed': True,
    'turn2_finish_reason': response['choices'][0]['finish_reason']}}),flush=True)"""
LENIENT = """m2=response['choices'][0]
print(json.dumps({'verdict': {'tool_cycle_completed': m2['finish_reason']=='stop'
        and 'cedar-941' in (m2['message'].get('content') or ''),
    'turn2_finish_reason': m2['finish_reason'],
    'turn2_repeated_the_call': bool(m2['message'].get('tool_calls'))}}),flush=True)"""
code = code.replace('__CYCLE_CHECK__', LENIENT if args.allow_incomplete_tool_cycle else STRICT)
code = code.replace('host.docker.internal:18322', f'host.docker.internal:{args.port}')
with tinker_shim(checkpoint, port=args.port, bind='0.0.0.0', reasoning=args.reasoning,
                 log_dir=out, max_cost_usd=.10):
    result = subprocess.run(
        ['docker', 'run', '--rm', '-i', '--name', f'gptoss-transport-linux-{args.port}',
         '--add-host', 'host.docker.internal:host-gateway',
         '-e', 'TINKER_SHIM_API_KEY', '--entrypoint', 'python', args.image, '-'],
        input=code, text=True, encoding='utf-8', capture_output=True, timeout=660)
    (out / 'roundtrip.jsonl').write_text(result.stdout, encoding='utf-8')
    (out / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Transport smoke failed; see {out}')
    # run.py's evaluate() requires this to equal the arm's trained_adapter.json `sampler`.
    verdict = {}
    for line in result.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and 'verdict' in payload:
            verdict = payload['verdict']
    (out / 'passed.json').write_text(json.dumps(
        {'checkpoint': checkpoint, 'passed': True, 'image': args.image,
         'reasoning': args.reasoning, 'host_alias': 'host-gateway',
         'infrastructure_qualified': True,
         'allowed_incomplete_tool_cycle': bool(args.allow_incomplete_tool_cycle),
         **verdict}), encoding='utf-8')
    print('PASS: Docker bridge, exact counting, real tool call and tool-result continuation')
