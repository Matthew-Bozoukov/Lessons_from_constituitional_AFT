# ABOUTME: Read-only Gemma 4 compatibility investigation; never rents or starts inference.
# ABOUTME: Saves pinned upstream metadata, template probes and current RunPod catalogue facts.
import hashlib
import json
import os
from pathlib import Path
import sys

import requests
from dotenv import load_dotenv
from transformers.utils.chat_template_utils import _compile_jinja_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
load_dotenv(Path(sys.argv[1]))
from src.infra import runpod
from src.infra.endpoints.vllm import pin_template, plan_serving
from src.model_profile import serving_params

OUT = ROOT / 'output' / '2026-10-09-gemma4-readiness'
OUT.mkdir(parents=True, exist_ok=True)

def get(url, json_result=True):
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.json() if json_result else r.text

models = {}
for model in ['google/gemma-4-31B-it', 'Qwen/Qwen3.6-27B']:
    info = get('https://huggingface.co/api/models/' + model)
    base = 'https://huggingface.co/' + model + '/resolve/' + info['sha'] + '/'
    tc = get(base + 'tokenizer_config.json')
    template = tc.get('chat_template') or get(base + 'chat_template.jinja', False)
    models[model] = dict(revision=info['sha'], gated=info['gated'],
                         embedded_template=bool(tc.get('chat_template')),
                         generation_config=get(base + 'generation_config.json'),
                         template_sha256=hashlib.sha256(template.encode()).hexdigest())
    if 'gemma' in model:
        (OUT / 'gemma-chat-template.jinja').write_text(template, encoding='utf-8')
        gemma_template = template

tools = [{'type': 'function', 'function': {'name': 'bash', 'description': 'Run a command',
         'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
messages = [{'role': 'system', 'content': 'Use the supplied tool.'},
            {'role': 'user', 'content': 'Inspect the current directory.'}]
calls = [{'id': 'call_1', 'type': 'function', 'function': {'name': 'bash', 'arguments': {'command': 'pwd'}}}]
history = messages + [{'role': 'assistant', 'content': None, 'reasoning': 'TRACE_SENTINEL', 'tool_calls': calls},
                      {'role': 'tool', 'tool_call_id': 'call_1', 'content': '/test'}]
compiled = _compile_jinja_template(pin_template(gemma_template, 'think'))
def render(msgs):
    return compiled.render(messages=msgs, tools=tools, bos_token='<bos>', add_generation_prompt=True)
initial, continued, later = render(messages), render(history), render(history + [{'role': 'assistant', 'content': 'Done.'}, {'role': 'user', 'content': 'New task.'}])
checks = dict(thinking_trigger='<|think|>' in initial,
              tool_definition='bash' in initial,
              reasoning_preserved_after_tool='TRACE_SENTINEL' in continued,
              reasoning_removed_after_new_user='TRACE_SENTINEL' not in later,
              tool_result_present='/test' in continued,
              tool_call_present='<|tool_call>call:bash' in continued,
              continuation_suffix=continued[-100:])
for key, value in checks.items():
    if isinstance(value, bool):
        assert value, key
try:
    plan_serving(dict(serving_params('google/gemma-4-31B-it'), native_context_window=262144),
                 {'context_window': 28000, 'needs_tool_calls': True}, 'google/gemma-4-31B-it', 'think')
except (SystemExit, ValueError, AssertionError) as e:
    current_launcher_block = str(e)
else:
    current_launcher_block = None

upstream = {}
for name, path in {'model': 'vllm/model_executor/models/registry.py', 'tools': 'vllm/tool_parsers/__init__.py', 'reasoning': 'vllm/reasoning/__init__.py'}.items():
    source = get('https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/' + path, False)
    upstream[name] = [line.strip() for line in source.splitlines() if 'gemma4' in line.lower()]
    assert upstream[name], name
provider = {}
try:
    provider['quotes_gpu_usd_hour'] = {gpu: runpod.gpu_price(gpu) for gpu in ['NVIDIA H100 80GB HBM3', 'NVIDIA H100 NVL', 'NVIDIA H200']}
    provider['account'] = runpod.graphql('query { myself { clientBalance currentSpendPerHr } }')['myself']
    provider['pods'] = [{k: p.get(k) for k in ('id', 'name', 'desiredStatus', 'costPerHr')} for p in runpod.active_pods()]
except Exception as e:
    provider['error_type'] = type(e).__name__
    provider['error'] = str(e).split('?')[0][:200]
result = dict(models=models, template_checks=checks, current_launcher_block=current_launcher_block,
              pinned_vllm_support=upstream, provider=provider,
              credentials_present={k: bool(os.getenv(k)) for k in ['HF_TOKEN', 'HF_ORG', 'USER_PREFIX', 'RUNPOD_API_KEY', 'OPENROUTER_API_KEY']})
(OUT / 'evidence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
