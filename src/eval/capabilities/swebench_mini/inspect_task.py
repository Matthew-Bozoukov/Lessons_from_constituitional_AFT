# ABOUTME: Optional inspect-evals SWE-bench task backend for the existing durable Lite fleet.
# ABOUTME: Runs one leased instance; native Inspect agent/logs, shared resource policy, deferred official grading.
import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import httpx2
import yaml
from inspect_ai import eval as inspect_eval, task_with
from inspect_ai.agent import react, AgentPrompt, AgentSubmit
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import solver
from inspect_ai.tool import tool, ToolError
from inspect_ai.util import sandbox, SandboxEnvironmentSpec
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.solvers import DEFAULT_TOOL_TIMEOUT

from src.eval.capabilities.swebench_mini.fleet_state import atomic, read, tool_slot
from src.eval.capabilities.swebench_mini.fleet_task import (
    resource_environment, install_container_limits, container_resources, forced_patch)
from src.eval.capabilities.swebench_mini.fleet_protocol import configure_protocol
from src.eval.capabilities.swebench_mini.inspect_transport import InspectTransport

ENV = Path(__file__).parent/'envs/inspect'
VERSIONS = {'inspect-ai':'0.3.268','inspect-evals':'0.21.0','swebench':'4.1.0','openai':'3.19.2'}


def container_id(request):
    ids = subprocess.check_output(['docker','ps','-q','--filter','label=lasr_attempt='+request['attempt']],text=True).split()
    assert len(ids) == 1, 'Inspect attempt must own exactly one sandbox'
    return ids[0]


def make_task(request, transport):
    out = Path(request['out']); out.mkdir(parents=True, exist_ok=True)
    # Local JSON directory uses the upstream dataset adapter, with no network or gold data.
    data_dir = out/'frozen-dataset'; data_dir.mkdir(exist_ok=True)
    row = dict(request['instance'])
    for key in ('patch','test_patch','hints_text'):
        row[key] = ''
    for key in ('PASS_TO_PASS','FAIL_TO_PASS'):
        row[key] = '[]'
    for key in ('environment_setup_commit','created_at'):
        row.setdefault(key, '')
    (data_dir/'test.jsonl').write_text(json.dumps(row)+'\n')
    compose = out/'compose.yaml'
    timeout = request.get('tool_timeout_seconds', 120)
    compose.write_text(yaml.safe_dump({'services':{'default':{
        'image':request['image'], 'pull_policy':'never', 'command':['sleep','infinity'],
        'working_dir':'/testbed', 'network_mode':'none', 'cpus':request['cpus'],
        'mem_limit':request['memory'], 'pids_limit':request['pids'],
        'environment':dict(request['environment'],**resource_environment(request['cpus'])),
        'labels':{'lasr_campaign':request['campaign'],'lasr_attempt':request['attempt']}}}}))

    @solver
    def prepare_sandbox():
        async def solve(state, generate):
            cid = await asyncio.to_thread(container_id, request)
            await asyncio.to_thread(install_container_limits,cid,request['cpus'],timeout)
            return state
        return solve

    @tool(name='bash')
    def bounded_bash():
        async def execute(command: str) -> str:
            """Execute a self-contained bash command in the repository.

            Args:
                command: Bash command to execute. Every call starts a fresh shell.
            """
            if transport.terminal or transport.rejected_batch:
                raise ToolError('No command executed: terminal budget or rejected tool-call batch.')
            slot = tool_slot(request['tool_slots_path'],request['tool_concurrency'],
                wait_seconds=request['tool_queue_timeout_seconds'],min_available_gib=request['min_available_memory_gib'])
            waited = await asyncio.to_thread(slot.__enter__)
            started = time.monotonic()
            try:
                result = await sandbox().exec(['bash','-c',command],timeout=timeout+15)
                text = result.stdout + result.stderr
                limit = request.get('tool_output_limit', 10000)
                if len(text)>limit:
                    text = text[:limit//2]+'\n[output truncated]\n'+text[-limit//2:]
                return f'<returncode>{result.returncode}</returncode>\n<output>\n{text}\n</output>'
            finally:
                await asyncio.to_thread(slot.__exit__,None,None,None)
                with (out/'tool-timing.jsonl').open('a') as f:
                    f.write(json.dumps({'queue_seconds':waited,'execution_seconds':time.monotonic()-started,'time':time.time()})+'\n')
        return execute

    async def on_continue(state):
        if transport.terminal:
            return False
        if transport.calls >= request['step_limit']:
            transport.terminal = 'step_limit'
            return False
        return True

    @tool(name='submit')
    def bounded_submit():
        async def execute(answer: str) -> str:
            """Submit the completed source-code change.

            Args:
                answer: Brief description of the change.
            """
            if transport.terminal or transport.rejected_batch:
                raise ToolError('No submission executed: terminal budget or rejected tool-call batch.')
            return answer
        return execute

    async def cleanup(state):
        cid = await asyncio.to_thread(container_id, request)
        inspection = json.loads(await asyncio.to_thread(subprocess.check_output, ['docker','inspect',cid]))[0]
        policy = inspection['HostConfig']
        atomic(out/'sandbox-policy.json', {k:policy[k] for k in ('NetworkMode','NanoCpus','Memory','PidsLimit')})
        atomic(out/'resources.json', await asyncio.to_thread(container_resources, cid))
        patch = await asyncio.to_thread(forced_patch,cid)
        (out/'model.patch').write_text(patch)
        # Both native Inspect logs and the legacy-compatible final transcript survive cleanup.
        messages = [m.model_dump(mode='json') for m in state.messages]
        atomic(out/'inspect-messages.json',messages)

    task = swe_bench(dataset=str(data_dir.resolve()),revision='local-frozen',arch='x86_64',
        image_name_template=request['image'],
        sandbox_config=lambda *_:SandboxEnvironmentSpec(type='docker',config=str(compose.resolve())),
        tool_timeout=max(timeout,DEFAULT_TOOL_TIMEOUT))
    task.metadata.update(agent_backend='inspect',protocol_version=request['protocol_version'],
                         official_grading='deferred to SWE-bench 4.1.0',sampling=request['sampling'],
                         versions=VERSIONS)
    # Explicit new agent protocol: bash + submit, not the default python/editor/bash_session agent.
    task.solver = react(prompt=AgentPrompt(instructions=request['inspect']['instructions'],
        assistant_prompt=None,handoff_prompt=None,submit_prompt=None),tools=[bounded_bash()],
        submit=AgentSubmit(tool=bounded_submit(),answer_only=True,keep_in_messages=True),attempts=1,
        on_continue=on_continue,retry_refusals=0,truncation='disabled',compaction=None)
    task.setup = prepare_sandbox()
    task.cleanup = cleanup
    task.scorer = None  # All model patches are graded later by the existing official 4.1.0 harness.
    task.message_limit = None  # Upstream 30 messages is NOT 30 model turns.
    task.turn_limit = request['step_limit']
    task.token_limit = request['max_task_tokens']
    task.token_limit_type = 'output'
    # Logging is durable; filesystem checkpoint/resume is intentionally not enabled yet.
    return task_with(task, checkpoint=False)


def export(request, transport, logs):
    out = Path(request['out']); log = logs[0]
    sample = log.samples[0] if log.samples else None
    error = bool(log.status != 'success' or not sample or sample.error)
    reason = transport.terminal
    if sample and sample.limit:
        reason = reason or ('step_limit' if sample.limit.type in ('turn','message') else 'task_token_limit')
    status = 'InfrastructureError' if error else 'LimitsExceeded' if reason else 'Submitted'
    messages = []
    native = read(out/'inspect-messages.json') if (out/'inspect-messages.json').exists() else []
    response_index = 0
    for m in native:
        role = m['role']; content = m.get('content') or ''
        if isinstance(content,list):
            reasoning = '\n'.join(x.get('reasoning','') for x in content if x.get('type')=='reasoning')
            content = '\n'.join(x.get('text','') for x in content if x.get('type')=='text')
        else:
            reasoning = ''
        item = {'role':role,'content':content}
        if reasoning: item['reasoning_content']=reasoning
        if role=='assistant' and response_index < len(transport.responses):
            raw = transport.responses[response_index];response_index+=1
            item['extra']={'response':raw}
            item['tool_calls']=raw['choices'][0]['message'].get('tool_calls') or []
        messages.append(item)
    info={'exit_status':status,'agent_backend':'inspect','limit_reason':reason,
          'model_stats':{'api_calls':transport.calls},'completion_tokens':transport.total,
          'inspect_status':log.status,'error':sample.error.model_dump(mode='json') if sample and sample.error else None}
    messages.append({'role':'exit','content':reason or status,'extra':info})
    atomic(out/'checkpoint.traj.json',{'info':info,'messages':messages})
    if not error:
        patch=(out/'model.patch').read_text()
        atomic(out/'preds.json',{request['instance']['instance_id']:{'model_name_or_path':request['model'],
             'model_patch':patch,'instance_id':request['instance']['instance_id']}})
    atomic(out/'inspect-result.json',info)
    return 1 if error else 0


def main():
    parser=argparse.ArgumentParser()
    options=parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--request');options.add_argument('--check-runtime',action='store_true')
    args=parser.parse_args()
    for package,version in VERSIONS.items():
        assert importlib.metadata.version(package)==version,(package,'dependency drift')
    if args.check_runtime:
        print(json.dumps(VERSIONS));return
    request=read(args.request)
    assert request['protocol_version'].startswith('lite-inspect-'), 'Use a distinct Inspect protocol and campaign'
    protocol=configure_protocol({'model':{'model_kwargs':{}}},request)
    atomic(Path(request['out'])/'protocol.json',dict(protocol,agent_backend='inspect',versions=VERSIONS))
    transport=InspectTransport(request)
    cfg=GenerateConfig(**{k:request['sampling'][k] for k in ('temperature','top_p','top_k','presence_penalty','frequency_penalty')},
        extra_body={k:request['sampling'][k] for k in ('top_k','min_p','repetition_penalty')},
        reasoning_history='all',max_tokens=request['max_response_tokens'],max_retries=0,
        timeout=request['model_request_timeout_seconds'],max_connections=1,parallel_tool_calls=False,cache=False)
    # Native vLLM provider doesn't forward a custom HTTP client in this pin. The supported
    # generic OpenAI-compatible provider does, without starting or reconfiguring a server.
    model=get_model('openai-api/lasr/'+request['model'].removeprefix('hosted_vllm/'),
        base_url=request['endpoint'],api_key='local-only',config=cfg,stream=False,strict_tools=False,
        max_retries=0,http_client=httpx2.AsyncClient(transport=transport,
            timeout=request['model_request_timeout_seconds'],trust_env=False))
    task=make_task(request,transport)
    logs=inspect_eval(task,model=model,log_dir=str(Path(request['out'])/'inspect'),
        display='plain',log_realtime=True,log_buffer=1,log_model_api=True,
        retry_on_error=0,fail_on_error=True,score=False,max_samples=1,max_sandboxes=1,
        sandbox_cleanup=True,sandbox_prebuilt=True)
    raise SystemExit(export(request,transport,logs))


if __name__=='__main__': main()
