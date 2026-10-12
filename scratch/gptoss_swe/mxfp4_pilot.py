# ABOUTME: Matched base-only ODCV/SWE pilot with raw-token vLLM and reference Tinker Harmony.
# ABOUTME: Freeze random tasks before outcomes, preserve paid evidence, and release our H200 before grading.
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

from omegaconf import OmegaConf
from scratch.gptoss_swe.openai_smoke import environment, IMAGE, save

C = OmegaConf.load('scratch/gptoss_swe/mxfp4_pilot.yaml')
ROOT = Path(C.root)
LABEL = 'lasr.campaign='+str(C.campaign)
CACHE = Path('output/gptoss_shared_smoke/cache/metadata')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def phase(root, state, **kw):
    save(root/'status.json', dict(phase=state, at=time.time(), **kw))


def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(dict(pid=os.getpid(), at=time.time(), argv=sys.argv), f)


def named(backend, kind):
    return f'{C.campaign}-{backend}-{kind}'


def prepare():
    from huggingface_hub import HfApi
    assert not ROOT.exists(), 'Never overwrite pilot history'
    ROOT.mkdir()
    claim(ROOT/'preparation-started.json')
    rng=random.Random(int(C.seed))
    bench=Path('src/eval/misalignment/odcv/third_party/odcv-bench')
    cells=sorted(v+'/'+p.name for v in ('mandated','incentivized') for p in (bench/(v+'_scenarios')).iterdir() if p.is_dir())
    assert len(cells)==80
    selected=sorted(rng.sample(cells,int(C.odcv_cells)))
    ids=sorted(rng.sample(sorted(read(CACHE/'images.json')),int(C.swe_tasks)))
    save(ROOT/'selection.json',dict(seed=int(C.seed),rule='Random without replacement before outcomes',odcv=selected,swe=ids))
    revision=HfApi().model_info('openai/gpt-oss-120b').sha
    save(ROOT/'model.json',dict(repo='openai/gpt-oss-120b',revision=revision,format='released MXFP4'))
    images=read(CACHE/'images.json'); verified=[]
    for iid in ids:
        info=images[iid]
        check=subprocess.run(['docker','image','inspect',info['name']],capture_output=True,text=True)
        if check.returncode:
            with (ROOT/'image-pulls.log').open('a',encoding='utf-8') as log:
                subprocess.run(['docker','pull',info['name']+'@'+info['digest'].split('@')[-1]],stdout=log,stderr=subprocess.STDOUT,check=True)
            # Pulling by digest may not establish the tag used by the historical worker.
            subprocess.run(['docker','tag',info['name']+'@'+info['digest'].split('@')[-1],info['name']],check=True)
        actual=json.loads(subprocess.check_output(['docker','image','inspect',info['name']]))[0]
        assert info['digest'] in actual['RepoDigests'],iid
        subprocess.run(['docker','run','--rm','--network','none','--entrypoint','/bin/bash',info['name'],
            '-lc','cd /testbed && git rev-parse HEAD && test -d /opt/miniconda3'],check=True)
        verified.append(dict(instance_id=iid,**info))
    save(ROOT/'images-verified.json',verified)
    include={v:[x.split('/',1)[1] for x in selected if x.startswith(v+'/')] for v in ('mandated','incentivized')}
    for backend in ('tinker','mxfp4'):
        r=ROOT/backend;r.mkdir()
        od=OmegaConf.load('configs/eval/odcv/lite.yaml')
        od=OmegaConf.merge(od,dict(published_key=None,concurrency=int(C.odcv_concurrency),passes=int(C.odcv_passes),
            temperature=.7,strict_tool_validation=True,endpoint_api_key_env='TINKER_API_KEY',model='openai/gpt-oss-120b',
            model_key=named(backend,'odcv'),base_url=f'http://host.docker.internal:{C.ports[backend].odcv}/v1',
            output_root=str((r/'odcv').resolve()),system_preamble_file=None,scenario_timeout_s=2400,
            judge_workers=4,progress_judge=True,judge_budget=dict(ledger=str((r/'judge-budget.json').resolve()),cap_usd=float(C.judge_ceiling_each),max_tokens=8192),
            include_scenarios=include))
        od.serving.context_window=int(C.odcv_context)
        OmegaConf.save(od,r/'odcv.yaml')
        sw=OmegaConf.load('output/gptoss_shared_smoke/swe.yaml')
        sw.campaign=named(backend,'swe');sw.instance_ids=ids;sw.workers=int(C.swe_workers)
        sw.output_root='/work/'+(r/'swe').as_posix();sw.grade=False;sw.grading.max_workers=2
        sw.tinker.context_window=int(C.swe_context);sw.tinker.budget_ledger='/work/'+(r/'swe-budget.json').as_posix()
        sw.worker.model_request_timeout_seconds=7200;sw.worker.model_request_attempts=2
        sw.worker.max_infrastructure_attempts=3;sw.worker.max_infrastructure_failures=6
        sw.worker.max_response_tokens=int(C.swe_response);sw.worker.max_task_tokens=int(C.swe_task_tokens);sw.worker.step_limit=int(C.swe_steps)
        sw.sampling.temperature=1.;sw.sampling.top_p=1.;sw.sampling.top_k=-1
        OmegaConf.save(sw,r/'swe.yaml')
    phase(ROOT,'prepared',model_revision=revision)


def start_shim(backend,kind):
    import requests
    r=ROOT/backend;env=environment();p=int(C.ports[backend][kind])
    settings=dict(TINKER_CONTEXT_WINDOW=str(C.odcv_context if kind=='odcv' else C.swe_context),
        DEFAULT_MAX_TOKENS='remaining' if kind=='odcv' else str(C.swe_response),
        TINKER_DEFAULT_TOP_P='.95' if kind=='odcv' else '1',TINKER_DEFAULT_TOP_K='20' if kind=='odcv' else '-1',
        TINKER_BIND_HOST='0.0.0.0',TINKER_RENDER_DATE='2026-10-12',REASONING_LEVEL='medium',
        TINKER_TRACE_DIR='/work/'+(r/(kind+'-traces')).as_posix())
    module='src.infra.endpoints.tinker_server'
    if backend=='tinker':
        settings.update(TINKER_CKPT='tinker://base',TINKER_BUDGET_USD=str(C.tinker_odcv_ceiling if kind=='odcv' else C.tinker_swe_ceiling),
            TINKER_BUDGET_LEDGER='/work/'+(r/(kind+'-budget.json')).as_posix(),TINKER_PRICE_RATES_JSON=json.dumps(OmegaConf.to_container(C.price_rates)))
    else:
        module='scratch.gptoss_swe.mxfp4_bridge'
        settings.update(MXFP4_URL=f'http://host.docker.internal:{C.raw_port}',MXFP4_REVISION=read(ROOT/'model.json')['revision'],
            MXFP4_INSTANCE=named(backend,kind),MXFP4_RAW_TRACES='/work/'+(r/(kind+'-raw')).as_posix())
    args=['docker','run','-d','--name',named(backend,kind+'-shim'),'--label',LABEL,'-p',f'127.0.0.1:{p}:1234',
        '-v',f'{Path.cwd()}:/work','-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
        '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface','-e','TINKER_API_KEY','-e','HF_TOKEN']
    if backend=='mxfp4':args+=['-e','MXFP4_API_KEY']
    for k,v in settings.items():args+=['-e',k+'='+v]
    subprocess.run(args+[IMAGE,'src/infra/endpoints/tinker_env/.venv/bin/python','-m',module],env=env,check=True)
    save(r/(kind+'-shim-settings.json'),settings)
    for _ in range(180):
        try:
            response=requests.get(f'http://127.0.0.1:{p}/v1/models',headers={'Authorization':'Bearer '+env['TINKER_API_KEY']},timeout=5)
            response.raise_for_status(); identity=response.json()
            assert identity['data'][0]['checkpoint']==('tinker://base' if backend=='tinker' else 'vllm://openai/gpt-oss-120b@'+read(ROOT/'model.json')['revision'])
            save(r/(kind+'-identity.json'),identity);return
        except requests.RequestException:time.sleep(2)
    raise TimeoutError('Local shim startup timed out')


def stop_shim(backend,kind):
    name=named(backend,kind+'-shim')
    check=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    if check.returncode:return
    with (ROOT/backend/(kind+'-shim.log')).open('w',encoding='utf-8') as log:
        subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT)
    if json.loads(check.stdout)[0]['State']['Running']:
        subprocess.run(['docker','stop','--time','30',name],check=True)


def odcv(backend):
    from src.eval.misalignment.odcv import runner,odcv_rollout,odcv_judge,progress_judge
    r=ROOT/backend;claim(r/'odcv-owner-started.json')
    for module in (odcv_rollout,odcv_judge,progress_judge):module.openrouter_usage=lambda settle_s=0:None
    runner.PASS_RETRIES=0;runner._prune_networks=lambda:None
    original=runner._run_pass;passes=[]
    def tracked(*args,**kwargs):
        phase(r/'odcv','inference',pass_number=len(passes)+1)
        value=original(*args,**kwargs);passes.append(value);save(r/'completed-passes.json',passes)
        if len(passes)==int(C.odcv_passes):
            stop_shim(backend,'odcv');save(r/'odcv-inference-ended.json',dict(at=time.time()))
            phase(r/'odcv','judging')
        return value
    runner._run_pass=tracked
    try:
        start_shim(backend,'odcv');cfg=OmegaConf.load(r/'odcv.yaml')
        result=runner.run(SimpleNamespace(model_name='openai/gpt-oss-120b',spec=SimpleNamespace(model_key=cfg.model_key),
            base_url=f'http://127.0.0.1:{C.ports[backend].odcv}/v1'),cfg,r/'odcv')
        save(r/'odcv-summary.json',result);phase(r/'odcv','complete')
    except BaseException as e:
        phase(r/'odcv','held',error=repr(e));raise
    finally:stop_shim(backend,'odcv')


def linux_swe(action,backend):
    from src.eval.capabilities.swebench_mini.fleet_state import State,atomic
    from src.eval.capabilities.swebench_mini.fleet_worker import consume
    from scratch.gptoss_swe.finish_smoke import consume_pool
    r=ROOT/backend;root=(r/'swe').resolve();cfg=OmegaConf.load(r/'swe.yaml')
    if action=='consume':
        import requests
        check=requests.post(f'http://host.docker.internal:{C.ports[backend].swe}/tokenize',
            headers={'Authorization':'Bearer '+os.environ['TINKER_API_KEY']},
            json=dict(model='openai/gpt-oss-120b',messages=[dict(role='user',content='transport probe')],tools=[]),timeout=30)
        check.raise_for_status();assert check.json()['count']>0
        state=State(root);assert not state.path.exists()
        (root/'metadata').mkdir(parents=True,exist_ok=True)
        (root/'rollouts').mkdir(parents=True,exist_ok=True)
        for f in ('images.json','swebench_lite_test.json'):shutil.copyfile(CACHE/f,root/'metadata'/f)
        atomic(root/'metadata/manifest.json',dict(campaign=str(cfg.campaign),config=OmegaConf.to_container(cfg),target=backend,selected_ids=list(cfg.instance_ids)))
        atomic(state.path,dict(tasks={i:dict(status='pending',attempts=[]) for i in cfg.instance_ids},pods=[],deadline=None,phase='inference',halt=None))
        worker=OmegaConf.merge(cfg.worker,dict(root=str(root),serving=dict(context_window=int(C.swe_context)),endpoint_api_key_env='TINKER_API_KEY',
            fleet_owner_root=str(ROOT.resolve()),protocol_version=cfg.protocol,sampling=cfg.sampling))
        admission=dict(directory=str(root/'metadata/token-slots'),budget_tokens=int(C.swe_context)*int(C.swe_workers),expires=time.time()+7*86400,fairness_seconds=30)
        consume_pool(consume,f'http://host.docker.internal:{C.ports[backend].swe}/v1',worker,backend,list(cfg.instance_ids),
            {i:dict(max_total_attempt_records=3) for i in cfg.instance_ids},admission,int(C.swe_workers))
    else:
        from scratch.gptoss_swe.run import grade_predictions
        state=read(root/'metadata/state.json');assert all(t['status']!='running' for t in state['tasks'].values())
        preds={i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status']=='valid'}
        result=grade_predictions(cfg,root,preds,named(backend,'grade')) if preds else dict(n_graded=0,n_resolved=0)
        atomic(r/'swe-summary.json',dict(**result,states={i:t['status'] for i,t in state['tasks'].items()}))


def swe_container(action,backend):
    args=['docker','run','--name',named(backend,action),'--label',LABEL,'-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock','-e','TINKER_API_KEY']
    for v,p in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),('harness','src/eval/capabilities/swebench_mini/envs/harness')]:args+=['-v',f'lasr-gptoss-{v}-env:/work/{p}/.venv']
    with (ROOT/backend/(action+'.log')).open('x',encoding='utf-8') as log:
        subprocess.run(args+[IMAGE,'scratch/swebench_cpu_env/.venv/bin/python','-m','scratch.gptoss_swe.mxfp4_pilot',action,'--backend',backend],env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)


def swe(backend):
    r=ROOT/backend;claim(r/'swe-owner-started.json')
    try:
        start_shim(backend,'swe');phase(r/'swe','inference');swe_container('consume',backend)
        stop_shim(backend,'swe');save(r/'swe-inference-ended.json',dict(at=time.time()))
        phase(r/'swe','awaiting_grading')
    except BaseException as e:
        phase(r/'swe','held',error=repr(e));raise
    finally:stop_shim(backend,'swe')


def bootstrap():
    revision=read(ROOT/'model.json')['revision']
    return f'''set -euo pipefail
mkdir -p /workspace /root/.ssh /run/sshd
exec > >(tee -a /workspace/boot.log) 2>&1
printf '%s\\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null
/usr/sbin/sshd
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv venv /workspace/vllmenv --python 3.12
uv pip install --python /workspace/vllmenv/bin/python 'vllm=={C.vllm_version}' ninja hf_transfer
/workspace/vllmenv/bin/python -c "import torch; print(torch.ones(1,device='cuda')); print(torch.cuda.get_device_name()); print(torch.__version__)"
export HF_HOME=/workspace/hf
export VLLM_USE_FLASHINFER_SAMPLER=0
/workspace/vllmenv/bin/python -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b --revision {revision} --tokenizer-revision {revision} --served-model-name gptoss-mxfp4 --dtype bfloat16 --kv-cache-dtype auto --quantization mxfp4 --max-model-len 131072 --max-num-seqs {C.max_num_seqs} --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 --enable-prefix-caching --generation-config vllm --enable-prompt-tokens-details --host 0.0.0.0 --port 8000 > /workspace/vllm.log 2>&1
'''


def provision():
    import requests
    from src.infra import runpod
    quote=runpod.gpu_price(str(C.gpu));assert quote and quote<=float(C.gpu_rate_ceiling)
    runpod.validate_scheduled_provision()
    name='nika-'+str(C.campaign)
    assert not any(p.get('name')==name for p in runpod.active_pods())
    key=Path.home()/'.ssh/msm_audit';assert key.exists() and key.with_suffix('.pub').exists()
    deadline=(datetime.now(timezone.utc)+timedelta(hours=float(C.gpu_max_hours))).isoformat()
    script=bootstrap();(ROOT/'bootstrap.sh').write_text(script,encoding='utf-8',newline='\n')
    save(ROOT/'provision-intent.json',dict(name=name,quote=quote,deadline=deadline,at=time.time()))
    pod=runpod.provision_runpod(runpod.ProvisionSpec(gpu=str(C.gpu),count=1,disk_gb=200,max_hours=float(C.gpu_max_hours),
        pubkey_path=str(key.with_suffix('.pub')),terminate_at=deadline),name=name,start_script=script,
        env={'VLLM_API_KEY':os.environ['MXFP4_API_KEY']},ports=('8000/http','22/tcp'))
    save(ROOT/'pod.json',dict(id=pod,name=name,quote=quote,created=time.time(),deadline=deadline))
    runpod.start_watchdog(pod,int(float(C.gpu_max_hours)*3600),ROOT/'watchdog.log',parent_pid=os.getpid())
    host,port=runpod._ssh_endpoint(pod,timeout_s=300)
    args=['ssh','-i',str(key),'-p',str(port),'-o','StrictHostKeyChecking=accept-new','-o','BatchMode=yes','-o','ConnectTimeout=10']
    save(ROOT/'ssh.json',dict(host=host,port=port,key=str(key)))
    until=time.monotonic()+300
    while subprocess.run(args+['root@'+host,'true'],capture_output=True).returncode:
        if time.monotonic()>until:raise TimeoutError('SSH did not become ready')
        time.sleep(5)
    log=(ROOT/'tunnel.log').open('a')
    tunnel=subprocess.Popen(args+['-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-N','-L',f'0.0.0.0:{C.raw_port}:127.0.0.1:8000','root@'+host],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    save(ROOT/'tunnel.json',dict(pid=tunnel.pid))
    until=read(ROOT/'pod.json')['created']+int(C.bootstrap_timeout)
    while time.time()<until:
        assert tunnel.poll() is None,'SSH tunnel exited'
        subprocess.run(['docker','ps','-q'],capture_output=True) # Avoid Docker Resource Saver during model loading.
        try:
            response=requests.get(f'http://127.0.0.1:{C.raw_port}/v1/models',headers={'Authorization':'Bearer '+os.environ['MXFP4_API_KEY']},timeout=3)
            response.raise_for_status();identity=response.json();assert identity['data'][0]['id']=='gptoss-mxfp4'
            save(ROOT/'server-identity.json',identity);return pod,tunnel
        except requests.RequestException:time.sleep(10)
    raise TimeoutError('GPU bootstrap exceeded 20-minute cutoff')


def collect_gpu():
    if not (ROOT/'ssh.json').exists():return
    s=read(ROOT/'ssh.json')
    for filename in ('boot.log','vllm.log'):
        result=subprocess.run(['ssh','-i',s['key'],'-p',str(s['port']),'-o','BatchMode=yes','-o','ConnectTimeout=10',
            'root@'+s['host'],'cat /workspace/'+filename],capture_output=True,timeout=30)
        (ROOT/filename).write_bytes(result.stdout)


def run():
    import secrets
    from src.infra import runpod
    os.environ.update(environment());os.environ['MXFP4_API_KEY']=secrets.token_urlsafe(36)
    claim(ROOT/'owner-started.json')
    save(ROOT/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=OmegaConf.to_container(C)))
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit source before launch'
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(ROOT/'source.tar.gz'),'HEAD'],check=True)
    jobs=[];tunnel=None;pod=None
    try:
        phase(ROOT,'bootstrapping_gpu');pod,tunnel=provision()
        # Bridge checks must pass before any benchmark rollout. Check raw token IDs live.
        qualify_live()
        for backend in ('mxfp4','tinker'):
            for kind in ('odcv','swe'):
                log=(ROOT/backend/(kind+'-owner.log')).open('x',encoding='utf-8')
                p=subprocess.Popen([sys.executable,'-m','scratch.gptoss_swe.mxfp4_pilot',kind,'--backend',backend],env=environment(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                jobs.append((backend,kind,p,log))
        phase(ROOT,'inference')
        # Release GPU as soon as its two inference workloads end, even if judges/reference still work.
        while True:
            done=all((ROOT/'mxfp4'/(kind+'-inference-ended.json')).exists() or p.poll() is not None for backend,kind,p,_ in jobs if backend=='mxfp4')
            if done:break
            subprocess.run(['docker','ps','-q'],capture_output=True);time.sleep(15)
        collect_gpu();assert runpod.terminate(pod);save(ROOT/'gpu-cleanup.json',dict(pod=pod,terminated=True,at=time.time()))
        tunnel.terminate();tunnel.wait(timeout=20);pod=None;tunnel=None
        exits={backend+'/'+kind:p.wait() for backend,kind,p,_ in jobs};save(ROOT/'exits.json',exits)
        phase(ROOT,'grading')
        for backend in ('mxfp4','tinker'):
            if (ROOT/backend/'swe/metadata/state.json').exists():swe_container('grade',backend)
        phase(ROOT,'ready_for_publication',exits=exits)
        cleanup()
        publish()
    except BaseException as e:
        phase(ROOT,'held',error=repr(e));raise
    finally:
        # Even a readiness failure after rent is covered, including when provision() did not return.
        if (ROOT/'pod.json').exists() and not (ROOT/'gpu-cleanup.json').exists():
            try:collect_gpu()
            finally:
                owned=read(ROOT/'pod.json')['id'];gone=runpod.terminate(owned)
                save(ROOT/'gpu-cleanup.json',dict(pod=owned,terminated=gone,at=time.time()))
        if tunnel is not None:tunnel.terminate()


def qualify_live():
    # Runs in the cached Linux environment, where the exact Harmony dependencies are installed.
    args=['docker','run','--rm','--label',LABEL,'-v',f'{Path.cwd()}:/work','-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
        '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface','-e','MXFP4_API_KEY',IMAGE,
        'src/infra/endpoints/tinker_env/.venv/bin/python','-m','scratch.gptoss_swe.mxfp4_pilot','probe']
    with (ROOT/'probe.log').open('x',encoding='utf-8') as log:subprocess.run(args,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)


def probe():
    import asyncio
    from src.infra.endpoints.tinker_harmony import HarmonyRenderer,generation_prompt,parse_completion
    from scratch.gptoss_swe.mxfp4_bridge import RawClient
    renderer=HarmonyRenderer(reasoning_effort='medium',current_date='2026-10-12')
    client=RawClient(f'http://host.docker.internal:{C.raw_port}',os.environ['MXFP4_API_KEY'],ROOT/'probe-raw')
    async def checks():
        receipts=[]
        for text in ('Reply with the number 7.', 'What is 2 plus 2? Reply briefly.'):
            prompt=generation_prompt(renderer,[dict(role='user',content=text)])
            result=await client.sample_async(prompt=prompt,num_samples=1,sampling_params=SimpleNamespace(max_tokens=512,temperature=0,top_p=1,top_k=-1,stop=renderer.get_stop_sequences()))
            seq=result.sequences[0];msg,finish,boundary=parse_completion(renderer,seq.tokens,seq.stop_reason)
            assert boundary=='final' and finish=='stop',(msg,finish,boundary)
            receipts.append(dict(message=msg,boundary=boundary,output_tokens=len(seq.tokens)))
        # Exercise the actual shared OpenAI-chat -> Harmony -> tool -> Harmony path.
        from src.infra.endpoints.tinker_server import build_messages
        from tinker_cookbook.renderers.base import ToolCall
        from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker
        runtime=SimpleNamespace(renderer=renderer,tool_call_type=ToolCall,convert_tools=openai_tools_to_tinker)
        tools=[dict(type='function',function=dict(name='get_value',description='Read the current integer value.',parameters=dict(type='object',properties={},additionalProperties=False)))]
        messages=[dict(role='user',content='Use get_value to read the current value, then report it.')]
        for turn in range(2):
            prompt=generation_prompt(renderer,build_messages(messages,tools,runtime))
            result=await client.sample_async(prompt=prompt,num_samples=1,sampling_params=SimpleNamespace(max_tokens=1024,temperature=0,top_p=1,top_k=-1,stop=renderer.get_stop_sequences()))
            seq=result.sequences[0];msg,finish,boundary=parse_completion(renderer,seq.tokens,seq.stop_reason)
            receipts.append(dict(turn=turn,message=msg,boundary=boundary,output_tokens=len(seq.tokens)))
            if turn==0:
                assert boundary in ('handoff','tool_call_return') and len(msg.get('tool_calls',[]))==1,(msg,boundary)
                call=msg['tool_calls'][0];assert call['function']['name']=='get_value'
                assert json.loads(call['function']['arguments'])=={}
                call['id']='probe_call';msg['harmony_boundary']=boundary
                messages.extend([msg,dict(role='tool',tool_call_id='probe_call',content='42')])
            else:assert boundary=='final' and '42' in msg['content'],(msg,boundary)
        save(ROOT/'live-qualification.json',dict(passed=True,checks=receipts))
    asyncio.run(checks())


def cleanup():
    # Only this pilot's labels/projects, including stopped containers. Never global pruning.
    all_ids=subprocess.check_output(['docker','ps','-aq'],text=True).split()
    objects=json.loads(subprocess.check_output(['docker','inspect',*all_ids])) if all_ids else []
    projects=[]
    for backend in ('tinker','mxfp4'):
        p=ROOT/backend/'completed-passes.json'
        if p.exists():
            for row in read(p):
                tag=hashlib.sha256(f"{named(backend,'odcv')}:{Path(row['path']).resolve()}".encode()).hexdigest()[:12]
                projects.append('odcv-'+tag+'-')
    owned=[]
    for obj in objects:
        labels=obj['Config'].get('Labels') or {}
        if (labels.get('lasr.campaign')==str(C.campaign) or
            labels.get('lasr_campaign') in [named(b,'swe') for b in ('tinker','mxfp4')] or
            any(labels.get('com.docker.compose.project','').startswith(p) for p in projects)):
            assert not obj['State']['Running'],'Owned container still active: '+obj['Name']
            owned.append(obj['Id'])
    save(ROOT/'before-cleanup.json',dict(containers=owned,compose_prefixes=projects))
    if owned:subprocess.run(['docker','rm',*owned],check=True)
    remaining=subprocess.check_output(['docker','ps','-aq','--filter','label='+LABEL],text=True).split()
    assert not remaining
    save(ROOT/'cleanup.json',dict(remaining=remaining,removed=owned,at=time.time()))


def publish():
    from huggingface_hub import HfApi,hf_hub_url
    import requests
    package=ROOT/'publication-package';assert not package.exists(),'Preserve prior publication'
    for folder in ('rollouts','results','metadata'):(package/folder).mkdir(parents=True)
    report={}
    for backend in ('tinker','mxfp4'):
        r=ROOT/backend;report[backend]={}
        for kind in ('odcv','swe'):
            if (r/(kind+'-summary.json')).exists():
                report[backend][kind]=read(r/(kind+'-summary.json'))
                shutil.copyfile(r/(kind+'-summary.json'),package/'results'/f'{backend}-{kind}.json')
        for directory in ('odcv/rollouts','swe/rollouts'):
            if (r/directory).exists():shutil.copytree(r/directory,package/'rollouts'/backend/directory)
        with tarfile.open(package/'metadata'/f'{backend}-raw.tar.gz','w:gz') as archive:
            for p in sorted(r.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:archive.add(p,arcname=p.relative_to(r).as_posix())
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.json','.log','.sh','.gz'):
            shutil.copyfile(p,package/'metadata'/p.name)
    complete=all(report.get(b,{}).get('odcv',{}).get('passes',{}).get('n_transcripts')==30 and
        report[b]['odcv'].get('n_dropped_all_na')==0 and report[b]['odcv'].get('progress',{}).get('n_dropped_all_na')==0 and
        report[b].get('swe',{}).get('n_graded')==4 for b in ('tinker','mxfp4'))
    save(package/'results/summary.json',dict(coverage_complete=complete,results=report))
    (package/'README.md').write_text('---\ntags: [eval-run, eval:odcv, eval:swebench-lite, model:gptoss120b]\n---\n\n# Base GPT-OSS-120B backend pilot\n\nMatched fresh random tasks: 10 ODCV cells times three passes, four SWE-bench Lite tasks per backend. Released MXFP4 on one H200 versus Tinker. Same reference Harmony token rendering and parsing. No custom coaching or JSON repair. ODCV112000 context, remaining-context response allowance,50cycles,T.7/p.95/k20. SWE131072 context,16384response,262144generated/task,500steps,T1/p1/k disabled.\n\nThis compares serving backends as well as precision; Tinker exact internal numerical precision is unverified. Small-sample smoke, not an equivalence test or a replacement benchmark. MXFP4 raw tokens are not retokenized. The reused wire field tinker_metadata contains Harmony diagnostics, not a claim that vLLM was Tinker. Limits/submission do not imply alignment or correctness. See results/summary.json for coverage and separate judging axes. Costs are scoped estimates, not invoices.\n',encoding='utf-8')
    hashes={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*') if p.is_file()}
    save(package/'metadata/file-hashes.json',hashes)
    hashes['metadata/file-hashes.json']=hashlib.sha256((package/'metadata/file-hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=environment()['HF_TOKEN']);api.create_repo(str(C.hf_repo),repo_type='dataset',exist_ok=True)
    revision=api.upload_folder(repo_id=str(C.hf_repo),repo_type='dataset',folder_path=package,commit_message='Matched base-only backend pilot with full raw evidence').oid
    receipt=dict(repo=str(C.hf_repo),revision=revision,complete=False,coverage_complete=complete,hashes=hashes)
    save(ROOT/'publication.json',receipt)
    def verify(item):
        name,digest=item
        url=hf_hub_url(str(C.hf_repo),name,repo_type='dataset',revision=revision)
        sha=hashlib.sha256()
        # Windows HF-cache symlinks failed the previous run's closeout; hash pinned bytes directly.
        with requests.get(url,headers={'Authorization':'Bearer '+environment()['HF_TOKEN']},stream=True,timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_content(1024*1024):sha.update(chunk)
        assert sha.hexdigest()==digest,name
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(verify,hashes.items()))
    receipt['complete']=True;save(ROOT/'publication.json',receipt)
    phase(ROOT,'complete' if complete else 'published_partial',publication=receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','odcv','swe','consume','grade','probe','publish','cleanup'])
    parser.add_argument('--backend',choices=['tinker','mxfp4']);args=parser.parse_args()
    if args.action in ('consume','grade'):linux_swe(args.action,args.backend)
    elif args.action in ('odcv','swe'):
        os.environ.update(environment());globals()[args.action](args.backend)
    else:globals()[args.action]()
