# ABOUTME: Finish both selected SWE smoke arms under explicit spending-cap removal, preserving histories.
# ABOUTME: Independent arm owners wait for old inference, keep paid accounting, grade and publish all evidence.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from omegaconf import OmegaConf
from scratch.gptoss_swe.openai_smoke import environment,save,IMAGE,TARGETS

C=OmegaConf.load('scratch/gptoss_swe/finish_smoke.yaml')
ROOT=Path(C.root)


def old_root(arm):
    return Path(C.base_prior) if arm=='base' else Path(C.original)/arm


def budget_path(arm):
    return Path(C.original)/arm/'swe-budget.json'


def name(arm):return str(C.campaign)+'-'+arm
def port(arm):return int(C.port_base)+(10 if arm=='control' else 0)


def running(container):
    p=subprocess.run(['docker','inspect',container],capture_output=True,text=True)
    if p.returncode:
        assert 'No such' in p.stderr,p.stderr
        return False
    return json.loads(p.stdout)[0]['State']['Running']


def prepare_arm(arm):
    r=ROOT/arm;prior=old_root(arm)
    save(r/'status.json',dict(phase='waiting_for_prior_inference',at=time.time()))
    old_names=(['gptoss-refresh-base-recovery-20261009-consume','gptoss-refresh-base-recovery-20261009-shim'] if arm=='base' else ['lasr-gptoss-refresh-control-swe-driver','lasr-gptoss-refresh-control-swe-shim'])
    while any(running(n) for n in old_names):time.sleep(15)
    state=json.loads((prior/'swe/metadata/state.json').read_text())
    assert len(state['tasks'])==10 and not any(t['status']=='running' for t in state['tasks'].values())
    assert json.loads((prior/'swe/metadata/manifest.json').read_text())['target']==TARGETS[arm]
    ledger=json.loads(budget_path(arm).read_text())
    assert all(v['state']=='completed' for v in ledger['requests'].values()),'Ambiguous paid request: hold'
    assert ledger.get('authorization_history') and (ledger['ceiling_usd'] is None or ledger['ceiling_usd']==float('inf'))
    with (r/'preparation-claim.json').open('x') as f:json.dump(dict(at=time.time(),source=str(prior)),f)
    cfg=OmegaConf.load(prior/('recovery.yaml' if arm=='base' else 'swe.yaml'))
    records={}
    for folder in ('metadata','rollouts'):
        for p in sorted((prior/'swe'/folder).rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(prior/'swe');dest=r/'swe'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            digest=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest;records[rel.as_posix()]=digest
    save(r/'prior-state.json',state)
    save(r/'copy-hashes.json',records)
    allowances={}
    for iid,t in state['tasks'].items():
        excluded=[]
        for attempt in t['attempts']:
            p=prior/'swe/rollouts'/iid/attempt['id']/'checkpoint.traj.json'
            if attempt.get('valid') is False and p.exists() and 'Frozen Tinker spending ceiling reached' in p.read_text():excluded.append(attempt['id'])
        allowances[iid]=dict(budget_denial_attempt_ids=excluded,max_total_attempt_records=int(C.infrastructure_attempts)+len(excluded))
    # Do not erase attempts. Only explicitly budget-denied attempts are excluded from the infrastructure retry count.
    save(r/'retry-accounting.json',allowances)
    state['halt']=None;state['deadline']=None
    save(r/'swe/metadata/state.json',state)
    cfg.output_root='/work/'+(r/'swe').as_posix()
    cfg.worker.model_request_timeout_seconds=int(C.request_timeout);cfg.worker.model_request_attempts=int(C.request_attempts)
    cfg.worker.max_infrastructure_attempts=int(C.infrastructure_attempts);cfg.worker.max_infrastructure_failures=int(C.infrastructure_breaker)
    cfg.tinker.budget_usd=None
    OmegaConf.save(cfg,r/'config.yaml')
    save(r/'status.json',dict(phase='prepared',at=time.time()))


def linux(action,arm):
    from src.eval.capabilities.swebench_mini.fleet_state import read,atomic,lock
    r=ROOT/arm;cfg=OmegaConf.load(r/'config.yaml');root=(r/'swe').resolve()
    p=budget_path(arm)
    with lock(p.with_suffix('.lock')):
        d=read(p);assert d.get('authorization_history') and (d['ceiling_usd'] is None or d['ceiling_usd']==float('inf'))
        d['ceiling_usd']=None;atomic(p,d)
    if action=='consume':
        import urllib.request
        from src.eval.capabilities.swebench_mini.fleet_worker import consume
        endpoint=f'http://host.docker.internal:{port(arm)}/v1'
        req=urllib.request.Request(endpoint[:-3]+'/tokenize',data=json.dumps(dict(model='openai/gpt-oss-120b',messages=[dict(role='user',content='transport probe')],tools=[])).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['TINKER_API_KEY']})
        with urllib.request.urlopen(req,timeout=60) as response:assert json.load(response)['count']>0
        worker=OmegaConf.merge(cfg.worker,dict(root=str(root),serving=dict(context_window=cfg.tinker.context_window),endpoint_api_key_env='TINKER_API_KEY',fleet_owner_root=str(Path(cfg.tinker.budget_ledger).parent),protocol_version=cfg.protocol,sampling=cfg.sampling))
        admission=dict(directory=str(root/'metadata/completion-token-slots'),budget_tokens=int(cfg.tinker.context_window),expires=time.time()+7*86400,fairness_seconds=30)
        allowances=read(r/'retry-accounting.json')
        for iid in cfg.instance_ids:
            if read(root/'metadata/state.json')['tasks'][iid]['status']=='valid':continue
            worker.max_infrastructure_attempts=allowances[iid]['max_total_attempt_records']
            consume(endpoint,'hosted_vllm/openai/gpt-oss-120b',worker,arm+'-completion-0',[iid],time.time()+7*86400,admission=admission)
    elif action=='grade':
        from scratch.gptoss_swe.run import grade_predictions
        state=read(root/'metadata/state.json');assert not any(t['status']=='running' for t in state['tasks'].values())
        preds={i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status']=='valid'}
        result=grade_predictions(cfg,root,preds,name(arm)) if preds else dict(n_graded=0,n_resolved=0,resolved_ids=[])
        atomic(r/'summary.json',dict(**result,target=TARGETS[arm],n_selected=10,states={i:t['status'] for i,t in state['tasks'].items()},halt=state.get('halt')))


def container(action,arm,env):
    args=['docker','run','--name',name(arm)+'-'+action,'--label','lasr.campaign='+str(C.campaign),'-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock','-e','TINKER_API_KEY']
    for v,p in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),('harness','src/eval/capabilities/swebench_mini/envs/harness')]:args+=['-v',f'lasr-gptoss-{v}-env:/work/{p}/.venv']
    with (ROOT/arm/(action+'.log')).open('x',encoding='utf-8') as f:
        code=subprocess.run(args+[IMAGE,'scratch/swebench_cpu_env/.venv/bin/python','-m','scratch.gptoss_swe.finish_smoke',action,'--arm',arm],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    save(ROOT/arm/(action+'-exit.json'),dict(code=code,at=time.time()))
    return code


def arm_run(arm):
    import requests
    r=ROOT/arm;r.mkdir(exist_ok=True)
    with (r/'owner-started.json').open('x') as f:json.dump(dict(pid=os.getpid(),at=time.time()),f)
    env=environment();shim=name(arm)+'-shim';started=False
    try:
        prepare_arm(arm)
        assert container('normalize',arm,env)==0
        state=json.loads((r/'swe/metadata/state.json').read_text())
        if any(t['status']!='valid' for t in state['tasks'].values()):
            args=['docker','run','-d','--name',shim,'--label','lasr.campaign='+str(C.campaign),'-p',f'127.0.0.1:{port(arm)}:1234','--memory','3g','--cpus','2','-v',f'{Path.cwd()}:/work','-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv','-v','lasr-gptoss-hf-cache:/root/.cache/huggingface']
            settings=dict(TINKER_CKPT=TARGETS[arm],TINKER_CONTEXT_WINDOW='131072',DEFAULT_MAX_TOKENS='16384',TINKER_BUDGET_USD='unlimited',TINKER_BUDGET_LEDGER='/work/'+budget_path(arm).as_posix(),TINKER_TRACE_DIR='/work/'+(r/'traces').as_posix(),TINKER_BIND_HOST='0.0.0.0',TINKER_RENDER_DATE='2026-10-09',REASONING_LEVEL='medium')
            for k,v in settings.items():args+=['-e',k+'='+v]
            for k in ('TINKER_API_KEY','HF_TOKEN'):args+=['-e',k]
            cid=subprocess.check_output(args+[IMAGE,'src/infra/endpoints/tinker_env/.venv/bin/python','-m','src.infra.endpoints.tinker_server'],env=env,text=True).strip();started=True
            save(r/'shim.json',dict(container=cid,settings=settings))
            for _ in range(180):
                try:
                    res=requests.get(f'http://localhost:{port(arm)}/v1/models',headers={'Authorization':'Bearer '+env['TINKER_API_KEY']},timeout=5);res.raise_for_status();assert res.json()['data'][0]['checkpoint']==TARGETS[arm];break
                except requests.RequestException:time.sleep(2)
            else:raise TimeoutError('Shim readiness')
            save(r/'status.json',dict(phase='inference',at=time.time()))
            container('consume',arm,env)
            with (r/'shim.log').open('w',encoding='utf-8') as f:subprocess.run(['docker','logs',shim],stdout=f,stderr=subprocess.STDOUT)
            subprocess.run(['docker','stop',shim],check=True);started=False
        save(r/'status.json',dict(phase='grading',at=time.time()));assert container('grade',arm,env)==0
        save(r/'status.json',dict(phase='graded',at=time.time()))
    except BaseException as exc:
        save(r/'status.json',dict(phase='held',at=time.time(),error=repr(exc)));raise
    finally:
        if started:subprocess.run(['docker','stop',shim],check=True)


def publish():
    from huggingface_hub import HfApi,hf_hub_download
    env=environment();out=ROOT/'package';out.mkdir(exist_ok=True)
    needles=[v.encode() for k,v in env.items() if len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    summaries={}
    for arm in TARGETS:
        s=json.loads((ROOT/arm/'summary.json').read_text());d=json.loads(budget_path(arm).read_text());assert d['ceiling_usd'] is None
        s['cumulative_inference_usd']=sum(v['upper_usd'] for v in d['requests'].values());summaries[arm]=s
        save(ROOT/arm/'final-budget.json',d)
        sources=[(ROOT/arm,'completion'),(Path(C.original)/arm,'original')]
        if arm=='base':sources.append((Path(C.base_prior),'prior-recovery'))
        with tarfile.open(out/(arm+'.tar.gz'),'w:gz') as tar:
            for source,prefix in sources:
                for p in sorted(source.rglob('*')):
                    if not p.is_file() or 'package' in p.relative_to(source).parts:continue
                    assert not any(v in p.read_bytes() for v in needles),p
                    tar.add(p,arcname=prefix+'/'+p.relative_to(source).as_posix(),recursive=False)
    save(out/'summary.json',summaries)
    for f in ('spending-cap-removal.json','source.tar.gz','owner-started.json'):shutil.copyfile(ROOT/f,out/f)
    lines=['---','license: mit','---','# GPT-OSS base/control SWE ten-task smoke completion','','User explicitly removed inference spending ceilings. Cost accounting and all original attempts retained. No valid model outcome rerun. Budget-denied attempts excluded from infrastructure retry eligibility only. Numerical model limits unchanged.','','| Arm | Resolved / graded | Selected | Cumulative inference USD |','|---|---:|---:|---:|']
    for arm,s in summaries.items():lines.append(f"| {arm} | {s['n_resolved']}/{s['n_graded']} | 10 | {s['cumulative_inference_usd']:.6f} |")
    lines+=['','Costs include earlier original/recovery inference: never sum them again. Not provider invoices. Partial coverage, if present in summary, must not be called a completed benchmark.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='hashes.json'};save(out/'hashes.json',hashes);hashes['hashes.json']=hashlib.sha256((out/'hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=env['HF_TOKEN']);api.create_repo(str(C.hf_repo),repo_type='dataset',exist_ok=True)
    rev=api.upload_folder(repo_id=str(C.hf_repo),repo_type='dataset',folder_path=out,commit_message='Complete authorized SWE smoke with preserved outcomes and uncapped accounting').oid
    for f,h in hashes.items():
        p=hf_hub_download(str(C.hf_repo),f,repo_type='dataset',revision=rev,token=env['HF_TOKEN']);assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save(ROOT/'publication.json',dict(repo=str(C.hf_repo),revision=rev,complete=True,sha256=hashes))


def run():
    with (ROOT/'owner-started.json').open('x') as f:json.dump(dict(pid=os.getpid(),at=time.time(),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f)
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(ROOT/'source.tar.gz'),'HEAD'],check=True)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    def invoke(arm):
        with (ROOT/(arm+'-owner.log')).open('x',encoding='utf-8') as f:return subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.finish_smoke','arm','--arm',arm],env=environment(),stdout=f,stderr=subprocess.STDOUT).returncode
    try:
        save(ROOT/'status.json',dict(phase='running',at=time.time()))
        with ThreadPoolExecutor(max_workers=2) as pool:exits=dict(zip(TARGETS,pool.map(invoke,TARGETS)))
        save(ROOT/'arm-exits.json',exits);assert not any(exits.values()),exits
        save(ROOT/'status.json',dict(phase='publishing',at=time.time()));publish()
        ids=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(C.campaign)],text=True).split()
        for cid in ids:
            assert not running(cid);subprocess.run(['docker','rm',cid],check=True)
        save(ROOT/'cleanup.json',dict(at=time.time(),owned_remaining=[]));save(ROOT/'status.json',dict(phase='complete',at=time.time()))
    except BaseException as exc:
        save(ROOT/'status.json',dict(phase='held',error=repr(exc),at=time.time()));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','arm','consume','grade','normalize','publish']);parser.add_argument('--arm',choices=['base','control']);args=parser.parse_args()
    if args.action in ('consume','grade','normalize'):linux(args.action,args.arm)
    elif args.action=='arm':arm_run(args.arm)
    else:globals()[args.action]()
