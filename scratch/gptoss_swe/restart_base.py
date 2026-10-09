# ABOUTME: Recover only infrastructure-invalid and unstarted base SWE tasks with the original budget.
# ABOUTME: Copies immutable history, uses bounded standard workers, then grades and publishes retained outcomes.
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from omegaconf import OmegaConf
from scratch.gptoss_swe.openai_smoke import environment, save, IMAGE

C = OmegaConf.load('scratch/gptoss_swe/restart_base.yaml')
ROOT, PRIOR = Path(C.root), Path(C.prior)
NAME = str(C.campaign)


def validate(state, ledger):
    assert len(state['tasks']) == 10
    assert state['halt'] == 'infrastructure failure circuit breaker'
    assert ledger['ceiling_usd'] == 12 and ledger['authorized_checkpoints'] == ['tinker://base']
    assert all(r['state'] == 'completed' for r in ledger['requests'].values()), 'Ambiguous paid calls cannot be retried'
    assert sum(r['upper_usd'] for r in ledger['requests'].values()) < 12
    invalid=[]
    for iid,t in state['tasks'].items():
        assert t['status'] in ('valid','invalid','pending'), 'Live task cannot be copied'
        if t['status']=='invalid':
            assert len(t['attempts'])==1 and t['attempts'][0]['exit_status']=='URLError'
            invalid.append(iid)
        elif t['status']=='pending': assert not t['attempts']
        else: assert t['attempts'][-1]['valid'] is True
    assert invalid==['pylint-dev__pylint-5859']
    result=deepcopy(state)
    result['halt']=None
    return result


def prepare():
    assert not ROOT.exists()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    for name in ('lasr-gptoss-refresh-base-swe-shim','lasr-gptoss-refresh-base-swe-driver'):
        assert not json.loads(subprocess.check_output(['docker','inspect',name]))[0]['State']['Running']
    state=json.loads((PRIOR/'swe/metadata/state.json').read_text())
    ledger=json.loads((PRIOR/'swe-budget.json').read_text())
    recovered=validate(state,ledger)
    cfg=OmegaConf.load(PRIOR/'swe.yaml')
    assert cfg.tinker.budget_ledger=='/work/'+(PRIOR/'swe-budget.json').as_posix()
    assert json.loads((PRIOR/'swe/metadata/manifest.json').read_text())['target']=='tinker://base'
    invalid=next((PRIOR/'swe/rollouts/pylint-dev__pylint-5859').glob('*/checkpoint.traj.json'))
    assert 'Network is unreachable' in invalid.read_text() and 'prompt_tokens' in invalid.read_text()
    ROOT.mkdir()
    records={}
    for folder in ('metadata','rollouts'):
        for path in sorted((PRIOR/'swe'/folder).rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(PRIOR/'swe'); dest=ROOT/'swe'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
            records[rel.as_posix()]=digest
    save(ROOT/'prior-state.json',state);save(ROOT/'prior-budget.json',ledger)
    shutil.copyfile(PRIOR/'swe.yaml',ROOT/'prior-config.yaml')
    cfg.output_root='/work/'+(ROOT/'swe').as_posix()
    cfg.worker.model_request_timeout_seconds=int(C.request_timeout_seconds)
    cfg.worker.model_request_attempts=int(C.request_attempts)
    cfg.worker.max_infrastructure_attempts=int(C.infrastructure_attempts)
    cfg.worker.max_infrastructure_failures=int(C.infrastructure_failure_limit)
    OmegaConf.save(cfg,ROOT/'recovery.yaml')
    save(ROOT/'swe/metadata/state.json',recovered)
    save(ROOT/'recovery-receipt.json',dict(at=time.time(),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        prior=str(PRIOR),authorization='User: restart base; prior request: 120-minute timeout and three infrastructure attempts',
        copied_sha256=records,prior_budget_usd=sum(r['upper_usd'] for r in ledger['requests'].values()),
        preserved_valid=[i for i,t in state['tasks'].items() if t['status']=='valid'],
        retriable_invalid=['pylint-dev__pylint-5859'],policy=OmegaConf.to_container(C)))
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(ROOT/'source.tar.gz'),'HEAD'],check=True)
    save(ROOT/'status.json',dict(phase='prepared',at=time.time()))


def linux(action):
    from src.eval.capabilities.swebench_mini.fleet_state import read,atomic
    cfg=OmegaConf.load(ROOT/'recovery.yaml');root=(ROOT/'swe').resolve()
    if action=='consume':
        from src.eval.capabilities.swebench_mini.fleet_worker import consume
        # Verify the same container-to-host path used by token counting without inference.
        import urllib.request
        counts=[]
        for _ in range(5):
            req=urllib.request.Request(f'http://host.docker.internal:{C.port}/tokenize',
                data=json.dumps(dict(model='openai/gpt-oss-120b',messages=[dict(role='user',content='transport probe')],tools=[],add_generation_prompt=True)).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['TINKER_API_KEY']})
            with urllib.request.urlopen(req,timeout=60) as response:counts.append(json.load(response)['count'])
        assert all(n>0 for n in counts)
        atomic(ROOT/'transport-probes.json',dict(counts=counts,inference_requests=0,host_gateway_override=False))
        worker=OmegaConf.merge(cfg.worker,dict(root=str(root),serving=dict(context_window=cfg.tinker.context_window),
            endpoint_api_key_env='TINKER_API_KEY',fleet_owner_root=str(Path(cfg.tinker.budget_ledger).parent),
            protocol_version=cfg.protocol,sampling=cfg.sampling))
        admission=dict(directory=str(root/'metadata/recovery-token-slots'),budget_tokens=131072,expires=time.time()+7*86400,fairness_seconds=30)
        consume(f'http://host.docker.internal:{C.port}/v1','hosted_vllm/openai/gpt-oss-120b',worker,
                'base-recovery-0',list(cfg.instance_ids),time.time()+7*86400,admission=admission)
    else:
        from scratch.gptoss_swe.run import grade_predictions
        state=read(root/'metadata/state.json');assert not any(t['status']=='running' for t in state['tasks'].values())
        preds={i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status']=='valid'}
        result=grade_predictions(cfg,root,preds,NAME)
        atomic(ROOT/'summary.json',dict(**result,n_valid=len(preds),n_selected=10,
            n_interrupted=sum(t['status']=='invalid' for t in state['tasks'].values()),
            n_unstarted=sum(t['status']=='pending' for t in state['tasks'].values()),halt=state.get('halt')))


def docker(action, env):
    args=['docker','run','--name',NAME+'-'+action,'--label','lasr.campaign='+NAME,
          '-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock','-e','TINKER_API_KEY']
    for volume,path in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),('harness','src/eval/capabilities/swebench_mini/envs/harness')]:
        args+=['-v',f'lasr-gptoss-{volume}-env:/work/{path}/.venv']
    # Docker Desktop resolves host.docker.internal itself; no host-gateway override.
    with (ROOT/(action+'.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run(args+[IMAGE,'scratch/swebench_cpu_env/.venv/bin/python','-m','scratch.gptoss_swe.restart_base',action],env=env,stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/(action+'-exit.json'),dict(code=result.returncode,at=time.time()))
    return result.returncode


def publish():
    from huggingface_hub import HfApi,hf_hub_download
    env=environment();out=ROOT/'package';out.mkdir(exist_ok=True)
    needles=[v.encode() for k,v in env.items() if len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    with tarfile.open(out/'evidence.tar.gz','w:gz') as tar:
        for source,prefix in [(ROOT,'recovery'),(PRIOR,'original-base')]:
            for file in sorted(source.rglob('*')):
                if not file.is_file() or out in file.parents:continue
                assert not any(n in file.read_bytes() for n in needles),file
                tar.add(file,arcname=prefix+'/'+file.relative_to(source).as_posix(),recursive=False)
    summary=json.loads((ROOT/'summary.json').read_text());ledger=json.loads((PRIOR/'swe-budget.json').read_text())
    cost=sum(r['upper_usd'] for r in ledger['requests'].values());assert cost<=12
    summary.update(inference_accounted_usd=cost,includes_original_attempts=True,target='tinker://base')
    save(out/'summary.json',summary)
    (out/'README.md').write_text('---\nlicense: mit\n---\n# Base GPT-OSS local SWE smoke infrastructure recovery\n\n'
        'User-authorized recovery of one network-interrupted task plus seven unstarted tasks. Two valid limit outcomes retained unchanged. '
        'Original $12 inference ledger shared, never reset; all historical attempts retained. One worker, local Docker/Tinker. '
        '7200s request timeout, two request attempts, three total infrastructure attempts, breaker six. Original sampling and numerical model limits unchanged.\n\n'
        f"Resolved {summary['n_resolved']}/{summary['n_graded']} graded; interrupted {summary['n_interrupted']}, unstarted {summary['n_unstarted']} of ten selected. "
        f'Inference accounting including original attempts ${cost:.6f}, not invoice. Partial coverage is not a full benchmark.\n',encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='hashes.json'}
    save(out/'hashes.json',hashes);hashes['hashes.json']=hashlib.sha256((out/'hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=env['HF_TOKEN']);api.create_repo(str(C.hf_repo),repo_type='dataset',exist_ok=True)
    rev=api.upload_folder(repo_id=str(C.hf_repo),repo_type='dataset',folder_path=out,commit_message='Publish base infrastructure recovery, preserving original ledger and outcomes').oid
    for name,digest in hashes.items():
        p=hf_hub_download(str(C.hf_repo),name,repo_type='dataset',revision=rev,token=env['HF_TOKEN'])
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
    save(ROOT/'publication.json',dict(repo=str(C.hf_repo),revision=rev,complete=True,sha256=hashes))


def run():
    import requests
    with (ROOT/'owner-started.json').open('x') as f:json.dump(dict(pid=os.getpid(),at=time.time()),f)
    env=environment();name=NAME+'-shim';started=False
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        cfg=OmegaConf.load(ROOT/'recovery.yaml')
        assert not subprocess.check_output(['docker','ps','-q','--filter','name=lasr-gptoss-refresh-base-swe-shim'],text=True).strip()
        args=['docker','run','-d','--name',name,'--label','lasr.campaign='+NAME,'-p',f'127.0.0.1:{C.port}:1234','--memory','3g','--cpus','2',
              '-v',f'{Path.cwd()}:/work','-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv','-v','lasr-gptoss-hf-cache:/root/.cache/huggingface']
        settings=dict(TINKER_CKPT='tinker://base',TINKER_CONTEXT_WINDOW='131072',DEFAULT_MAX_TOKENS='16384',TINKER_BUDGET_USD='12',
            TINKER_BUDGET_LEDGER=str(cfg.tinker.budget_ledger),TINKER_TRACE_DIR='/work/'+(ROOT/'traces').as_posix(),TINKER_BIND_HOST='0.0.0.0',TINKER_RENDER_DATE='2026-10-09',REASONING_LEVEL='medium')
        for k,v in settings.items():args+=['-e',k+'='+v]
        for k in ('TINKER_API_KEY','HF_TOKEN'):args+=['-e',k]
        cid=subprocess.check_output(args+[IMAGE,'src/infra/endpoints/tinker_env/.venv/bin/python','-m','src.infra.endpoints.tinker_server'],env=env,text=True).strip();started=True
        save(ROOT/'shim.json',dict(container=cid,settings=settings))
        for _ in range(180):
            try:
                r=requests.get(f'http://localhost:{C.port}/v1/models',headers={'Authorization':'Bearer '+env['TINKER_API_KEY']},timeout=5);r.raise_for_status()
                assert r.json()['data'][0]['checkpoint']=='tinker://base';break
            except requests.RequestException:time.sleep(2)
        else:raise TimeoutError('Recovery shim readiness failed')
        save(ROOT/'status.json',dict(phase='inference',at=time.time()))
        code=docker('consume',env)
        with (ROOT/'shim.log').open('w',encoding='utf-8') as log:subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['docker','stop',name],check=True);started=False
        save(ROOT/'status.json',dict(phase='grading',inference_exit=code,at=time.time()))
        assert docker('grade',env)==0
        save(ROOT/'status.json',dict(phase='publishing',at=time.time()));publish()
        ids=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+NAME],text=True).split()
        for cid in ids:
            assert not json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']['Running']
            subprocess.run(['docker','rm',cid],check=True)
        save(ROOT/'cleanup.json',dict(at=time.time(),owned_remaining=[]))
        save(ROOT/'status.json',dict(phase='complete',at=time.time()))
    except BaseException as exc:
        save(ROOT/'status.json',dict(phase='held',error=repr(exc),at=time.time()));raise
    finally:
        if started:subprocess.run(['docker','stop',name],check=True)
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','consume','grade','publish']);action=parser.parse_args().action
    if action in ('consume','grade'):linux(action)
    else:globals()[action]()
