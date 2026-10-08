# ABOUTME: Base-only local shared-interface smoke, with separate immutable bounded inference ledgers.
# ABOUTME: Each eval launches once; no resampling, cloud rentals, or changes to historical runs.
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

ROOT = Path('output/gptoss_shared_smoke')

def linux_swe():
    from omegaconf import OmegaConf
    from scratch.gptoss_swe.run import run
    cfg=OmegaConf.load(ROOT/'swe.yaml')
    cfg.tinker.budget_usd=12
    cfg.tinker.budget_ledger='/work/output/gptoss_shared_smoke/swe-budget.json'
    target=SimpleNamespace(base_url='http://host.docker.internal:18146/v1',model_name='openai/gpt-oss-120b',
                           spec=SimpleNamespace(hf_path='tinker://base'))
    print(json.dumps(run(target,cfg,ROOT.resolve()/'swe')),flush=True)

def start_shim(kind, env):
    import requests
    port,context,allowance,cap=(18145,28000,8192,3) if kind=='odcv' else (18146,131072,16384,12)
    name='lasr-gptoss-shared-'+kind+'-shim'
    args=['docker','run','-d','--name',name,'--label','lasr.campaign=gptoss-shared-smoke-20261008',
          '-p',f'127.0.0.1:{port}:1234','--memory','3g','--cpus','2',
          '-v',f'{Path.cwd()}:/work','-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
          '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface']
    settings=dict(TINKER_CKPT='tinker://base',TINKER_CONTEXT_WINDOW=str(context),DEFAULT_MAX_TOKENS=str(allowance),
                  TINKER_BUDGET_USD=str(cap),TINKER_BUDGET_LEDGER=f'/work/{ROOT.as_posix()}/{kind}-budget.json',
                  TINKER_TRACE_DIR=f'/work/{ROOT.as_posix()}/{kind}-traces',TINKER_BIND_HOST='0.0.0.0',
                  TINKER_RENDER_DATE='2026-10-08',REASONING_LEVEL='medium')
    for key,value in settings.items(): args+=['-e',key+'='+value]
    for key in ('TINKER_API_KEY','HF_TOKEN'): args+=['-e',key]
    args+=['lasr-gptoss-shared-smoke:20261008','src/infra/endpoints/tinker_env/.venv/bin/python',
           '-m','src.infra.endpoints.tinker_server']
    cid=subprocess.check_output(args,env=env,text=True).strip()
    (ROOT/(kind+'-shim.json')).write_text(json.dumps(dict(container=cid,name=name,settings=settings),indent=2))
    for _ in range(180):
        try:
            r=requests.get(f'http://localhost:{port}/v1/models',headers={'Authorization':'Bearer '+env['TINKER_API_KEY']},timeout=5)
            r.raise_for_status()
            (ROOT/(kind+'-identity.json')).write_text(json.dumps(r.json(),indent=2))
            return name
        except requests.RequestException:
            status=subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',name],text=True).strip()
            if status!='true': raise RuntimeError('Shim exited; inspect its Docker log')
            time.sleep(2)
    raise TimeoutError('Shim readiness exceeded six minutes')

def host(kind, resume=None):
    from dotenv import dotenv_values
    from omegaconf import OmegaConf
    rootenv=Path.home()/'source/repos/LASR/teaching_claude_why_replication/.env'
    env={**os.environ,**{k:v for k,v in dotenv_values(rootenv).items() if v}}
    ROOT.mkdir(parents=True,exist_ok=True)
    claim=ROOT/(kind+('-recovery-started.json' if resume else '-started.json'))
    with claim.open('x') as f: json.dump(dict(at=time.time(),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f)
    shim=None
    try:
        shim=start_shim(kind,env)
        if kind=='swe':
            assert (ROOT/'images-verified.json').exists()
            args=['docker','run','--name','lasr-gptoss-shared-swe-driver','--label','lasr.campaign=gptoss-shared-smoke-20261008',
                  '--add-host','host.docker.internal:host-gateway','-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock']
            for volume,path in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),
                                ('harness','src/eval/capabilities/swebench_mini/envs/harness')]:
                args+=['-v',f'lasr-gptoss-{volume}-env:/work/{path}/.venv']
            args+=['-e','TINKER_API_KEY','lasr-gptoss-shared-smoke:20261008',
                   'scratch/swebench_cpu_env/.venv/bin/python','-m','scratch.gptoss_swe.shared_smoke','linux-swe']
            subprocess.run(args,env=env,check=True)
        else:
            from src.eval.misalignment.odcv.odcv import VARIANTS,scenario_names
            from src.eval.misalignment.odcv import odcv_rollout
            cfg=OmegaConf.load('configs/eval/odcv/lite.yaml')
            cfg.model='openai/gpt-oss-120b'
            cfg.model_key='gptoss120b-base-shared-smoke'
            cfg.base_url='http://host.docker.internal:18145/v1'
            cfg.output_root=str((ROOT/'odcv').resolve())
            cfg.strict_tool_validation=True
            cfg.endpoint_api_key_env='TINKER_API_KEY'
            cfg.passes=1
            cfg.concurrency=1
            cfg.progress_judge=False
            cfg.include_scenarios={v:scenario_names(Path(cfg.bench_dir),v)[:5] for v in VARIANTS}
            OmegaConf.save(cfg,ROOT/'odcv.yaml')
            os.environ.update(env)
            if resume:
                original=Path(resume)
                assert original.is_dir()
                saved=list(original.glob('agent_logs/*/experiments/*/messages_record.txt'))
                assert len(saved)==3 and all(p.stat().st_size for p in saved)
                (ROOT/'odcv-preserved-before-recovery.json').write_text(json.dumps(dict(
                    paths=[str(p) for p in saved],reason='Terminal tool-return marker acceptance matched official cookbook; existing completed outcomes retained'),indent=2))
            result=odcv_rollout.main(str(ROOT/'odcv.yaml'),resume=resume)
            (ROOT/'odcv-rollout.json').write_text(json.dumps(dict(path=str(result))))
        (ROOT/(kind+'-finished.json')).write_text(json.dumps(dict(at=time.time(),phase='rollouts_complete')))
    except BaseException as error:
        (ROOT/(kind+'-failure.json')).write_text(json.dumps(dict(at=time.time(),error=repr(error))))
        raise
    finally:
        if shim:
            with (ROOT/(kind+'-shim.log')).open('w',encoding='utf-8') as log:
                subprocess.run(['docker','logs',shim],stdout=log,stderr=subprocess.STDOUT)
            subprocess.run(['docker','stop','--time','30',shim],check=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['odcv','swe','swe-wait','linux-swe'])
    parser.add_argument('--resume-odcv')
    args=parser.parse_args()
    if args.action=='swe-wait':
        with (ROOT/'swe-waiter.json').open('x') as stream:
            json.dump(dict(at=time.time()),stream)
        deadline=time.monotonic()+7200
        while not (ROOT/'images-verified.json').is_file():
            error=ROOT/'prepare.err'
            if error.exists() and error.stat().st_size:
                raise RuntimeError('Image preparation failed; inspect prepare.err')
            if time.monotonic()>deadline:
                raise TimeoutError('Image preparation exceeded two hours; no sampling started')
            time.sleep(5)
        args.action='swe'
    linux_swe() if args.action=='linux-swe' else host(args.action,args.resume_odcv)
