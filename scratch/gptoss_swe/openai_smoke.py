# ABOUTME: Matched base/control local ODCV and SWE smokes with OpenAI reference Harmony.
# ABOUTME: Fresh exclusive claims and fixed per-stage caps; no outcome retries or cloud rentals.
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

ROOT = Path('output/gptoss_openai_smoke')
OLD = Path('output/gptoss_shared_smoke')
TARGETS = {'base': 'tinker://base', 'control': 'tinker://c8be8040-3551-5baf-a8aa-b9e606a91ed2:train:0/sampler_weights/2026-10-06-gptoss120b-0-plain'}
DATE = '2026-10-09'
IMAGE = 'lasr-gptoss-shared-smoke:20261008'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def environment():
    from dotenv import dotenv_values
    keys = dotenv_values(Path.home()/'source/repos/LASR/teaching_claude_why_replication/.env')
    return {**os.environ, **{k:v for k,v in keys.items() if v}, 'PYTHONUTF8':'1', 'PYTHONIOENCODING':'utf-8'}


def port(arm, kind):
    return 18155 + (0 if arm == 'base' else 10) + {'odcv':0,'swe':1,'compat':2}[kind]


def prepare():
    from omegaconf import OmegaConf
    selection = json.loads((OLD/'selection.json').read_text())
    root = ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root/'prepared.json').open('x') as f:
        json.dump(dict(at=time.time(),targets=TARGETS,total_cap_usd=36,selection=selection), f, indent=2)
    images = json.loads((OLD/'cache/metadata/images.json').read_text())
    verified = []
    for iid in selection['ids']:
        expected = images[iid]
        actual = json.loads(subprocess.check_output(['docker','image','inspect',expected['name']]))[0]
        assert expected['digest'] in actual['RepoDigests'], iid
        subprocess.run(['docker','run','--rm','--network','none','--entrypoint','/bin/bash',expected['name'],
                        '-lc','cd /testbed && git rev-parse HEAD && test -d /opt/miniconda3'],check=True)
        verified.append(dict(instance_id=iid,**expected))
    save(root/'images-verified.json', verified)
    for arm in TARGETS:
        cfg = OmegaConf.load(OLD/'swe.yaml')
        cfg.campaign = 'gptoss-openai-smoke-20261009-'+arm
        cfg.run_name = 'gptoss120b-'+arm+'-openai-smoke'
        cfg.output_root = '/work/'+(root/arm/'swe').as_posix()
        cfg.tinker.budget_usd = 12
        cfg.tinker.budget_ledger = '/work/'+(root/arm/'swe-budget.json').as_posix()
        cfg.tinker.render_date = DATE
        cfg.sampling.temperature, cfg.sampling.top_p, cfg.sampling.top_k = 1.0, 1.0, -1
        (root/arm).mkdir(exist_ok=True)
        OmegaConf.save(cfg,root/arm/'swe.yaml')


def start_shim(arm, kind, env):
    import requests
    root = ROOT/arm
    context, allowance, cap = (131072,16384,12) if kind=='swe' else (28000,8192,1 if kind=='compat' else 3)
    name = 'lasr-gptoss-openai-'+arm+'-'+kind+'-shim'
    args=['docker','run','-d','--name',name,'--label','lasr.campaign=gptoss-openai-smoke-20261009',
          '-p',f'127.0.0.1:{port(arm,kind)}:1234','--memory','3g','--cpus','2',
          '-v',f'{Path.cwd()}:/work','-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
          '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface']
    settings=dict(TINKER_CKPT=TARGETS[arm], TINKER_CONTEXT_WINDOW=str(context),DEFAULT_MAX_TOKENS=str(allowance),
        TINKER_BUDGET_USD=str(cap),TINKER_BUDGET_LEDGER=f'/work/{root.as_posix()}/{kind}-budget.json',
        TINKER_TRACE_DIR=f'/work/{root.as_posix()}/{kind}-traces',TINKER_BIND_HOST='0.0.0.0',
        TINKER_RENDER_DATE=DATE,REASONING_LEVEL='medium')
    for key,value in settings.items(): args+=['-e',key+'='+value]
    for key in ('TINKER_API_KEY','HF_TOKEN'): args+=['-e',key]
    args += [IMAGE,'src/infra/endpoints/tinker_env/.venv/bin/python','-m','src.infra.endpoints.tinker_server']
    cid=subprocess.check_output(args,env=env,text=True).strip()
    save(root/(kind+'-shim.json'),dict(container=cid,name=name,settings=settings))
    try:
        for _ in range(180):
            try:
                r=requests.get(f'http://localhost:{port(arm,kind)}/v1/models',
                    headers={'Authorization':'Bearer '+env['TINKER_API_KEY']},timeout=5)
                r.raise_for_status()
                identity=r.json()
                assert identity['data'][0]['checkpoint']==TARGETS[arm]
                save(root/(kind+'-identity.json'),identity)
                return name
            except requests.RequestException:
                running=subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',name],text=True).strip()
                if running!='true': raise RuntimeError('Shim exited; inspect Docker logs')
                time.sleep(2)
        raise TimeoutError('Shim readiness exceeded six minutes')
    except BaseException:
        stop_shim(arm,kind,name)
        raise


def stop_shim(arm,kind,name):
    with (ROOT/arm/(kind+'-shim.log')).open('w',encoding='utf-8') as log:
        subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT)
    subprocess.run(['docker','stop','--time','30',name],check=True)


def linux_swe(arm):
    from omegaconf import OmegaConf
    from scratch.gptoss_swe.run import run
    cfg=OmegaConf.load(ROOT/arm/'swe.yaml')
    target=SimpleNamespace(base_url=f'http://host.docker.internal:{port(arm,"swe")}/v1',
        model_name='openai/gpt-oss-120b',spec=SimpleNamespace(hf_path=TARGETS[arm]))
    print(json.dumps(run(target,cfg,(ROOT/arm/'swe').resolve())),flush=True)


def host(arm,kind):
    from omegaconf import OmegaConf
    root,env=ROOT/arm,environment()
    root.mkdir(parents=True,exist_ok=True)
    if kind!='compat':
        assert (ROOT/'interface-audit.json').exists(), 'Interface audit required before benchmark sampling'
        audit=json.loads((ROOT/'interface-audit.json').read_text())
        assert audit['passed'] and audit['source']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    with (root/(kind+'-started.json')).open('x') as f:
        json.dump(dict(at=time.time(),source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f)
    shim=None
    try:
        shim=start_shim(arm,kind,env)
        if kind=='compat':
            directory=Path('output/openai-gpt-oss-reference/compatibility-test')
            env.update(SMOKE_URL=f'http://localhost:{port(arm,kind)}/v1',SMOKE_ARM=arm)
            subprocess.run(['node', '--import','tsx','index.ts','--provider',arm,'-k','1'],cwd=directory,env=env,check=True)
            files=list(directory.glob('rollout_'+arm+'_*.jsonl'))
            assert len(files)==1
            records=[json.loads(s) for s in files[0].read_text(encoding='utf-8-sig').splitlines()]
            save(root/'compat-results.json', records)
            # Do not equate malformed model JSON or hallucinated tool names with
            # transport incompatibility. The audit below diagnoses every failure.
        elif kind=='swe':
            assert (ROOT/'images-verified.json').exists()
            args=['docker','run','--name','lasr-gptoss-openai-'+arm+'-swe-driver',
                  '--label','lasr.campaign=gptoss-openai-smoke-20261009','--add-host','host.docker.internal:host-gateway',
                  '-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock']
            for volume,path in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),
                                ('harness','src/eval/capabilities/swebench_mini/envs/harness')]:
                args+=['-v',f'lasr-gptoss-{volume}-env:/work/{path}/.venv']
            args+=['-e','TINKER_API_KEY',IMAGE,'scratch/swebench_cpu_env/.venv/bin/python',
                   '-m','scratch.gptoss_swe.openai_smoke','linux-swe','--arm',arm]
            subprocess.run(args,env=env,check=True)
        else:
            from src.eval.misalignment.odcv import odcv_rollout
            cfg=OmegaConf.load(OLD/'odcv.yaml')
            cfg.model_key='gptoss120b-'+arm+'-openai-smoke'
            cfg.base_url=f'http://host.docker.internal:{port(arm,kind)}/v1'
            cfg.output_root=str((root/'odcv').resolve())
            cfg.endpoint_api_key_env='TINKER_API_KEY'
            cfg.temperature=1.0
            OmegaConf.save(cfg,root/'odcv.yaml')
            os.environ.update(env)
            result=odcv_rollout.main(str(root/'odcv.yaml'))
            save(root/'odcv-rollout.json',dict(path=str(result)))
        save(root/(kind+'-finished.json'),dict(at=time.time(),phase='complete'))
    except BaseException as error:
        save(root/(kind+'-failure.json'),dict(at=time.time(),error=repr(error)))
        raise
    finally:
        if shim: stop_shim(arm,kind,shim)


def judge_odcv(arm):
    from omegaconf import OmegaConf
    from src.eval.misalignment.odcv.odcv_judge import main
    os.environ.update(environment())
    root=ROOT/arm
    with (root/'odcv-judge-started.json').open('x') as f: json.dump(dict(at=time.time(),cap_usd=2),f)
    cfg=OmegaConf.load(root/'odcv.yaml')
    cfg.published_key='gpt-oss-120b'
    cfg.judge_budget=dict(ledger=str((root/'judge-budget.json').resolve()),cap_usd=2,max_tokens=8192)
    OmegaConf.save(cfg,root/'odcv-judge.yaml')
    run=json.loads((root/'odcv-rollout.json').read_text())['path']
    main(run,str(root/'odcv-judge.yaml'),max_workers=2)
    save(root/'odcv-judge-finished.json',dict(at=time.time()))


def audit():
    """Read-only inference audit: retain all cases, including upstream error records."""
    import hashlib
    directory=Path('output/openai-gpt-oss-reference/compatibility-test')
    arms={}
    for arm in TARGETS:
        root=ROOT/arm
        files=list(directory.glob('rollout_'+arm+'_*.jsonl'))
        assert len(files)==1
        records=[json.loads(s) for s in files[0].read_text(encoding='utf-8-sig').splitlines()]
        assert len(records)==30 and len({r['test_case'] for r in records})==30
        save(root/'compat-results.json',records)
        errors=[r['error'] for r in records if 'error' in r]
        # These are explicit model behavior exceptions in the official test runner.
        # Unknown API/transport failures hold admission for diagnosis.
        assert all(e.startswith('Tool ') and ' not found in agent ' in e or
                   e.startswith('Failed to run function tools: SyntaxError:') for e in errors), errors
        assert all(r['result']['validResponse'] for r in records if 'result' in r)
        traces=list((root/'compat-traces').glob('*/request.json'))
        assert traces
        for request_path in traces:
            request=json.loads(request_path.read_text(encoding='utf-8'))
            response=json.loads(request_path.with_name('response.json').read_text(encoding='utf-8'))
            options=request['sampling']
            assert (options['temperature'],options['top_p'],options['top_k'])==(1,1,-1)
            assert set(options['stop'])=={200002,200012}
            assert response['tinker_metadata']['checkpoint']==TARGETS[arm]
            assert response['tinker_metadata']['tokenizer_revision']=='openai-harmony==0.0.8:HARMONY_GPT_OSS'
        passed=sum(bool(r.get('success')) for r in records)
        arms[arm]=dict(cases=30,passed=passed,failed=30-passed,model_exceptions=errors,
            returned_api_shapes_valid=True,requests=len(traces),
            official_recommended_over_90_percent=passed/30>.9)
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
        'src/infra/endpoints/tinker_harmony.py','src/infra/endpoints/tinker_server.py',
        'src/infra/endpoints/tinker_env/uv.lock']}
    save(ROOT/'interface-audit.json',dict(passed=True,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources_sha256=sources,offline_tests=36,arms=arms,
        scope='Non-streaming text Chat Completions with auto function tools. Not full API certification.',
        admission='OpenAI reference rendering/parsing, correct wire sampling and valid response shapes; retain model-output failures per user instruction. Quality thresholds are reported, not silently passed.',
        sources=['https://github.com/openai/harmony','https://github.com/openai/gpt-oss#recommended-sampling-parameters',
                 'https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot',
                 'https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit','compat','odcv','swe','linux-swe','judge-odcv'])
    parser.add_argument('--arm',choices=list(TARGETS),default='base')
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='audit': audit()
    elif args.action=='linux-swe': linux_swe(args.arm)
    elif args.action=='judge-odcv': judge_odcv(args.arm)
    else: host(args.arm,args.action)
