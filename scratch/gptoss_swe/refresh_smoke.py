# ABOUTME: Own a fresh bounded local base/control smoke without modifying or resuming historical runs.
# ABOUTME: Serial phases, parallel arms, retained grading, raw-trace audit and immutable HF publication.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from omegaconf import OmegaConf

CFG = OmegaConf.load('scratch/gptoss_swe/refresh_smoke.yaml')
os.environ.update(GPTOSS_SMOKE_ROOT=str(CFG.root), GPTOSS_SMOKE_PREFIX=str(CFG.prefix),
                  GPTOSS_SMOKE_CAMPAIGN=str(CFG.campaign), GPTOSS_SMOKE_PORT_BASE=str(CFG.port_base))
from scratch.gptoss_swe import openai_smoke as smoke
ROOT = smoke.ROOT
save = smoke.save


def source():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def docker_args(name):
    args = ['docker', 'run', '--name', name, '--label', 'lasr.campaign='+str(CFG.campaign),
            '-v', f'{Path.cwd()}:/work', '-v', '/var/run/docker.sock:/var/run/docker.sock']
    for volume, path in [('cpu', 'scratch/swebench_cpu_env'),
                         ('agent', 'src/eval/capabilities/swebench_mini/envs/agent'),
                         ('harness', 'src/eval/capabilities/swebench_mini/envs/harness')]:
        args += ['-v', f'lasr-gptoss-{volume}-env:/work/{path}/.venv']
    return args + [smoke.IMAGE, 'scratch/swebench_cpu_env/.venv/bin/python',
                   '-m', 'scratch.gptoss_swe.refresh_smoke']


def prepare():
    assert not ROOT.exists(), 'New campaign root must not exist; never clear existing evidence'
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert not subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(CFG.campaign)],text=True).strip()
    smoke.prepare()
    prepared = json.loads((ROOT/'prepared.json').read_text())
    prepared.update(total_cap_usd=int(CFG.total_cap_usd), source=source(),
                    prior_run='output/gptoss_openai_smoke',
                    note='Explicitly requested fresh smoke. No old ledger or outcome reused.')
    save(ROOT/'prepared.json', prepared)
    for arm in smoke.TARGETS:
        cfg = OmegaConf.load(ROOT/arm/'swe.yaml')
        cfg.workers = int(CFG.workers_per_swe_arm)
        cfg.grade = False  # Grade after both inference owners stop, to bound local memory.
        OmegaConf.save(cfg, ROOT/arm/'swe.yaml')
    args = ['docker','run','--rm','-v',f'{Path.cwd()}:/work',
            '-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
            '-v','lasr-gptoss-agent-env:/work/src/eval/capabilities/swebench_mini/envs/agent/.venv',
            '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface',smoke.IMAGE,
            'src/infra/endpoints/tinker_env/.venv/bin/python','-m','pytest',
            'tests/test_tinker_server.py','tests/test_tinker_budget.py','-q']
    with (ROOT/'offline-tests.log').open('w',encoding='utf-8') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    text = (ROOT/'offline-tests.log').read_text()
    assert '38 passed' in text and 'skipped' not in text
    files = ['src/infra/endpoints/tinker_harmony.py','src/infra/endpoints/tinker_server.py',
             'src/infra/endpoints/tinker_env/uv.lock','tests/test_tinker_server.py','tests/test_tinker_budget.py']
    save(ROOT/'interface-audit.json',dict(passed=True,source=source(),offline_tests=38,
        sources_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        prior_live_compatibility='output/gptoss_openai_smoke/interface-audit.json',
        note='Fresh offline sampling/history/tool/budget checks; prior live compatibility not rerun or claimed fresh.'))
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(ROOT/'source.tar.gz'),'HEAD'],check=True)
    save(ROOT/'status.json',dict(phase='prepared',at=time.time(),source=source()))


def invoke(action, arm):
    with (ROOT/arm/(action+'-owner.log')).open('x',encoding='utf-8') as log:
        result = subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.openai_smoke',
                                 action,'--arm',arm],env=smoke.environment(),stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/arm/(action+'-owner-exit.json'),dict(code=result.returncode,at=time.time()))
    return result.returncode


def grade():
    from scratch.gptoss_swe.run import grade_predictions
    for arm, target in smoke.TARGETS.items():
        root = ROOT/arm/'swe'
        state = json.loads((root/'metadata/state.json').read_text())
        assert not any(t['status']=='running' for t in state['tasks'].values())
        manifest = json.loads((root/'metadata/manifest.json').read_text())
        assert manifest['target'] == target
        preds = {i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status']=='valid'}
        output = root/'results/retained-grading.json'
        if output.exists():
            result = json.loads(output.read_text())
        else:
            cfg = OmegaConf.create(manifest['config'])
            result = grade_predictions(cfg,root.resolve(),preds,cfg.campaign+'-retained') if preds else dict(n_graded=0,n_resolved=0,resolved_ids=[])
            save(output,result)
        assert result['n_graded']==len(preds)
        save(ROOT/arm/'swe-summary.json',dict(**result,checkpoint=target,n_selected=len(state['tasks']),
             n_valid_rollouts=len(preds),n_interrupted=sum(t['status']=='invalid' for t in state['tasks'].values()),
             n_not_started=sum(t['status']=='pending' for t in state['tasks'].values()),task_statuses=state['tasks']))


def publish():
    from huggingface_hub import HfApi, hf_hub_download
    from scratch.gptoss_swe.audit_shared_smoke import main as audit
    env = smoke.environment()
    summaries = {}
    for arm,target in smoke.TARGETS.items():
        root = ROOT/arm
        audit(root,root/'swe',smoke.OLD/'cache',False)
        for kind, sampling in [('odcv',CFG.odcv_sampling),('swe',CFG.swe_sampling)]:
            requests = list((root/(kind+'-traces')).glob('*/request.json'))
            assert requests, f'No requests: {arm}/{kind}'
            for file in requests:
                req = json.loads(file.read_text())
                assert all(req['sampling'][key]==value for key,value in sampling.items()), file
                res = file.with_name('response.json')
                if res.exists(): assert json.loads(res.read_text())['tinker_metadata']['checkpoint']==target
        costs = {}
        for kind in ('odcv','swe'):
            ledger = json.loads((root/(kind+'-budget.json')).read_text())
            assert ledger['authorized_checkpoints']==[target]
            costs[kind]=sum(r['upper_usd'] for r in ledger['requests'].values())
            assert costs[kind]<=ledger['ceiling_usd']
        costs['judge']=sum(r['charged_or_reserved_usd'] for r in json.loads((root/'judge-budget.json').read_text()))
        odcvroot=Path(json.loads((root/'odcv-rollout.json').read_text())['path'])
        odcv=json.loads((odcvroot/'results.json').read_text())
        assert odcv['n_judged']==10 and odcv['n_dropped_all_na']==0
        swe=json.loads((root/'swe-summary.json').read_text())
        assert swe['n_graded']+swe['n_interrupted']+swe['n_not_started']==10
        summaries[arm]=dict(target=target,swe=swe,odcv=odcv,costs_usd=costs,
                           diagnostics=json.loads((root/'audit.json').read_text()))
    cost=sum(sum(r['costs_usd'].values()) for r in summaries.values())
    assert cost<=int(CFG.total_cap_usd)
    out=ROOT/'package'; out.mkdir(exist_ok=True)
    for directory in ('rollouts','metadata','results'): (out/directory).mkdir(exist_ok=True)
    needles=[v.encode() for k,v in env.items() if len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    for arm in smoke.TARGETS:
        with tarfile.open(out/'rollouts'/f'{arm}.tar.gz','w:gz') as tar:
            for file in sorted((ROOT/arm).rglob('*')):
                if not file.is_file():continue
                raw=file.read_bytes()
                assert not any(n in raw for n in needles),f'Secret in {file}'
                tar.add(file,arcname=file.relative_to(ROOT),recursive=False)
        save(out/'results'/f'{arm}.json',summaries[arm])
    import shutil
    for file in ROOT.iterdir():
        if file.is_file() and file.name not in ('publication.json',): shutil.copyfile(file,out/'metadata'/file.name)
    url='https://huggingface.co/datasets/'+str(CFG.hf_repo)
    lines=['---','license: mit','---','# GPT-OSS base/control local smoke: ODCV temperature 0.7','',
           'Ten ODCV cells and ten SWE tasks selected per arm. Diagnostic subset, not full benchmark scores.',
           'ODCV: temperature .7, top_p .95, top_k 20. SWE: temperature 1, top_p 1, top_k disabled.',
           'Both: OpenAI Harmony 0.0.8, medium reasoning, no custom JSON coaching or outcome retries.',
           'ODCV limits: 8192 response, 28000 context, 50 cycles. SWE: 16384 response, 131072 context, 262144 generated/task, 500 steps.',
           '', '| Metric | Base | Control |','|---|---:|---:|']
    def row(name,fn):lines.append('| '+name+' | '+' | '.join(str(fn(summaries[a])) for a in smoke.TARGETS)+' |')
    row('ODCV task_complete / 10',lambda r:sum(c['tool_completed'] for c in r['diagnostics']['odcv']['cells']))
    row('ODCV context/cycle/response limits',lambda r:'/'.join(str(sum(c[k] for c in r['diagnostics']['odcv']['cells'])) for k in ('context_limit','step_limit','response_limit')))
    row('ODCV misalignment %',lambda r:r['odcv']['ours']['overall']['mr_pct'])
    row('SWE resolved / graded',lambda r:f"{r['swe']['n_resolved']}/{r['swe']['n_graded']}")
    row('SWE Submitted / LimitsExceeded',lambda r:'/'.join(str(r['diagnostics']['swe']['ended_statuses'].get(k,0)) for k in ('Submitted','LimitsExceeded')))
    row('SWE interrupted / unstarted',lambda r:f"{r['swe']['n_interrupted']}/{r['swe']['n_not_started']}")
    lines+=['',f'Conservative accounted spend ${cost:.6f}, not invoice; ceiling $34. Unknown requests retain their full reservation.',
            'All paid attempts are retained. Limit endings are not automatically unresolved; official grading determines resolution. Interruptions/unstarted tasks are separate.',
            'Prior smoke is separate; these are explicitly authorized fresh attempts. No Vast/RunPod rentals.']
    text='\n'.join(lines)+'\n';(out/'README.md').write_text(text,encoding='utf-8')
    hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='file-hashes.json'}
    save(out/'metadata/file-hashes.json',hashes)
    hashes['metadata/file-hashes.json']=hashlib.sha256((out/'metadata/file-hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=env['HF_TOKEN']); api.create_repo(str(CFG.hf_repo),repo_type='dataset',exist_ok=True)
    revision=api.upload_folder(repo_id=str(CFG.hf_repo),repo_type='dataset',folder_path=out,commit_message='Publish fresh ODCV07 and SWE smoke, all outcomes retained').oid
    receipt=dict(repo=str(CFG.hf_repo),revision=revision,complete=False,verified_files=0,sha256=hashes)
    save(ROOT/'publication.json',receipt)
    for name,expected in hashes.items():
        path=hf_hub_download(str(CFG.hf_repo),name,repo_type='dataset',revision=revision,token=env['HF_TOKEN'])
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,name
        receipt['verified_files']+=1
    receipt['complete']=True;save(ROOT/'publication.json',receipt)
    save(ROOT/'final-report.json',dict(artifact_url=url+'/tree/'+revision,cost_usd=cost,text=text))


def run():
    with (ROOT/'owner-started.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid(),source=source()),f)
    # Prevent sleep only while this owner is alive; Windows resets it when the process exits.
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for kind in ('odcv','swe'):
            save(ROOT/'status.json',dict(phase=kind,at=time.time(),source=source()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                exits=dict(zip(smoke.TARGETS,pool.map(lambda arm:invoke(kind,arm),smoke.TARGETS)))
            save(ROOT/(kind+'-exits.json'),exits)
        save(ROOT/'status.json',dict(phase='grading',at=time.time()))
        for arm in smoke.TARGETS:
            assert invoke('judge-odcv',arm)==0, f'ODCV judge failed: {arm}'
        with (ROOT/'retained-grading.log').open('x',encoding='utf-8') as log:
            subprocess.run(docker_args(str(CFG.prefix)+'-grader')+['grade'],stdout=log,stderr=subprocess.STDOUT,check=True)
        save(ROOT/'status.json',dict(phase='publishing',at=time.time()))
        publish()
        names=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(CFG.campaign)],text=True).split()
        for cid in names:
            item=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            assert not item['State']['Running']
            subprocess.run(['docker','rm',cid],check=True)
        assert not subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(CFG.campaign)],text=True).strip()
        save(ROOT/'cleanup.json',dict(at=time.time(),owned_remaining=[]))
        save(ROOT/'status.json',dict(phase='complete',at=time.time(),publication=json.loads((ROOT/'publication.json').read_text())))
    except BaseException as error:
        save(ROOT/'status.json',dict(phase='held',at=time.time(),error=repr(error)))
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','grade','publish'])
    globals()[parser.parse_args().action]()
