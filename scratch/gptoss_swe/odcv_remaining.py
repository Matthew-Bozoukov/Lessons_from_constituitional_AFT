# ABOUTME: Fresh matched ODCV cells on local Docker/Tinker with no independent response-token cap.
# ABOUTME: Owns one attempt per cell, separate fixed budgets, judging, raw publication and scoped cleanup.
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from omegaconf import OmegaConf

CFG=OmegaConf.load('scratch/gptoss_swe/odcv_remaining.yaml')
os.environ.update(GPTOSS_SMOKE_ROOT=str(CFG.root),GPTOSS_SMOKE_PREFIX=str(CFG.prefix),
    GPTOSS_SMOKE_CAMPAIGN=str(CFG.campaign),GPTOSS_SMOKE_PORT_BASE=str(CFG.port_base),
    GPTOSS_ODCV_CONFIG=str(Path(CFG.root)/'odcv-selection.yaml'))
from scratch.gptoss_swe import openai_smoke as smoke
ROOT=smoke.ROOT
save=smoke.save


def source():
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def prepare():
    from src.eval.misalignment.odcv.odcv_rollout import scenario_names
    assert not ROOT.exists(), 'Never clear an existing campaign root'
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert not subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(CFG.campaign)],text=True).strip()
    cfg=OmegaConf.load('output/gptoss_refresh_smoke_20261009/base/odcv.yaml')
    excluded={tuple(x) for x in CFG.excluded_cells}
    assert excluded=={(v,n) for v,names in cfg.include_scenarios.items() for n in names}
    pool=sorted((v,n) for v in ('mandated','incentivized') for n in scenario_names(Path(cfg.bench_dir),v) if (v,n) not in excluded)
    selected=sorted(random.Random(int(CFG.seed)).sample(pool,10))
    assert selected==[tuple(x) for x in CFG.selected_cells] and len(pool)==70
    cfg.include_scenarios={v:sorted(n for variant,n in selected if variant==v) for v in ('mandated','incentivized')}
    cfg.temperature=.7
    ROOT.mkdir(parents=True)
    for arm in smoke.TARGETS:(ROOT/arm).mkdir()
    OmegaConf.save(cfg,ROOT/'odcv-selection.yaml')
    save(ROOT/'selection.json',dict(seed=int(CFG.seed),available_cells=pool,excluded=sorted(excluded),selected=selected,selection_rule='Random(seed).sample(sorted(remaining cells),10); identical cells for both arms'))
    args=['docker','run','--rm','-v',f'{Path.cwd()}:/work',
          '-v','lasr-gptoss-shim-env:/work/src/infra/endpoints/tinker_env/.venv',
          '-v','lasr-gptoss-agent-env:/work/src/eval/capabilities/swebench_mini/envs/agent/.venv',
          '-v','lasr-gptoss-hf-cache:/root/.cache/huggingface',smoke.IMAGE,
          'src/infra/endpoints/tinker_env/.venv/bin/python','-m','pytest',
          'tests/test_tinker_server.py','tests/test_tinker_budget.py','-q']
    with (ROOT/'offline-tests.log').open('w',encoding='utf-8') as log:
        subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    text=(ROOT/'offline-tests.log').read_text()
    assert '40 passed' in text and 'skipped' not in text
    files=['src/infra/endpoints/tinker_harmony.py','src/infra/endpoints/tinker_server.py',
           'src/infra/endpoints/tinker_env/uv.lock','tests/test_tinker_server.py','tests/test_tinker_budget.py']
    save(ROOT/'interface-audit.json',dict(passed=True,source=source(),offline_tests=40,
         sources_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
         note='Tests cover exact remaining-context allowance, explicit caps, refusal before sampling, tool history and budgets.'))
    save(ROOT/'manifest.json',dict(source=source(),targets=smoke.TARGETS,config=OmegaConf.to_container(CFG),
         sampling=dict(temperature=.7,top_p=.95,top_k=20,reasoning='medium'),context_window=28000,
         response_allowance='context_window minus exact rendered Harmony prompt tokens',cycles=50,
         per_arm_caps=dict(inference=3,judging=2),total_cap_usd=10,
         comparison_caveat='New cases and changed response cap: not a paired causal estimate of cap effect.'))
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(ROOT/'source.tar.gz'),'HEAD'],check=True)
    save(ROOT/'status.json',dict(phase='prepared',at=time.time(),source=source()))


def invoke(action,arm):
    with (ROOT/arm/(action+'-owner.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.openai_smoke',action,'--arm',arm],
                              env=smoke.environment(),stdout=log,stderr=subprocess.STDOUT)
    save(ROOT/arm/(action+'-owner-exit.json'),dict(code=result.returncode,at=time.time()))
    return result.returncode


def publish():
    from huggingface_hub import HfApi,hf_hub_download
    from scratch.gptoss_swe.audit_shared_smoke import main as audit
    summaries={}
    for arm,target in smoke.TARGETS.items():
        root=ROOT/arm
        assert (root/'odcv-judge-finished.json').exists()
        audit(root,root/'unused-swe',smoke.OLD/'cache',False)
        requests=list((root/'odcv-traces').glob('*/request.json'))
        assert requests
        for path in requests:
            d=json.loads(path.read_text());p=d['sampling']
            assert (p['temperature'],p['top_p'],p['top_k'])==(.7,.95,20)
            assert p['max_tokens']==28000-len(d['prompt_tokens']),path
            response=path.with_name('response.json')
            if response.exists():assert json.loads(response.read_text())['tinker_metadata']['checkpoint']==target
        run=Path(json.loads((root/'odcv-rollout.json').read_text())['path'])
        results=json.loads((run/'results.json').read_text())
        assert results['n_judged']==10 and results['n_dropped_all_na']==0
        ledger=json.loads((root/'odcv-budget.json').read_text())
        assert ledger['authorized_checkpoints']==[target] and ledger['ceiling_usd']==3
        costs=dict(inference=sum(x['upper_usd'] for x in ledger['requests'].values()),
                   judging=sum(x['charged_or_reserved_usd'] for x in json.loads((root/'judge-budget.json').read_text())))
        assert costs['inference']<=3 and costs['judging']<=2
        diag=json.loads((root/'audit.json').read_text())['odcv']
        assert len(diag['cells'])==10
        summaries[arm]=dict(target=target,results=results,diagnostics=diag,costs_usd=costs)
    env=smoke.environment();needles=[v.encode() for k,v in env.items() if len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    out=ROOT/'package'
    for folder in ('metadata','results','rollouts'):(out/folder).mkdir(parents=True,exist_ok=True)
    for arm in smoke.TARGETS:
        with tarfile.open(out/'rollouts'/f'{arm}.tar.gz','w:gz') as tar:
            for file in sorted((ROOT/arm).rglob('*')):
                if not file.is_file():continue
                raw=file.read_bytes();assert not any(n in raw for n in needles),f'Secret in {file}'
                tar.add(file,arcname=file.relative_to(ROOT),recursive=False)
        save(out/'results'/f'{arm}.json',summaries[arm])
    for file in ROOT.iterdir():
        if file.is_file() and file.name!='publication.json':shutil.copyfile(file,out/'metadata'/file.name)
    cost=sum(sum(s['costs_usd'].values()) for s in summaries.values());assert cost<=10
    lines=['---','license: mit','---','# GPT-OSS base/control: ten new ODCV cells, remaining-context responses','',
           'Local Docker plus Tinker. Same randomly selected ten previously unused scenario/variant cells per arm.',
           'T0.7, top_p0.95, top_k20, medium reasoning, OpenAI Harmony0.0.8; no custom JSON coaching or malformed-output resampling.',
           'Context 28000 tokens, 50 cycles, no independent output cap: each response may use the exact remaining context.',
           'Cases differ from the earlier 8k-cap smoke, so this does not isolate the causal effect of the cap.',
           '', '| Metric | Base | Control |','|---|---:|---:|']
    def row(name,fn):lines.append('| '+name+' | '+' | '.join(str(fn(summaries[a])) for a in smoke.TARGETS)+' |')
    row('Task complete / 10',lambda r:sum(c['tool_completed'] for c in r['diagnostics']['cells']))
    row('Context/cycle/response-limit endings',lambda r:'/'.join(str(sum(c[k] for c in r['diagnostics']['cells'])) for k in ('context_limit','step_limit','response_limit')))
    row('Misalignment %',lambda r:r['results']['ours']['overall']['mr_pct'])
    row('Accounted USD',lambda r:round(sum(r['costs_usd'].values()),6))
    lines+=['',f'Total conservative accounting ${cost:.6f}, not provider invoice. Fixed $10 ceiling; unknown requests remain reserved.',
            'Task completion and misalignment are distinct. Diagnostic subset only, not a full ODCV benchmark.',
            'A length-ended generation in this run exhausted available context, not an independent 8192-token response allowance.',
            'All raw prompts, sampling options, sampled tokens, tool feedback, transcripts, source and budget records are retained.']
    text='\n'.join(lines)+'\n';(out/'README.md').write_text(text,encoding='utf-8')
    hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='file-hashes.json'}
    save(out/'metadata/file-hashes.json',hashes);hashes['metadata/file-hashes.json']=hashlib.sha256((out/'metadata/file-hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=env['HF_TOKEN']);api.create_repo(str(CFG.hf_repo),repo_type='dataset',exist_ok=True)
    revision=api.upload_folder(repo_id=str(CFG.hf_repo),repo_type='dataset',folder_path=out,commit_message='Publish ten fresh matched ODCV cells without an independent response cap').oid
    receipt=dict(repo=str(CFG.hf_repo),revision=revision,complete=False,verified_files=0,sha256=hashes);save(ROOT/'publication.json',receipt)
    for name,digest in hashes.items():
        file=hf_hub_download(str(CFG.hf_repo),name,repo_type='dataset',revision=revision,token=env['HF_TOKEN'])
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,name
        receipt['verified_files']+=1
    receipt['complete']=True;save(ROOT/'publication.json',receipt)
    save(ROOT/'final-report.json',dict(artifact_url=f'https://huggingface.co/datasets/{CFG.hf_repo}/tree/{revision}',cost_usd=cost,text=text))


def run():
    with (ROOT/'owner-started.json').open('x') as f:json.dump(dict(pid=os.getpid(),source=source(),at=time.time()),f)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for action in ('odcv','judge-odcv'):
            save(ROOT/'status.json',dict(phase=action,at=time.time()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                exits=dict(zip(smoke.TARGETS,pool.map(lambda arm:invoke(action,arm),smoke.TARGETS)))
            save(ROOT/(action+'-exits.json'),exits)
            assert not any(exits.values()),f'{action} failed; no inference automatically retried'
        save(ROOT/'status.json',dict(phase='publishing',at=time.time()));publish()
        ids=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(CFG.campaign)],text=True).split()
        for cid in ids:
            assert not json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']['Running']
            subprocess.run(['docker','rm',cid],check=True)
        assert not subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign='+str(CFG.campaign)],text=True).strip()
        save(ROOT/'cleanup.json',dict(at=time.time(),owned_remaining=[]))
        save(ROOT/'status.json',dict(phase='complete',at=time.time(),publication=json.loads((ROOT/'publication.json').read_text())))
    except BaseException as error:
        save(ROOT/'status.json',dict(phase='held',at=time.time(),error=repr(error)));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','publish'])
    globals()[parser.parse_args().action]()
