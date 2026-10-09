# ABOUTME: Run fifty authorized fresh SWE attempts on one qualified CPU, preserving ten base outcomes.
# ABOUTME: Three independent checkpoint owners share a 32-command gate and retain raw evidence and costs.
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
from collections import Counter
from omegaconf import OmegaConf

C=OmegaConf.load(os.environ.get('GPTOSS_TWENTY_CONFIG','scratch/gptoss_swe/twenty.yaml'))
ROOT=Path(C.root)
INPUT=Path(C.input)


def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expanded_state(prior,selection,arm):
    assert len(selection['old_ids'])==len(selection['new_ids'])==10
    assert len(selection['ids'])==len(set(selection['ids']))==20
    assert selection['ids']==selection['old_ids']+selection['new_ids']
    tasks={i:dict(status='pending',attempts=[]) for i in selection['ids']}
    if arm=='base':
        assert set(prior['tasks'])==set(selection['old_ids'])
        assert not prior.get('halt') and all(t['status']=='valid' for t in prior['tasks'].values())
        tasks.update(json.loads(json.dumps(prior['tasks'])))
    return dict(tasks=tasks,pods=[],deadline=None,phase='inference',halt=None)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'prepare-claim.json').open('x') as f:json.dump(dict(at=time.time(),pid=os.getpid()),f)
    selection=read(INPUT/'selection.json');save(ROOT/'selection.json',selection)
    baseline=OmegaConf.load(INPUT/'base-config.yaml')
    assert baseline.tinker.context_window==131072 and baseline.worker.max_response_tokens==16384
    assert baseline.worker.max_task_tokens==262144 and baseline.worker.step_limit==500
    assert baseline.sampling.temperature==1 and baseline.sampling.top_p==1 and baseline.sampling.top_k==-1
    assert sha(INPUT/'cache/metadata/swebench_lite_test.json')==baseline.dataset_sha256
    prior=read(INPUT/'base-prior/metadata/state.json')
    assert read(INPUT/'base-prior/metadata/manifest.json')['target']==C.targets.base
    hashes={}
    for arm,target in C.targets.items():
        r=ROOT/arm;r.mkdir(exist_ok=True)
        cfg=OmegaConf.create(OmegaConf.to_container(baseline,resolve=True))
        cfg.campaign=str(C.campaign)+'-'+arm;cfg.run_name='gptoss120b-'+arm+'-smoke20'
        cfg.output_root=str(r/'swe');cfg.cached_campaign=str(INPUT/'cache');cfg.instance_ids=selection['ids'];cfg.subset.n=20
        cfg.workers=int(C.workers[arm]);cfg.grading.max_workers=int(C.grading_workers)
        cfg.worker.tool_concurrency=int(C.tool_concurrency);cfg.worker.min_available_memory_gib=int(C.min_available_memory_gib)
        cfg.tinker.budget_usd=None;cfg.tinker.budget_ledger=str(r/'budget.json');cfg.tinker.render_date=str(C.render_date)
        OmegaConf.save(cfg,r/'config.yaml')
        meta=r/'swe/metadata';meta.mkdir(parents=True)
        for filename in ('images.json','swebench_lite_test.json'):shutil.copyfile(INPUT/'cache/metadata'/filename,meta/filename)
        if arm=='base':
            for folder in ('rollouts','results'):
                for p in sorted((INPUT/'base-prior'/folder).rglob('*')):
                    if not p.is_file():continue
                    rel=p.relative_to(INPUT/'base-prior');dest=r/'swe'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(p,dest);digest=sha(p);assert sha(dest)==digest;hashes[rel.as_posix()]=digest
            shutil.copytree(INPUT/'base-prior/metadata',r/'prior-metadata')
            ledger=read(INPUT/'base-budget.json');assert ledger['ceiling_usd'] is None and all(v['state']=='completed' for v in ledger['requests'].values())
            assert ledger['authorized_checkpoints']==[target]
            save(r/'budget.json',ledger);save(r/'prior-summary.json',read(INPUT/'base-summary.json'))
        save(meta/'manifest.json',dict(campaign=cfg.campaign,target=target,protocol=cfg.protocol,selected_ids=selection['ids'],config=OmegaConf.to_container(cfg,resolve=True),retained_ids=selection['old_ids'] if arm=='base' else [],source=read('/srv/lasr/deployment.json')))
        save(meta/'state.json',expanded_state(prior,selection,arm))
        save(r/'status.json',dict(phase='prepared',at=time.time()))
    save(ROOT/'base-copy-hashes.json',hashes)
    shutil.copyfile(INPUT/'prior-publication.json',ROOT/'prior-publication.json')
    save(ROOT/'authorization.json',dict(at=time.time(),instruction='User explicitly authorized fresh control20 and DA15 20; base retains10 and runs10 new. All50 new tasks concurrently. SWE spending guards removed previously and remain removed.',retained_base_inference_usd=sum(v['upper_usd'] for v in read(INPUT/'base-budget.json')['requests'].values()),old_control_excluded=True))
    save(ROOT/'status.json',dict(phase='prepared',at=time.time()))


def qualify():
    import psutil
    from scratch.swebench_cpu_load import WORKLOADS
    info=json.loads(subprocess.check_output(['docker','info','--format','{{json .}}'],text=True))
    assert info['NCPU']>=C.min_cpus and info['MemTotal']/2**30>=C.min_ram_gib
    assert shutil.disk_usage(info['DockerRootDir']).free/2**30>C.min_free_gib
    rows=read(INPUT/'cache/metadata/swebench_lite_test.json');images=read(INPUT/'cache/metadata/images.json')
    ids=list(read(ROOT/'selection.json')['ids'])
    load_ids=[next(r['instance_id'] for r in rows if r['repo']==repo) for repo in WORKLOADS]
    wanted=list(dict.fromkeys(ids+load_ids))
    def pull(iid):
        image=images[iid];log=ROOT/'qualification/pulls'/f'{iid}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:
            subprocess.run(['docker','pull',image['digest']],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
        actual=json.loads(subprocess.check_output(['docker','image','inspect',image['digest']],text=True))[0]
        assert image['digest'] in actual['RepoDigests']
        subprocess.run(['docker','tag',image['digest'],image['name']],check=True)
        subprocess.run(['docker','run','--rm','--network','none','--entrypoint','bash',image['digest'],'-lc','cd /testbed && git rev-parse HEAD && test -d /opt/miniconda3'],capture_output=True,check=True,timeout=90)
        print('Image verified: '+iid,flush=True)
        return dict(instance_id=iid,**image)
    with ThreadPoolExecutor(max_workers=int(C.image_pull_workers)) as pool:verified=list(pool.map(pull,wanted))
    save(ROOT/'qualification/images.json',verified)
    # The exact existing Qwen qualification, including agent imports and 80 environments / 32 tools.
    readiness=Path('/srv/lasr/runs/cpu-readiness/metadata');readiness.mkdir(parents=True,exist_ok=True)
    for n in ('swebench_lite_test.json','images.json'):shutil.copyfile(INPUT/'cache/metadata'/n,readiness/n)
    out=ROOT/'qualification/capacity'
    with (ROOT/'qualification/capacity.log').open('x') as log:
        subprocess.run([sys.executable,'-m','scratch.swebench_cpu_load','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(out/'results.json');assert result['status']=='finished'
    for phase in result['phases']:
        assert not any(t.get('error') or any(c['returncode'] for c in t['commands']) or t.get('resources',{}).get('memory.events',{}).get('oom_kill',0) for t in phase['tasks'])
    assert any(p['agents']==80 and p['tool_limit']==32 for p in result['phases'])
    from scratch.gptoss_swe.run import grade_predictions
    qroot=ROOT/'qualification/harness';(qroot/'metadata').mkdir(parents=True)
    shutil.copyfile(INPUT/'cache/metadata/swebench_lite_test.json',qroot/'metadata/swebench_lite_test.json')
    cfg=OmegaConf.load(ROOT/'base/config.yaml')
    by_id={r['instance_id']:r for r in rows}
    gold={i:dict(model_name_or_path='infrastructure-gold',model_patch=by_id[i]['patch']) for i in load_ids}
    good=grade_predictions(cfg,qroot,gold,str(C.campaign)+'-gold')
    assert good['n_resolved']==good['n_graded']==len(load_ids),good
    negative='diff --git a/.lasr-negative-control b/.lasr-negative-control\nnew file mode 100644\n--- /dev/null\n+++ b/.lasr-negative-control\n@@ -0,0 +1 @@\n+Infrastructure check only.\n'
    bad=grade_predictions(cfg,qroot,{i:dict(model_name_or_path='infrastructure-no-fix',model_patch=negative) for i in load_ids},str(C.campaign)+'-no-fix')
    assert bad['n_graded']==len(load_ids) and bad['n_resolved']==0,bad
    save(ROOT/'qualification/gold-no-fix.json',dict(gold=good,no_fix=bad))
    # Prove durable artifact write/read access before making paid model requests.
    from huggingface_hub import HfApi,hf_hub_download
    proof=ROOT/'qualification/proof';proof.mkdir()
    for name,path in [('capacity.json',out/'results.json'),('gold-no-fix.json',ROOT/'qualification/gold-no-fix.json'),('images.json',ROOT/'qualification/images.json'),('deployment.json',Path('/srv/lasr/deployment.json'))]:shutil.copyfile(path,proof/name)
    (proof/'README.md').write_text('---\nlicense: mit\ntags: [infrastructure-check, swebench-lite]\n---\n# CPU-only twenty-task preparation\n\nImage hashes, real repository load tests, and gold/no-fix checks. No model inference.\n')
    hashes={p.name:sha(p) for p in proof.iterdir() if p.is_file()}
    api=HfApi(token=os.environ['HF_TOKEN']);repo=str(C.hf_repo)+'-infrastructure';api.create_repo(repo,repo_type='dataset',exist_ok=True)
    revision=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=proof,commit_message='Verify CPU-only preparation before inference').oid
    for name,digest in hashes.items():assert sha(hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=os.environ['HF_TOKEN']))==digest
    save(ROOT/'qualification/publication.json',dict(repo=repo,revision=revision,sha256=hashes,verified=True))
    save(ROOT/'qualification/ready.json',dict(passed=True,at=time.time(),host_cpus=info['NCPU'],host_memory_bytes=info['MemTotal'],selected_images=20,total_images=len(verified),capacity_sha256=sha(out/'results.json')))


def arm_inference(arm):
    from src.eval.capabilities.swebench_mini.fleet_worker import consume
    from scratch.gptoss_swe.finish_smoke import consume_pool
    import requests
    r=ROOT/arm;cfg=OmegaConf.load(r/'config.yaml');port=int(C.ports[arm]);target=C.targets[arm]
    settings=dict(TINKER_CKPT=target,TINKER_CONTEXT_WINDOW=str(cfg.tinker.context_window),DEFAULT_MAX_TOKENS=str(cfg.worker.max_response_tokens),TINKER_BUDGET_USD='unlimited',TINKER_BUDGET_LEDGER=str(r/'budget.json'),TINKER_TRACE_DIR=str(r/'traces'),TINKER_BIND_HOST='127.0.0.1',TINKER_PORT=str(port),TINKER_RENDER_DATE=str(C.render_date),REASONING_LEVEL='medium')
    # Native shim accepts PORT (not Docker port mapping); verify its bound endpoint below.
    settings['PORT']=str(port)
    with (r/'shim.log').open('x') as log:
        proc=subprocess.Popen(['src/infra/endpoints/tinker_env/.venv/bin/python','-u','-m','src.infra.endpoints.tinker_server'],env=os.environ|settings,stdout=log,stderr=subprocess.STDOUT)
        save(r/'shim-owner.json',dict(pid=proc.pid,at=time.time(),settings=settings))
        try:
            endpoint=f'http://127.0.0.1:{port}/v1'
            for _ in range(240):
                assert proc.poll() is None,'Shim exited; see log'
                try:
                    res=requests.get(endpoint+'/models',headers={'Authorization':'Bearer '+os.environ['TINKER_API_KEY']},timeout=5);res.raise_for_status();identity=res.json()
                    assert identity['data'][0]['checkpoint']==target;save(r/'identity.json',identity);break
                except requests.RequestException:time.sleep(2)
            else:raise TimeoutError('Shim startup')
            for _ in range(5):
                response=requests.post(endpoint[:-3]+'/tokenize',headers={'Authorization':'Bearer '+os.environ['TINKER_API_KEY']},json=dict(model='openai/gpt-oss-120b',messages=[dict(role='user',content='transport probe')],tools=[]),timeout=60);response.raise_for_status();assert response.json()['count']>0
            save(r/'transport.json',dict(passed=True,tokenization_only_probes=5))
            worker=OmegaConf.merge(cfg.worker,dict(root=str(r/'swe'),serving=dict(context_window=cfg.tinker.context_window),endpoint_api_key_env='TINKER_API_KEY',fleet_owner_root=str(ROOT),protocol_version=cfg.protocol,sampling=cfg.sampling))
            state=read(r/'swe/metadata/state.json');ids=[i for i in cfg.instance_ids if state['tasks'][i]['status']!='valid']
            allowances={i:dict(max_total_attempt_records=int(worker.max_infrastructure_attempts)) for i in ids}
            admission=dict(directory=str(r/'token-slots'),budget_tokens=int(cfg.tinker.context_window)*int(cfg.workers),expires=time.time()+7*86400,fairness_seconds=30)
            save(r/'status.json',dict(phase='inference',at=time.time(),workers=int(cfg.workers)))
            consume_pool(consume,endpoint,worker,arm,ids,allowances,admission,int(cfg.workers))
            save(r/'status.json',dict(phase='inference_finished',at=time.time()))
        except BaseException as exc:
            save(r/'status.json',dict(phase='held',at=time.time(),error=repr(exc)));raise
        finally:
            proc.terminate()
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            save(r/'shim-exit.json',dict(code=proc.returncode,at=time.time()))


def grade():
    from scratch.gptoss_swe.run import grade_predictions
    for arm,target in C.targets.items():
        r=ROOT/arm;cfg=OmegaConf.load(r/'config.yaml');state=read(r/'swe/metadata/state.json')
        assert not any(t['status']=='running' for t in state['tasks'].values())
        preds={i:t['attempts'][-1]['prediction'] for i,t in state['tasks'].items() if t['status']=='valid'}
        save(r/'status.json',dict(phase='grading',at=time.time()))
        result=grade_predictions(cfg,r/'swe',preds,str(C.campaign)+'-'+arm) if preds else dict(n_graded=0,n_resolved=0,resolved_ids=[])
        save(r/'summary.json',dict(**result,target=target,n_selected=20,n_excluded=20-len(preds)))
        save(r/'status.json',dict(phase='graded',at=time.time()))


def limit_reason(traj):
    for message in reversed(traj.get('messages',[])):
        reason=message.get('extra',{}).get('limit_reason')
        if reason:return reason
    info=traj.get('info',{})
    return info.get('limit_reason') or ('step_limit' if info.get('model_stats',{}).get('api_calls',0)>=500 else 'other_limit')


def status():
    results={}
    for arm in C.targets:
        r=ROOT/arm;state=read(r/'swe/metadata/state.json');counts=Counter(t['status'] for t in state['tasks'].values());ends=Counter();limits=Counter()
        for iid,t in state['tasks'].items():
            if t['status']!='valid':continue
            a=t['attempts'][-1];ends[a['exit_status']]+=1
            if a['exit_status']=='LimitsExceeded':
                directory=r/'swe/rollouts'/iid/a['id'];files=list(directory.glob('**/'+iid+'.traj.json'));traj=read(files[0] if files else directory/'checkpoint.traj.json')
                limits[limit_reason(traj)]+=1
        summary=read(r/'summary.json') if (r/'summary.json').exists() else read(r/'prior-summary.json') if (r/'prior-summary.json').exists() else dict(n_graded=0,n_resolved=0)
        d=read(r/'budget.json') if (r/'budget.json').exists() else dict(requests={})
        results[arm]=dict(success=summary['n_resolved'],graded=summary['n_graded'],submitted=ends['Submitted'],failed=summary['n_graded']-summary['n_resolved'],limits=dict(limits),awaiting_grading=counts['valid']-summary['n_graded'],running=counts['running'],waiting=counts['pending'],infrastructure_excluded=counts['invalid'],cost_usd=sum(v['upper_usd'] for v in d['requests'].values()),reservations=sum(v['state']!='completed' for v in d['requests'].values()),phase=read(r/'status.json')['phase'])
    return results


def publish():
    from huggingface_hub import HfApi,hf_hub_download
    summary=status();save(ROOT/'summary.json',summary)
    for a in C.targets:assert summary[a]['running']==0 and summary[a]['reservations']==0
    out=ROOT/'package';out.mkdir(exist_ok=True)
    for d in ('rollouts','results','metadata'):(out/d).mkdir(exist_ok=True)
    needles=[v.encode() for k,v in os.environ.items() if len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    for a in C.targets:
        with tarfile.open(out/'rollouts'/f'{a}.tar.gz','w:gz') as tar:
            for p in sorted((ROOT/a).rglob('*')):
                if p.is_file():
                    assert not any(n in p.read_bytes() for n in needles),str(p)
                    tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        save(out/'results'/f'{a}.json',read(ROOT/a/'summary.json'))
    for p in ROOT.iterdir():
        if p.is_file() and p.name not in ('publication.json',):shutil.copyfile(p,out/'metadata'/p.name)
    shutil.copytree(ROOT/'qualification',out/'metadata/qualification',dirs_exist_ok=True)
    lines=['---','license: mit','tags: [eval-run, "eval:swebench_mini", "model:gptoss120b", "mode:think", swebench-lite, diagnostic-subset]','---','# GPT-OSS-120B matched twenty-task SWE smoke','',
      'experiment: 20 matched task IDs per arm; 10 retained base outcomes and 50 newly authorized attempts.',
      'date_generated: 2026-10-09','constitution: none','source_repo: Matthew-Bozoukov/Lessons_from_constituitional_AFT@'+read('/srv/lasr/deployment.json')['git_commit'],
      'models: exact Tinker checkpoint paths in each manifest; native Harmony 0.0.8, medium reasoning.',
      'generation_config: T1/top_p1/top_k disabled; 16384 response/131072 context/262144 generated task tokens/500 steps; 7200-second requests, two request attempts, three infrastructure attempts.',
      'schema: rollouts archives contain raw requests/responses/attempts; results contain official grades; metadata contains configuration, qualification and source.',
      'provenance: python -m scratch.gptoss_swe.twenty prepare; qualify; run. Original base raw evidence is pinned by prior-publication.json.',
      'Fresh control replaces the earlier control replicate only in this declared comparison; original results remain untouched. This is a diagnostic subset, not a full SWE-bench Lite score.',
      'No JSON coaching, output repair, model-failure retries or hidden outcome selection. Memory/command controls match qualified Qwen CPU policy.',
      'Base cost includes the historical cumulative base ledger; control and DA15 costs are fresh. Never add old base cost twice. Costs are conservative accounting, not invoices.','',
      '| Metric | Base | Control | DA15 |','|---|---:|---:|---:|']
    for metric in ('success','graded','submitted','failed','awaiting_grading','running','waiting','infrastructure_excluded','cost_usd'):lines.append('| '+metric+' | '+' | '.join(str(summary[a][metric]) for a in C.targets)+' |')
    for reason in ('context_limit','response_token_limit','step_limit','task_token_limit','other_limit'):lines.append('| '+reason+' | '+' | '.join(str(summary[a]['limits'].get(reason,0)) for a in C.targets)+' |')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='file-hashes.json'};save(out/'metadata/file-hashes.json',hashes);hashes['metadata/file-hashes.json']=sha(out/'metadata/file-hashes.json')
    api=HfApi(token=os.environ['HF_TOKEN']);api.create_repo(str(C.hf_repo),repo_type='dataset',exist_ok=True)
    rev=api.upload_folder(repo_id=str(C.hf_repo),repo_type='dataset',folder_path=out,commit_message='Publish matched twenty-task three-arm SWE smoke').oid
    save(ROOT/'publication.json',dict(repo=str(C.hf_repo),revision=rev,complete=False,sha256=hashes))
    for path,digest in hashes.items():assert sha(hf_hub_download(str(C.hf_repo),path,repo_type='dataset',revision=rev,token=os.environ['HF_TOKEN']))==digest
    save(ROOT/'publication.json',dict(repo=str(C.hf_repo),revision=rev,complete=True,sha256=hashes))


def run():
    from dotenv import load_dotenv
    load_dotenv('/srv/lasr/credentials.env')
    assert read(ROOT/'qualification/ready.json')['passed']
    with (ROOT/'owner-started.json').open('x') as f:json.dump(dict(pid=os.getpid(),at=time.time(),source=read('/srv/lasr/deployment.json')),f)
    try:
        save(ROOT/'status.json',dict(phase='inference',at=time.time()))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={a:pool.submit(arm_inference,a) for a in C.targets};errors={}
            for a,f in futures.items():
                try:f.result()
                except Exception as exc:errors[a]=repr(exc)
        save(ROOT/'inference-exits.json',errors)
        assert not errors,errors
        save(ROOT/'status.json',dict(phase='grading',at=time.time()));grade()
        save(ROOT/'status.json',dict(phase='publishing',at=time.time()));publish()
        ids=[]
        for a in C.targets:ids+=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr_campaign='+str(C.campaign)+'-'+a],text=True).split()
        assert not ids,'Task cleanup incomplete'
        save(ROOT/'cleanup.json',dict(at=time.time(),owned_remaining=[]))
        save(ROOT/'status.json',dict(phase='complete',at=time.time()))
    except BaseException as exc:
        save(ROOT/'status.json',dict(phase='held',at=time.time(),error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','qualify','run','status','publish']);a=parser.parse_args()
    if a.action=='status':print(json.dumps(status(),indent=2))
    else:globals()[a.action]()
