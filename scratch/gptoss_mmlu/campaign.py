# ABOUTME: Qualified local three-arm MMLU owner using the maintained eval framework.
# ABOUTME: Run: CPU environment python -m scratch.gptoss_mmlu.campaign prepare|load|smoke|run|arm PHASE ARM
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid

ROOT=Path('/work/output/gptoss_mmlu_20261014')
CPU='/work/scratch/swebench_cpu_env/.venv/bin/python'
SHIM='/work/src/infra/endpoints/tinker_env/.venv/bin/python'
TARGETS={'base':'tinker://base',
 'control':'tinker://c8be8040-3551-5baf-a8aa-b9e606a91ed2:train:0/sampler_weights/2026-10-06-gptoss120b-0-plain',
 'da15':'tinker://5c74fa19-a0b6-58a8-bfdf-610961c28e12:train:0/sampler_weights/2026-10-06-gptoss120b-0-da-15'}
PORTS=dict(base=18805,control=18815,da15=18825)

def save(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)

def config():
    from omegaconf import OmegaConf
    return OmegaConf.merge(OmegaConf.load('configs/eval/mmlu.yaml'),
                           OmegaConf.load('scratch/gptoss_mmlu/protocol.yaml'))

def prepare():
    from omegaconf import OmegaConf
    from src.eval.capabilities.mmlu import runner as r
    from src.eval.capabilities.mmlu.mmlu import load_split,build_subset,subset_hash,build_prompt
    from src.eval.run_eval import _credentials_preflight
    _credentials_preflight(True)
    cfg=config()
    q=build_subset(load_split('test',str(cfg.subset.dataset),'all',revision=cfg.subset.revision),
        per_subject=10,seed=0,shuffle_choices=True)
    assert len(q)==570 and subset_hash(q)=='3952064292260029'
    shots=r._shots_by_subject(cfg,0)
    validation=load_split('validation',str(cfg.subset.dataset),'all',revision=cfg.subset.revision)
    for row in validation: row['uid']='validation:'+row['uid']
    smoke=build_subset(validation,per_subject=1,seed=0,subjects=['abstract_algebra'],shuffle_choices=True)
    assert len(smoke)==1
    save(ROOT/'targets.json',TARGETS); save(ROOT/'questions.json',q); save(ROOT/'shots.json',shots)
    save(ROOT/'smoke-question.json',smoke)
    save(ROOT/'prompts.json',[dict(uid=x['uid'],prompt=build_prompt(x,shots[x['subject']],
        instruction=str(cfg.prompt.instruction),cue=str(cfg.prompt.cue))) for x in q+smoke])
    OmegaConf.save(cfg,ROOT/'config.yaml')
    save(ROOT/'prepared.json',dict(at=time.time(),questions=len(q),subset_hash=subset_hash(q),smoke_questions=1))

@contextlib.contextmanager
def bridge(phase,arm):
    import requests
    env={**os.environ,'MMLU_PHASE':phase,'MMLU_ARM':arm,'PORT':str(PORTS[arm]),'MMLU_INSTANCE':uuid.uuid4().hex}
    directory=ROOT/phase/arm; directory.mkdir(parents=True,exist_ok=True)
    log=(directory/'bridge.log').open('w')
    proc=subprocess.Popen([SHIM,'-m','scratch.gptoss_mmlu.bridge'],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if proc.poll() is not None: raise RuntimeError(f'{arm} bridge exited {proc.returncode}')
            try:
                resp=requests.get(f'http://127.0.0.1:{PORTS[arm]}/v1/models',
                    headers={'Authorization':'Bearer '+os.environ['TINKER_API_KEY']},timeout=2)
                if resp.status_code==200:
                    item=resp.json()['data'][0]
                    assert item['checkpoint']==TARGETS[arm] and item['instance_id']==env['MMLU_INSTANCE']
                    assert item['reasoning_effort']=='medium' and item['context_window']==131072
                    save(directory/'identity.json',resp.json()); break
            except requests.RequestException: pass
            time.sleep(1)
        else: raise TimeoutError('Bridge readiness')
        yield
    finally:
        proc.terminate()
        try: proc.wait(20)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        log.close()

def arm(phase,name):
    from omegaconf import OmegaConf
    from src.eval.capabilities.mmlu import runner as r
    if phase!='load':
        source=json.loads((ROOT/'source.json').read_text())
        for path,digest in source['files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        # Windows worktree .git points outside the Linux mount; use the verified
        # clean checkout manifest, not the container's otherwise-missing Git metadata.
        import src.utils
        src.utils.git_sha=lambda:source['sha']
    cfg=config(); directory=ROOT/phase/name
    cfg.output_root=str(directory/'eval')
    if phase in ('load','smoke'):
        q=json.loads((ROOT/('smoke-question.json' if phase=='smoke' else 'questions.json')).read_text())
        if phase=='load': q=q[:64]
        cfg.generation.parallel=1 if phase=='smoke' else 64
        desc=dict(name=name,served='openai/gpt-oss-120b',adapter=None,target=TARGETS[name],
                  revision=TARGETS[name],synthetic_fraction=None,role='target')
        with bridge(phase,name):
            score=r.run_arm(desc,q,json.loads((ROOT/'shots.json').read_text()),cfg,
                f'http://127.0.0.1:{PORTS[name]}/v1',directory,api_key=os.environ['TINKER_API_KEY'])
        save(directory/'summary.json',score)
    else:
        import src.eval.run_eval as framework
        from dataclasses import replace
        from src.infra.endpoints.tinker import sampler_name
        from src.naming import undated
        resolve=framework.resolve_target
        framework.resolve_target=lambda target,**kwargs:replace(resolve(target,**kwargs),
            model_key=undated(sampler_name(target)))
        os.environ['TINKER_SHIM_PORT']=str(PORTS[name])
        framework._tinker_endpoint=lambda spec,cfg: bridge(phase,name)
        cfgpath=directory/'config.yaml'; directory.mkdir(parents=True,exist_ok=True); OmegaConf.save(cfg,cfgpath)
        def run_with_evidence(target,cfg,out):
            try:
                result=r.run(target,cfg,out)
            except ValueError as error:
                if 'MMLU instrument-health check failed' not in str(error): raise
                # A full set of real model format failures is a diagnostic result,
                # never a reason to repair/resample or label the arm healthy.
                import shutil
                arm_dir=out/'think'/target.spec.model_key
                records=[json.loads(l) for l in (arm_dir/'records.jsonl').read_text().splitlines()]
                assert len(records)==570 and all(x['finish_reason']!='timeout' for x in records)
                measured=max(p for p in arm_dir.iterdir() if p.is_dir())
                result=json.loads((measured/'metrics.json').read_text())
                assert result['valid'] is False
                from src.eval.layout import publish_layout
                rollouts,results,metadata=publish_layout(out)
                shutil.copy2(arm_dir/'records.jsonl',rollouts/'records.jsonl')
                shutil.copy2(arm_dir/'attempts.jsonl',rollouts/'attempts.jsonl')
                shutil.copy2(measured/'metrics.json',results/'metrics.json')
                shutil.copy2(measured/'raw_samples.md',results/'raw_samples.md')
                meta=json.loads((measured/'run_meta.json').read_text())
                meta['records_file']='rollouts/records.jsonl'
                save(metadata/'mmlu_run_meta.json',meta)
                save(metadata/'health-failure.json',dict(message=str(error),valid=False,no_rerolls=True))
            with tarfile.open(out/'metadata/native-traces.tar.gz','w:gz') as archive:
                archive.add(directory/'traces',arcname='traces')
            for p in ('config.yaml','prepared.json','token-preflight.json','load-qualification.json','smoke-complete.json','targets.json','source.json','source.tar.gz'):
                (out/'metadata'/p).write_bytes((ROOT/p).read_bytes())
            (out/'metadata/bridge-identity.json').write_bytes((directory/'identity.json').read_bytes())
            (out/'metadata/budget-at-arm-completion.json').write_bytes((ROOT/'budget.json').read_bytes())
            save(directory/'framework-output.json',dict(out=str(out)))
            return result
        framework.main(['--target',TARGETS[name],'--name','mmlu','--config',str(cfgpath)],runner=run_with_evidence)
    save(directory/'finished.json',dict(at=time.time()))

def phase_run(phase):
    from scratch.gptoss_mmlu.bridge import working_memory
    with (ROOT/f'{phase}-started.json').open('x') as f: json.dump(dict(at=time.time(),pid=os.getpid()),f)
    children={}; logs=[]; samples=[]; released=False
    try:
        for name in TARGETS:
            path=ROOT/phase/name; path.mkdir(parents=True,exist_ok=True)
            log=(path/'owner.log').open('w'); logs.append(log)
            children[name]=subprocess.Popen([CPU,'-m','scratch.gptoss_mmlu.campaign','arm',phase,name],stdout=log,stderr=subprocess.STDOUT)
        while any(p.poll() is None for p in children.values()):
            mem=int(Path('/sys/fs/cgroup/memory.current').read_text())
            samples.append(dict(at=time.time(),memory=mem,working_memory=working_memory(),cpu=Path('/sys/fs/cgroup/cpu.stat').read_text()))
            if phase=='load' and not released:
                peaks=[json.loads((ROOT/f'load-{a}-active.json').read_text())['peak'] if (ROOT/f'load-{a}-active.json').exists() else 0 for a in TARGETS]
                if peaks==[64,64,64]:
                    (ROOT/'load-release').write_text('192 mock requests simultaneously admitted')
                    released=True
            save(ROOT/f'{phase}-progress.json',dict(at=time.time(),codes={a:p.poll() for a,p in children.items()},memory=mem))
            time.sleep(1)
        codes={a:p.returncode for a,p in children.items()}
        save(ROOT/f'{phase}-resources.json',samples)
        assert all(c==0 for c in codes.values()),codes
        budget=json.loads((ROOT/('load-budget.json' if phase=='load' else 'budget.json')).read_text())
        rows=list(budget['requests'].values()); stage=[r for r in rows if r['phase']==phase]
        assert all(r['state']=='completed' for r in stage)
        expected=192 if phase=='load' else 3 if phase=='smoke' else 1710
        assert len(stage)==expected,(len(stage),expected)
        if phase=='load':
            assert released and max(s['working_memory'] for s in samples)<3.5*2**30
            events=dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
            assert int(events['oom_kill'])==0
            for a in TARGETS:
                records=[json.loads(l) for l in (ROOT/phase/a/'think'/a/'records.jsonl').read_text().splitlines()]
                assert len(records)==64 and all(r['usage']['completion_tokens']==32768 and r['answer']=='A' and r['think'] for r in records)
                assert len(list((ROOT/phase/a/'traces').glob('*/response.json')))==64
        if phase=='smoke':
            assert sum(r['upper_usd'] for r in stage)<.49
            for a in TARGETS:
                records=[json.loads(l) for l in (ROOT/phase/a/'think'/a/'records.jsonl').read_text().splitlines()]
                assert len(records)==1 and records[0]['finish_reason']=='stop' and records[0]['think'] and records[0]['parsed']
        save(ROOT/('load-qualification.json' if phase=='load' else f'{phase}-complete.json'),
             dict(at=time.time(),requests=len(stage),peak_memory=max(s['memory'] for s in samples),
                  peak_working_memory=max(s['working_memory'] for s in samples),
                  accounted_usd=sum(r['upper_usd'] for r in rows),phase_usd=sum(r['upper_usd'] for r in stage),codes=codes))
    finally:
        for log in logs: log.close()

def qualify_smoke():
    budget=json.loads((ROOT/'budget.json').read_text())
    rows=list(budget['requests'].values())
    assert len(rows)==3 and all(x['state']=='completed' and x['phase']=='smoke' for x in rows)
    failures=[]
    for a in TARGETS:
        directory=ROOT/'smoke'/a
        records=[json.loads(l) for l in (directory/'think'/a/'records.jsonl').read_text().splitlines()]
        assert len(records)==1 and records[0]['finish_reason']=='stop'
        responses=list((directory/'traces').glob('*/response.json'))
        assert len(responses)==1
        response=json.loads(responses[0].read_text())
        assert response['tinker_metadata']['checkpoint']==TARGETS[a]
        assert response['tinker_metadata']['boundary']=='final'
        assert records[0]['think'] or records[0]['answer']
        if not records[0]['parsed']: failures.append(a)
    cost=sum(x['upper_usd'] for x in rows)
    assert cost<.49
    save(ROOT/'smoke-complete.json',dict(at=time.time(),requests=3,accounted_usd=cost,
        transport_qualified=True,model_format_failures=failures,
        interpretation='Analysis-only model output remains unparseable; no repair, retry or sampling change. Full screen will report health.'))

if __name__=='__main__':
    command=sys.argv[1]
    if command=='prepare': prepare()
    elif command=='qualify-smoke': qualify_smoke()
    elif command=='arm': arm(sys.argv[2],sys.argv[3])
    else: phase_run(command)
