# ABOUTME: Audit every scored MMLU answer and publish immutable campaign closeout evidence.
# ABOUTME: Run after all inference exits; never sends model requests or reruns an evaluation.
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi,hf_hub_download
from src.infra.huggingface import hf_token,push_run_dir
from src.eval.capabilities.mmlu.mmlu import paired_diff
from src.eval.layout import assert_layout,run_tags
from src.eval.run_eval import _card_fields
from src.infra.endpoints.tinker import sampler_name
from src.naming import eval_name,undated
from omegaconf import OmegaConf
from scratch.gptoss_mmlu.campaign import ROOT,TARGETS,save

def main():
    import sys
    publish='--publish' in sys.argv
    budget=json.loads((ROOT/'budget.json').read_text())
    rows=list(budget['requests'].values())
    assert len(rows)==1713 and all(r['state']=='completed' for r in rows)
    assert sum(r['upper_usd'] for r in rows)<=15
    data={}; outputs={}; results={}
    for arm,target in TARGETS.items():
        out=Path(json.loads((ROOT/'run'/arm/'framework-output.json').read_text())['out'])
        outputs[arm]=out
        records=sorted([json.loads(l) for l in (out/'rollouts/records.jsonl').read_text().splitlines()],key=lambda r:r['uid'])
        assert len(records)==570 and len({r['uid'] for r in records})==570
        assert all(r['finish_reason']!='timeout' for r in records)
        metrics=json.loads((out/'results/metrics.json').read_text())
        own=[r for r in rows if r['checkpoint']==target]
        measured=[r for r in own if r['phase']=='run']
        assert len(own)==571 and len(measured)==570
        assert sum(r['completion_tokens'] for r in measured)==sum(r['usage']['completion_tokens'] for r in records)
        assert sum(r['prompt_tokens'] for r in measured)==sum(r['usage']['prompt_tokens'] for r in records)
        traces=list((ROOT/'run'/arm/'traces').glob('*/response.json'))
        assert len(traces)==570
        for path in traces:
            trace=json.loads(path.read_text())
            assert trace['tinker_metadata']['checkpoint']==target
            assert trace['tinker_metadata']['reasoning_effort']=='medium'
            assert trace['tinker_metadata']['sampling']['top_p']==1
            assert trace['tinker_metadata']['sampling']['top_k']==-1
        data[arm]=records
        results[arm]=dict(correct=sum(r['correct'] for r in records),n=570,
            accuracy=sum(r['correct'] for r in records)/570,
            parsed=sum(r['parsed'] is not None for r in records),
            empty_final=sum(not r['answer'].strip() for r in records),
            analysis_only=sum(not r['answer'].strip() and bool(r['think'].strip()) for r in records),
            truncated=sum(r['finish_reason']=='length' for r in records),
            valid=metrics['valid'],health_issues=metrics['health_issues'],
            scored_usd=sum(r['upper_usd'] for r in measured),
            smoke_usd=sum(r['upper_usd'] for r in own if r['phase']=='smoke'),
            total_usd=sum(r['upper_usd'] for r in own),
            input_tokens=sum(r['prompt_tokens'] for r in measured),
            output_tokens=sum(r['completion_tokens'] for r in measured))
    comparisons={}
    for a,b in [('control','da15'),('base','control'),('base','da15')]:
        value=paired_diff(data[a],data[b])
        value['non_inferior']=value['ci_lower']>=-.03
        value['capability_interpretation_valid']=results[a]['valid'] and results[b]['valid']
        comparisons[b+'-minus-'+a]=value
    report=dict(at=time.time(),arms=results,comparisons=comparisons,
        accounted_usd=sum(r['upper_usd'] for r in rows),
        smoke_usd=sum(r['upper_usd'] for r in rows if r['phase']=='smoke'),
        reservations=0,scored_requests=1710,smoke_requests=3,
        note='570-question matched screen. Final channel only; empty/unparseable outputs count wrong. Format failures are not repaired or resampled. Health-failed arms are diagnostic results, not a clean capability comparison. Costs are accounting estimates, not invoices.')
    save(ROOT/'final-results.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not publish: return
    api=HfApi(token=hf_token()); published={}
    for arm,out in outputs.items():
        original=out/'think'
        if original.exists():
            original.rename(out/'metadata'/'instrument-health-original-output')
        evidence=out/'metadata'/'campaign-closeout'; evidence.mkdir(exist_ok=True)
        for p in ('final-results.json','budget.json','smoke-complete.json','load-qualification.json',
                  'source.json','token-preflight.json','run-resources.json','host-resources.json',
                  'driver-packages.txt','shim-packages.txt'):
            shutil.copy2(ROOT/p,evidence/p)
        shutil.copy2(__file__,evidence/'closeout.py')
        with tarfile.open(evidence/'qualification-evidence.tar.gz','w:gz') as archive:
            for p in ('smoke','source-snapshots','load-resources.json','load-owner.log','smoke-owner.log'):
                archive.add(ROOT/p,arcname=p)
            for a in TARGETS:
                archive.add(ROOT/'load'/a/'summary.json',arcname=f'load/{a}/summary.json')
        meta=json.loads((out/'metadata/run_meta.json').read_text())
        key=undated(sampler_name(TARGETS[arm])); repo='dougalldeepmind/'+eval_name('mmlu',key)
        card=_card_fields('mmlu',OmegaConf.create(meta['config']),meta['command'],
            experiment=f'Native GPT-OSS MMLU matched screen: {arm}; valid={results[arm]["valid"]}',
            models=json.dumps(dict(target=TARGETS[arm],base='openai/gpt-oss-120b:peft:131072')),
            source_revision=meta['git_sha'])
        assert_layout(out)
        push_run_dir(out,repo,card,front_matter={'tags':run_tags('mmlu',key,'think')})
        revision=api.repo_info(repo,repo_type='dataset').sha
        files={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in out.rglob('*') if p.is_file() and '.cache' not in p.parts}
        remote=set(api.list_repo_files(repo,repo_type='dataset',revision=revision))
        assert set(files)<=remote
        def verify(item):
            name,digest=item
            path=hf_hub_download(repo,name,repo_type='dataset',revision=revision,token=hf_token())
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,name
        with ThreadPoolExecutor(4) as pool: list(pool.map(verify,files.items()))
        published[arm]=dict(repo=repo,revision=revision,files=len(files),sha256=files,
            url=f'https://huggingface.co/datasets/{repo}/tree/{revision}')
        save(ROOT/f'publication-{arm}.json',published[arm])
        print(json.dumps(dict(arm=arm,revision=revision,verified_files=len(files))),flush=True)
    save(ROOT/'publication.json',dict(complete=True,arms=published,at=time.time()))

if __name__=='__main__': main()
