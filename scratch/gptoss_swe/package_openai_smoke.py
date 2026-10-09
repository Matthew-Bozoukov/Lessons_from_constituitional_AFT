# ABOUTME: Publish both completed reference-Harmony smokes with all raw model failures retained.
# ABOUTME: Checks coverage, target identity, budgets, credential absence, and immutable HF hashes.
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from scratch.gptoss_swe.openai_smoke import ROOT, OLD, TARGETS, save

REPO='dougalldeepmind/2026-10-09-gptoss120b-base-control-openai-interface-smoke'


def main():
    from dotenv import dotenv_values
    from huggingface_hub import HfApi,hf_hub_download
    from scratch.gptoss_swe.audit_shared_smoke import main as audit
    keys=dotenv_values(Path.home()/'source/repos/LASR/teaching_claude_why_replication/.env')
    needles=[v.encode() for k,v in keys.items() if v and len(v)>16 and ('KEY' in k or 'TOKEN' in k)]
    summaries={}
    for arm,target in TARGETS.items():
        root=ROOT/arm
        assert (root/'odcv-judge-finished.json').exists()
        audit(root,root/'swe-pending',OLD/'cache',False)
        swe=json.loads((root/'swe-summary.json').read_text())
        assert swe['n_selected']==10 and swe['n_valid_rollouts']==swe['n_graded']
        assert swe['n_graded']+swe['n_interrupted']+swe['n_not_started']==10
        assert swe['checkpoint']==target
        odcvroot=Path(json.loads((root/'odcv-rollout.json').read_text())['path'])
        odcv=json.loads((odcvroot/'results.json').read_text())
        assert odcv['n_judged']==10 and odcv['n_dropped_all_na']==0
        costs={}
        for kind in ('compat','odcv','swe'):
            data=json.loads((root/(kind+'-budget.json')).read_text())
            assert data['authorized_checkpoints']==[target]
            assert all(x['state'] in ('completed','reserved') for x in data['requests'].values())  # unknown calls retain their full charge
            costs[kind]=sum(x['upper_usd'] for x in data['requests'].values())
        costs['judge']=sum(x['charged_or_reserved_usd'] for x in json.loads((root/'judge-budget.json').read_text()))
        info=json.loads((root/'audit.json').read_text())
        from collections import Counter
        info['swe']['ended_statuses']=dict(Counter(t['attempts'][-1]['exit_status'] for t in swe['task_statuses'].values() if t['status']=='valid'))
        info['swe']['task_statuses']=dict(Counter(t['status'] for t in swe['task_statuses'].values()))
        outcomes={}
        for iid,t in swe['task_statuses'].items():
            if t['status']!='valid': continue
            attempt=t['attempts'][-1]
            stage=Path(t['output_root']).name
            traj=root/stage/'rollouts'/iid/attempt['id']/iid/(iid+'.traj.json')
            data=json.loads(traj.read_text())
            outcomes[iid]=dict(exit_status=attempt['exit_status'],limit_reasons=sorted({m.get('extra',{}).get('limit_reason') for m in data['messages'] if m.get('extra',{}).get('limit_reason')}))
        info['swe']['outcomes']=outcomes
        info['swe']['stage_halts']={st:json.loads((root/st/'metadata/state.json').read_text()).get('halt') for st in ('swe-recovered','swe-pending')}

        summaries[arm]=dict(target=target,swe=swe,odcv=odcv,diagnostics=info,costs_usd=costs)
    assert sum(sum(x['costs_usd'].values()) for x in summaries.values())<=36
    out=ROOT/'package'
    assert not out.exists(), 'Existing package must be verified/published without rebuilding'
    for folder in ('results','metadata','rollouts'): (out/folder).mkdir(parents=True)
    def archive(destination, paths):
        with tarfile.open(destination,'w:gz') as tar:
            for path in paths:
                for f in sorted(path.rglob('*')) if path.is_dir() else [path]:
                    if not f.is_file(): continue
                    raw=f.read_bytes()
                    assert not any(n in raw for n in needles), f'Credential detected in {f}'
                    tar.add(f,arcname=f.relative_to(ROOT),recursive=False)
    for arm in TARGETS:
        root=ROOT/arm
        archive(out/'rollouts'/f'{arm}-odcv.tar.gz',[root/'odcv'])
        archive(out/'rollouts'/f'{arm}-swe.tar.gz',[root/'swe-recovered/rollouts']+[root/'swe-pending/rollouts'])
        archive(out/'metadata'/f'{arm}-raw-interface.tar.gz',[root/(k+'-traces') for k in ('compat','odcv','swe')])
        archive(out/'metadata'/f'{arm}-sources-grading-state.tar.gz',[root/'swe-recovered/metadata',root/'swe-recovered/results']+[root/'swe-pending/metadata',root/'swe-pending/results'])
        archive(out/'metadata'/f'{arm}-startup-failures.tar.gz',[root/'swe',root/'swe-startup-failure'])
        archive(out/'metadata'/f'{arm}-driver-evidence.tar.gz',[p for p in root.iterdir() if p.is_file()])
        save(out/'results'/f'{arm}.json',summaries[arm])
    for name in ('prepared.json','images-verified.json','interface-audit.json'):
        shutil.copyfile(ROOT/name,out/'metadata'/name)
    compatibility=Path('output/openai-gpt-oss-reference/compatibility-test')
    for name in ('providers.ts','package-lock.json','cases.jsonl','tools.ts','runCase.ts','index.ts'):
        shutil.copyfile(compatibility/name,out/'metadata'/('official-compatibility-'+name))
    manifest=dict(base_model='openai/gpt-oss-120b',targets=TARGETS,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol='OpenAI reference Harmony 0.0.8; T1/top_p1/top_k disabled; medium reasoning',
        scope='Matched ten-cell ODCV and ten-task SWE diagnostic smoke per arm; not full benchmark scores',
        costs_usd={a:r['costs_usd'] for a,r in summaries.items()},invoice=False,
        caveats=['All outcomes retained; no malformed-generation resampling.',
                 'Base SWE first two paid attempts interrupted by local network failure/circuit breaker; only untouched tasks continued with original ledger. Control also had a strict UTF-8 logging error; paid interruptions were not rerun. Terminal budget holds, if present, are separate from model limits. All evidence retained.',
                 'SWE initial startup rejected disabled top_k before any requests; original failures kept and same ledgers reused.',
                 'Compatibility wrapper initially failed on Windows UTF-8 decoding after cases ended; retained results audited without rerun.',
                 'Official compatibility quality failures are reported, not hidden or repaired.',
                 'Control strict UTF-8 diagnostic decoding crashed after one paid sample; that exact completion was not written before the error. All available evidence is preserved. The subsequent general fix saves exact tokens first and uses the reference replacement decoder for display. Only two untouched control tasks used this fix.',
                 'Base spending admission stopped the remaining stage; interrupted and unattempted tasks are not model limit outcomes. No spending ceilings were raised.',
                 'Numerical response/context/step limits retained; sampling differs from the previous smoke.',
                 'No AWS, Vast CPU, or RunPod GPU rentals. Local Docker caches retained.'])
    save(out/'metadata/manifest.json',manifest)
    lines=['---','license: mit','---','# GPT-OSS-120B base/control OpenAI-interface smoke','',
        'Matched diagnostic subsets only. Official OpenAI Harmony rendering/parsing, no custom JSON coaching, no malformed-generation resampling.',
        '', '| Metric | Base | Control |','|---|---:|---:|']
    def add(label,fn): lines.append('| '+label+' | '+' | '.join(str(fn(summaries[a])) for a in TARGETS)+' |')
    add('SWE officially resolved / graded',lambda r:str(r['swe']['n_resolved'])+'/'+str(r['swe']['n_graded']))
    add('SWE interrupted / 10',lambda r:r['swe']['n_interrupted'])
    add('SWE not started / 10',lambda r:r['swe']['n_not_started'])
    add('SWE limit-ended among graded',lambda r:r['diagnostics']['swe']['ended_statuses'].get('LimitsExceeded',0))
    add('ODCV task_complete / 10',lambda r:sum(c['tool_completed'] for c in r['diagnostics']['odcv']['cells']))
    add('ODCV context / cycle / response limits',lambda r:'/'.join(str(sum(c[k] for c in r['diagnostics']['odcv']['cells'])) for k in ('context_limit','step_limit','response_limit')))
    add('ODCV diagnostic misalignment %',lambda r:r['odcv']['ours']['overall']['mr_pct'])
    add('Accounted USD, not invoice',lambda r:round(sum(r['costs_usd'].values()),6))
    lines+=['','Both evaluations: T1, top_p1, top_k disabled, medium reasoning. ODCV: 8192 output, 28000 context, 50 cycles. SWE: 16384 output, 131072 context, 262144 generated tokens per task, 500 steps. Local Docker, no cloud CPU/GPU rental.',
            '', 'Limit-ended attempts can retain working patches; only official grading establishes SWE resolution. SWE coverage is partial: see the exact graded, interrupted, and not-started counts above. Interrupted paid attempts were not rerun. This prevents a clean paired SWE score comparison. ODCV misconduct and task completion measure different things.',
            '', 'Read results/*.json for per-conversation malformed JSON, immediate corrections, no-tool streaks, limit reasons and grades. metadata/ contains rendered prompt tokens, raw responses, original startup failures, configs, budgets and source snapshots.',
            '', 'Sources: [OpenAI Harmony](https://github.com/openai/harmony), [implementation verification](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations), [reasoning history](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot), [sampling](https://github.com/openai/gpt-oss#recommended-sampling-parameters).',
            '', 'The official live compatibility cases were run once each: base 27/30, control 28/30. Base did not exceed the suggested 90% quality threshold; API shapes were valid. Failures were one malformed JSON and two missing expected calls on base, and two unoffered tool names on control. This is not full streaming/Responses API certification.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    save(out/'metadata/file-hashes.json',hashes)
    hashes['metadata/file-hashes.json']=hashlib.sha256((out/'metadata/file-hashes.json').read_bytes()).hexdigest()
    api=HfApi(token=keys['HF_TOKEN'])
    api.create_repo(REPO,repo_type='dataset',exist_ok=True)
    revision=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=out,
        commit_message='Publish matched base/control OpenAI-interface smoke with all outcomes').oid
    receipt=dict(repo=REPO,revision=revision,verified_files=0,sha256=hashes,complete=False)
    save(ROOT/'publication.json',receipt)
    for name,expected in hashes.items():
        downloaded=hf_hub_download(REPO,name,repo_type='dataset',revision=revision,token=keys['HF_TOKEN'])
        assert hashlib.sha256(Path(downloaded).read_bytes()).hexdigest()==expected,name
        receipt['verified_files']+=1
    receipt['complete']=True
    save(ROOT/'publication.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'}))


if __name__=='__main__': main()
