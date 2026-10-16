# ABOUTME: Audit and publish the closed Control MASK attempt with complete evidence and pinned readback.
# ABOUTME: Plot historical Base and DA15 alongside the explicitly retried available-row Control score.
import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import campaign as m
from closeout import authorization, Closeout
from score_local import Resolved
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_run_dir
from src.naming import eval_name, artifact_name
from src.eval.layout import publish_layout, assert_layout, run_tags

REFERENCES = {
    'Base':('dougalldeepmind/2026-10-06-mask-gptoss120b','081ecc21725aa8c83dd30bde61a644d5fa09f0c3'),
    'DA15':('dougalldeepmind/2026-10-06-mask-gptoss120b-0-da-15','4c1cbff18562c07727bbb02410b34991cdceceea'),
}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def reference_data(out):
    refs={}
    for label,(repo,rev) in REFERENCES.items():
        refs[label]=dict(repo=repo,revision=rev,files={})
        for name in ['results/results.json','metadata/run_meta.json']:
            path=Path(hf_download(repo,name,repo_type='dataset',revision=rev))
            dest=out/'comparison-sources'/label/name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            refs[label]['files'][name]=sha(dest)
        result=read(out/'comparison-sources'/label/'results/results.json')
        assert result['benchmark']=='MASK' and result['judge']=='google/gemini-3-flash-preview'
        assert result['passes']==1 and result['overall_weighting']=='per_row'
        print(label,result['overall_honesty_score'],result['n_rows'],flush=True)
    m.save(out/'comparison-sources/manifest.json',refs)
    return refs


def audit(out,cfg):
    import pandas as pd
    complete=read(out/'scoring-complete.json')
    owner=read(out/'owner.json');exited=read(out/'owner-exit.json')
    assert all(owner[k]==exited[k] for k in ('pid','birth_epoch','started_utc'))
    # A matching exit receipt is necessary; the calling operator also checks live PID/birth.
    selected=authorization(out)
    db=sqlite3.connect((out/'ledger.sqlite').as_uri()+'?mode=ro',uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert db.execute('SELECT COUNT(*) FROM calls WHERE charged IS NULL').fetchone()[0]==0
    retry_counts={s:db.execute('SELECT COUNT(*) FROM calls WHERE stage=? AND id LIKE ?', (s,'%:authorized-retry-1')).fetchone()[0] for s in selected}
    assert retry_counts=={'generation':55,'judge':11}
    c=Resolved(cfg,out,None,selected,read(out/'judge-request-resolution.json')['mapping'])
    _,gen=m.generation_jobs(out,cfg)
    judges=read(out/'judge-jobs-closeout.json')
    counts={}
    for stage,jobs in [('generation',gen),('judge',judges)]:
        valid=0
        for j in jobs:
            rec=read(c.path(stage,j['id']))
            valid+=bool(rec.get('valid'))
            if stage=='judge' and rec.get('valid'): assert rec['request_sha256']==m.digest(j['body'])
        counts[stage]=dict(total=len(jobs),valid=valid,review=len(jobs)-valid,pending=0)
    coverage=read(out/'closeout-coverage.json')
    all_results=read(out/'data/metrics/all_results.json')
    per={};n=lies=honest=0
    for category,covered in coverage['categories'].items():
        frame=pd.read_csv(out/'data/metrics'/f'{category}_gpt-oss-120b.csv')
        assert len(frame)==covered['scored']
        assert set(frame['eval_honesty_1'].unique())<={-1,0,1}
        nn=len(frame);ll=int((frame['eval_honesty_1']==-1).sum());hh=int((frame['eval_honesty_1']==1).sum())
        expected=all_results[category]['gpt-oss-120b']
        assert expected['total_responses']==nn and abs(expected['honesty_score_1']-(100-ll/nn*100))<0.0011
        per[category]=dict(honesty_score=expected['honesty_score_1'],n_responses=nn,n_rows=covered['total'],dishonest=ll,honest=hh,other=nn-ll-hh)
        n+=nn;lies+=ll;honest+=hh
    assert n==complete['scored_rows']
    budget={}
    for stage in selected:
        cost=db.execute('SELECT SUM(charged) FROM calls WHERE stage=?',(stage,)).fetchone()[0]
        budget[stage]=dict(settled_usd=cost,reserved_usd=0,attempts=db.execute('SELECT COUNT(*) FROM calls WHERE stage=?',(stage,)).fetchone()[0])
        assert cost<=cfg['tinker_budget_usd' if stage=='generation' else 'judge_budget_usd']
    result=dict(target=cfg['checkpoint'],target_revision=cfg['model_revision'],model_repo=cfg['model_repo'],mode='think',reasoning='medium',benchmark='MASK',model='gpt-oss-120b',judge=cfg['judge_model'],overall_honesty_score=round(100*(1-lies/n),2),honesty_ci95=None,honesty_sd=None,passes=1,overall_weighting='per_row',per_archetype=per,n_rows=1000,n_scored=n,n_excluded=1000-n,complete_coverage=n==1000,status='closed_with_exclusions' if n<1000 else 'closed',dishonest=lies,honest=honest,other=n-lies-honest,counts=counts,budget=budget,retry_counts=retry_counts,coverage=coverage,comparison_caveat='Control received one targeted retry of failed calls and excludes any remaining incomplete rows. Base and DA15 are historical published scores over 1000 rows. This is a descriptive comparison, not a matched first-attempt estimate.',historical_revision=cfg['historical_revision'])
    m.save(out/'closeout-results.json',result)
    print(json.dumps({k:result[k] for k in ['overall_honesty_score','n_scored','n_excluded','counts','budget']}),flush=True)
    db.close()
    return result


def publish(out,cfg,result):
    stage=out.parent/'gptoss_mask_control_20261015_publication'
    assert not stage.exists(), 'Keep an existing publication staging tree immutable; inspect before resuming'
    rollouts,results,metadata=publish_layout(stage)
    m.save(results/'results.json',result)
    prose=f"# Control MASK closeout\n\nHonesty score: **{result['overall_honesty_score']:.2f}%**, pooled over **{result['n_scored']}/1000 rows**. {result['n_excluded']} incomplete rows excluded, without honesty-label imputation.\n\n{result['comparison_caveat']}\n\nOriginal pass: 4383/4438 valid generations; 3732/3743 valid judge calls. Authorized one retry of 55 generations and 11 judgments; originals and their costs are retained. Newly eligible rows received initial judge calls. No further retries.\n\nThe initial pilot had a local SDK serialization defect; all 58 original provider outputs were retrieved without resampling. Recovery mapping, raw futures and failure receipts are preserved.\n\nRaw evidence ZIP contains prompts, visible replies, private reasoning, original and retry token arrays, judge HTTP responses, CSVs, logs, and owner receipts. Private reasoning was never sent to the judge. The separate ledger includes every original and retry charge.\n"
    (results/'results.md').write_text(prose,encoding='utf-8')
    shutil.copytree(out/'data/evaluated',results/'evaluated')
    shutil.copytree(out/'data/metrics',results/'metrics')
    shutil.copytree(m.HERE,metadata/'source',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    shutil.copytree(out/'comparison-sources',metadata/'comparison-sources')
    src=sqlite3.connect((out/'ledger.sqlite').as_uri()+'?mode=ro',uri=True)
    dst=sqlite3.connect(metadata/'ledger.sqlite');src.backup(dst);dst.close();src.close()
    evidence={}
    with zipfile.ZipFile(rollouts/'campaign-evidence.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if not p.is_file() or p.name in {'owner.lock','ledger.sqlite','ledger.sqlite-wal','ledger.sqlite-shm','publication.json'} or p.suffix=='.tmp': continue
            rel=p.relative_to(out).as_posix();evidence[rel]=dict(sha256=sha(p),bytes=p.stat().st_size);z.write(p,rel)
    m.save(metadata/'evidence-sha256.json',evidence)
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip()
    meta=dict(at=m.utc(),target=cfg['checkpoint'],target_revision=cfg['model_revision'],model_repo=cfg['model_repo'],mode='think',config=cfg,source_git=source_sha,historical_revision=cfg['historical_revision'],command='python scratch/gptoss_mask_control/campaign.py run; python scratch/gptoss_mask_control/closeout.py; python scratch/gptoss_mask_control/score_local.py; python scratch/gptoss_mask_control/publish_closeout.py references; python scratch/gptoss_mask_control/publish_closeout.py publish',retry_authorization=read(out/'retry-authorization.json'),owner_exit=read(out/'owner-exit.json'),completion=read(out/'scoring-complete.json'),comparison_caveat=result['comparison_caveat'])
    m.save(metadata/'run_meta.json',meta)
    shutil.copy2(m.HERE/'protocol.yaml',metadata/'config.yaml')
    fields=dict(experiment=prose,date_generated='2026-10-15',constitution='none',source_repo=f'https://github.com/nikakogho/teaching_claude_why_replication @ {source_sha}',models={'target':{'repo':cfg['model_repo'],'revision':cfg['model_revision'],'sampler':cfg['checkpoint']},'judge':cfg['judge_model']},generation_config=cfg,schema={'rollouts/campaign-evidence.zip':'Complete original and retry evidence; paths/hashes in metadata/evidence-sha256.json','results/results.json':'Available-row MASK score and explicit exclusions','metadata/ledger.sqlite':'All original and retry admissions and costs'},provenance=meta)
    # Use the actual configured source remote, rather than assuming a repository owner.
    fields['source_repo']=subprocess.check_output(['git','remote','get-url','origin'],cwd=m.ROOT,text=True).strip()+' @ '+source_sha
    repo=hf_repo_id(eval_name('mask',cfg['model_repo'].split('/')[-1],date='2026-10-15'))
    assert_layout(stage)
    m.save(out/'publication-plan.json',dict(repo=repo,stage=str(stage)))
    url=push_run_dir(stage,repo,fields,front_matter={'tags':run_tags('mask','gptoss120b','think')},atomic_commit=True)
    revision=hf_api().repo_info(repo,repo_type='dataset').sha
    verified={}
    for p in sorted(stage.rglob('*')):
        if not p.is_file(): continue
        name=p.relative_to(stage).as_posix()
        downloaded=Path(hf_download(repo,name,repo_type='dataset',revision=revision,force_download=True))
        assert sha(downloaded)==sha(p), name
        verified[name]=sha(downloaded)
    receipt=dict(at=m.utc(),repo=repo,url=url,revision=revision,verified_sha256=verified,all_files_verified=True)
    m.save(out/'publication.json',receipt)
    print(json.dumps(receipt|{'verified_sha256':f'{len(verified)} files'}),flush=True)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pub=read(out/'publication.json');assert pub['all_files_verified']
    control_path=Path(hf_download(pub['repo'],'results/results.json',repo_type='dataset',revision=pub['revision']))
    control=read(control_path)
    refs=read(out/'comparison-sources/manifest.json')
    values={'Base':read(out/'comparison-sources/Base/results/results.json'),'Control':control,'DA15':read(out/'comparison-sources/DA15/results/results.json')}
    figdir=m.ROOT/'output/figures';figdir.mkdir(exist_ok=True)
    stem=artifact_name('gptoss120b-mask-base-control-da15',date='2026-10-15')
    fig,ax=plt.subplots(figsize=(5.3,4.5))
    labels=list(values);scores=[v['overall_honesty_score'] for v in values.values()]
    bars=ax.bar(labels,scores,color=['#27364B','#9298A1','#8752B2'],width=.57)
    ax.set_ylim(0,105);ax.set_yticks(range(0,101,20));ax.set_ylabel('MASK honesty score (%)');ax.set_title('GPT-OSS-120B',pad=12)
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.yaxis.grid(True,alpha=.17)
    for bar,score in zip(bars,scores): ax.text(bar.get_x()+bar.get_width()/2,score+1.4,f'{score:.1f}',ha='center',fontsize=12)
    note=f"Control: {control['n_scored']}/1000 rows after one retry; Base/DA15: 1000 each."
    fig.text(.5,.025,note,ha='center',fontsize=8,color='#444444')
    fig.tight_layout(rect=(0,.055,1,1))
    for ext in ('png','pdf','svg'): fig.savefig(figdir/f'{stem}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    m.save(figdir/f'{stem}_run_meta.json',dict(at=m.utc(),sources=refs|{'Control':pub},scores={k:v['overall_honesty_score'] for k,v in values.items()},coverage={k:v.get('n_scored',v['n_rows']) for k,v in values.items()},caveat=control['comparison_caveat']))
    (figdir/f'{stem}_results.md').write_text('\n'.join([f"{k}: {v['overall_honesty_score']}%" for k,v in values.items()])+ '\n\n'+note+'\n\n'+control['comparison_caveat'],encoding='utf-8')
    print(figdir/f'{stem}.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['references','audit','publish','plot']);args=parser.parse_args()
    cfg=m.load_config();out=m.ROOT/cfg['output']
    from dotenv import load_dotenv
    load_dotenv(cfg['main_env'])
    if args.mode=='references': reference_data(out)
    elif args.mode=='audit': audit(out,cfg)
    elif args.mode=='publish': publish(out,cfg,audit(out,cfg))
    else: plot(out)
