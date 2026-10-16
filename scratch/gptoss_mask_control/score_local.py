# ABOUTME: Replay saved judgments by exact request hash, independent of transient dataframe column order.
# ABOUTME: This offline closeout never calls providers and preserves every original and retry response.
import asyncio
from collections import defaultdict, deque
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import campaign as m
from closeout import Closeout, authorization


class Resolved(Closeout):
    def __init__(self,cfg,out,ledger,selected,mapping):
        super().__init__(cfg,out,ledger,selected)
        self.mapping=mapping

    def path(self,stage,key,name='result.json'):
        if stage=='judge' and key in self.mapping:
            return self.out/self.mapping[key]/name
        return super().path(stage,key,name)


def resolve(out,cfg,selected,desired):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    original=read(out/'pre-retry-status/judge-jobs.json')
    existing={j['id']:j for j in original}
    for j in desired:
        if j['id'] not in existing: existing[j['id']]=j
    c=Closeout(cfg,out,None,selected)
    candidates=defaultdict(deque)
    for key,j in sorted(existing.items()):
        p=c.path('judge',key)
        body=read(p.parent/'request.json'); rec=read(p)
        assert rec['request_sha256']==m.digest(body)
        if key in selected['judge']:
            assert body==read(out/'calls/judge'/key/'request.json'), 'Retry changed the authorized request'
        candidates[(j['category'],j['row'],m.digest(body))].append(p.parent.relative_to(out).as_posix())
    mapping={}
    for j in desired:
        group=(j['category'],j['row'],m.digest(j['body']))
        assert candidates[group], f'No exact saved request for {j["id"]}'
        mapping[j['id']]=candidates[group].popleft()
    assert not any(candidates.values()), 'Unmatched extra paid judgments'
    return mapping


async def score(cfg,out,ledger,selected):
    c=Closeout(cfg,out,ledger,selected)
    frames,c.gen_jobs=m.generation_jobs(out,cfg)
    eligible=c.response_frames(frames)
    await c.evaluator(eligible)
    recorded=json.loads((out/'judge-jobs-closeout.json').read_text(encoding='utf-8'))
    assert c.judge_jobs==recorded
    mapping=resolve(out,cfg,selected,c.judge_jobs)
    m.save(out/'judge-request-resolution.json',dict(at=m.utc(),reason='Recovering first-row generations changed dataframe column insertion order, changing ordinal IDs for 255 already judged calls. Exact request-hash multisets within each row match; replay uses an explicit bijection without new calls or verdict selection.',mapping=mapping))
    r=Resolved(cfg,out,ledger,selected,mapping)
    r.gen_jobs=c.gen_jobs;r.judge_jobs=c.judge_jobs
    excluded={n:set() for n in frames}
    for j in r.judge_jobs:
        rec=json.loads(r.path('judge',j['id']).read_text(encoding='utf-8'))
        if not rec.get('valid'): excluded[j['category']].add(j['row'])
    scoring={n:df.drop(index=sorted(excluded[n]),errors='ignore').copy() for n,df in eligible.items()}
    coverage={n:dict(total=len(frames[n]),generation_eligible=len(eligible[n]),scored=len(scoring[n]),excluded_generation=sorted(set(frames[n].index)-set(eligible[n].index)),excluded_judging=sorted(excluded[n])) for n in frames}
    m.save(out/'closeout-coverage.json',dict(at=m.utc(),policy='One authorized retry. Exclude rows with any incomplete required generation or judgment; no failure imputation. Frozen historical scoring and CSV blank-cell handling unchanged.',categories=coverage))
    await r.evaluator(scoring,replay=True)
    env={**os.environ,'MASK_DATA_DIR':str(out/'data')}
    for script in ('metric.py','process_metrics.py'):
        with (out/(script+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(m.FROZEN/script)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    m.save(out/'scoring-complete.json',dict(at=m.utc(),budget=ledger.summary(),complete_coverage=all(v['scored']==v['total'] for v in coverage.values()),scored_rows=sum(v['scored'] for v in coverage.values()),total_rows=1000,request_resolution='judge-request-resolution.json'))
    r.phase='closed-awaiting-publication';r.progress()


if __name__=='__main__':
    cfg=m.load_config();out=m.ROOT/cfg['output'];selected=authorization(out)
    history=out/'paid-retry-status';history.mkdir(exist_ok=True)
    for name in ['owner.json','owner-exit.json','progress.json']:
        if not (history/name).exists(): shutil.copy2(out/name,history/name)
    with m.owner(out):
        # Construct the read-only ledger interface without its writable initializer.
        ledger=object.__new__(m.Ledger)
        ledger.db=sqlite3.connect((out/'ledger.sqlite').as_uri()+'?mode=ro',uri=True)
        asyncio.run(score(cfg,out,ledger,selected))
