# ABOUTME: One explicitly authorized retry, followed by transparent available-row MASK scoring.
# ABOUTME: Original responses and charges remain immutable; retry admissions have distinct durable IDs.
import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import campaign as m


class RetryLedger:
    def __init__(self, ledger, selected):
        self.ledger, self.selected = ledger, selected

    def key(self, stage, key):
        logical = key.rsplit('-', 1)[0] if stage == 'judge' else key
        return key + ':authorized-retry-1' if logical in self.selected[stage] else key

    def reserve(self, stage, key, *args):
        return self.ledger.reserve(stage, self.key(stage, key), *args)

    def finish(self, stage, key, *args):
        return self.ledger.finish(stage, self.key(stage, key), *args)

    def summary(self):
        return self.ledger.summary()


class Closeout(m.Campaign):
    def __init__(self, cfg, out, ledger, selected):
        super().__init__(cfg, out, RetryLedger(ledger, selected))
        self.selected = selected

    def path(self, stage, key, name='result.json'):
        if key in self.selected[stage]:
            return self.out/'authorized-retry-1'/stage/key/name
        return super().path(stage, key, name)


def authorization(out):
    path = out/'retry-authorization.json'
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
    else:
        record = dict(at=m.utc(), instruction='yeah retry once and then whatever happens call it a day, upload on HF and also give me visualization of mask results vertical bar chart of base vs control vs da15', selected={})
        for stage, expected in [('generation',55),('judge',11)]:
            record['selected'][stage] = {p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'calls'/stage).glob('*/result.json')) if not json.loads(p.read_text(encoding='utf-8')).get('valid')}
            assert len(record['selected'][stage]) == expected
        m.save(path, record)
    for stage, entries in record['selected'].items():
        for key, sha in entries.items():
            assert hashlib.sha256((out/'calls'/stage/key/'result.json').read_bytes()).hexdigest() == sha
    return {s:set(v) for s,v in record['selected'].items()}


async def run(cfg, out, ledger, selected):
    cfg = dict(cfg, max_explicit_rejection_attempts=1)
    c = Closeout(cfg,out,ledger,selected)
    frames,c.gen_jobs = m.generation_jobs(out,cfg)
    ticker = asyncio.create_task(c.ticker())
    try:
        from frozen.harmony import make_renderer
        import tinker
        from tinker.lib.retry_handler import RetryConfig
        renderer=make_renderer(cfg['reasoning'],current_date=cfg['render_date'])
        service=tinker.ServiceClient(max_retries=0)
        sampler=service.create_sampling_client(base_model=cfg['base_model'],model_path=cfg['checkpoint'],retry_config=RetryConfig(enable_retry_logic=False,enable_stuck_detection=True,progress_timeout=cfg['generation_timeout_s']))
        send=sampler._send_asample_request
        async def recorded(request_id,num_samples,prompt,*args,**kwargs):
            m.save(c.path('generation',m.CURRENT_JOB.get(),'provider-dispatch.json'),dict(sampler_id=sampler._sampling_session_id,request_id=f'{sampler._sampling_session_id}:{request_id}',prompt_sha256=m.digest(prompt.to_ints()),at=m.utc()))
            return await send(request_id,num_samples,prompt,*args,**kwargs)
        sampler._send_asample_request=recorded
        c.phase='authorized-generation-retry'
        await c.generate([j for j in c.gen_jobs if j['id'] in selected['generation']],renderer,sampler)
        if c.halt: raise m.Hold(c.halt)
        eligible=c.response_frames(frames)
        c.phase='authorized-judge-retry-and-newly-eligible'
        await c.evaluator(eligible)
        m.save(out/'judge-jobs-closeout.json',c.judge_jobs)
        assert len(c.judge_jobs)<=cfg['judge_calls_max']
        await c.judge(c.judge_jobs)
        if c.halt: raise m.Hold(c.halt)
        excluded={name:set() for name in frames}
        for j in c.judge_jobs:
            p=c.path('judge',j['id'])
            if not p.exists() or not json.loads(p.read_text(encoding='utf-8')).get('valid'):
                excluded[j['category']].add(j['row'])
        scoring={n:df.drop(index=sorted(excluded[n]),errors='ignore').copy() for n,df in eligible.items()}
        coverage={n:dict(total=len(frames[n]),generation_eligible=len(eligible[n]),scored=len(scoring[n]),excluded_generation=sorted(set(frames[n].index)-set(eligible[n].index)),excluded_judging=sorted(excluded[n])) for n in frames}
        m.save(out/'closeout-coverage.json',dict(at=m.utc(),policy='One authorized retry of 55 generation and 11 judge failures. Score only rows with all required valid generations and judgments; no failure imputation. Historical CSV blank-cell behavior and metric unchanged.',categories=coverage))
        await c.evaluator(scoring,replay=True)
        c.phase='scoring-available-rows';c.progress()
        env={**os.environ,'MASK_DATA_DIR':str(out/'data')}
        for script in ('metric.py','process_metrics.py'):
            with (out/(script+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(m.FROZEN/script)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        m.save(out/'scoring-complete.json',dict(at=m.utc(),budget=ledger.summary(),complete_coverage=all(v['scored']==v['total'] for v in coverage.values()),scored_rows=sum(v['scored'] for v in coverage.values()),total_rows=1000))
        c.phase='closed-awaiting-publication';c.progress()
    finally:
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError): await ticker


def main():
    cfg=m.load_config();out=m.ROOT/cfg['output']
    from dotenv import load_dotenv
    load_dotenv(cfg['main_env'])
    # Preserve previous owner receipts before obtaining the exclusive owner lock.
    history=out/'pre-retry-status';history.mkdir(exist_ok=True)
    for name in ['owner.json','owner-exit.json','progress.json','judge-jobs.json','generation-row-audit.json']:
        if (out/name).exists() and not (history/name).exists(): shutil.copy2(out/name,history/name)
    with m.owner(out):
        selected=authorization(out)
        ledger=m.Ledger(out/'ledger.sqlite',m.freeze(cfg,out))
        m.save(out/f'closeout-code-{int(time.time())}.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in m.HERE.glob('*.py')})
        try:
            m.preflight(cfg,out)
            asyncio.run(run(cfg,out,ledger,selected))
        except BaseException as e:
            m.save(out/'closeout-failure.json',dict(type=type(e).__name__,message=str(e),at=m.utc()))
            raise


if __name__=='__main__': main()
