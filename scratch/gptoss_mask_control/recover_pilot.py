# ABOUTME: Retrieve original pilot futures after a local SDK-dataclass serialization defect.
# ABOUTME: Read-only provider retrieval; no generation, no judge, no resampling or replaced evidence.
import asyncio
import json
from pathlib import Path
import shutil
import sqlite3
import campaign as c

SAMPLER='0fae1641-6301-5218-9f7c-b616fb786be6:sample:0'


def main():
    from dotenv import load_dotenv
    cfg=c.load_config();load_dotenv(cfg['main_env']);out=c.ROOT/cfg['output']
    # Require original owner to have exited before changing any local receipts.
    original=json.loads((out/'owner.json').read_text())
    exit_record=json.loads((out/'owner-exit.json').read_text())
    assert original['pid']==exit_record['pid'] and original['birth_epoch']==exit_record['birth_epoch']
    import tinker
    from tinker.lib.internal_client_holder import ClientConnectionPoolType
    from tinker.types.future_retrieve_request import FutureRetrieveRequest
    service=tinker.ServiceClient(max_retries=0); rest=service.create_rest_client()
    info=rest.get_sampler(SAMPLER).result().model_dump()
    assert info['model_path']==cfg['checkpoint'] and info['base_model']==cfg['base_model']
    recovery=out/'provider-recovery'; recovery.mkdir(exist_ok=True)
    c.save(recovery/'sampler.json',info)
    ledger=c.Ledger(out/'ledger.sqlite',json.loads((out/'identity.json').read_text()))
    rows=ledger.db.execute("SELECT id,hash FROM calls WHERE stage='generation' ORDER BY rowid").fetchall()
    assert len(rows)==58
    # There is one SamplingClient, an original (not cloned) counter starting at 0,
    # no retries, and no awaits between admission/save and scheduling sample_async.
    # asyncio schedules wait_for-created tasks FIFO; SDK assigns seq_id before its
    # first awaited dispatch. Thus this durable admission order is the request order.
    async def fetch():
        with rest.holder.aclient(ClientConnectionPoolType.RETRIEVE_FUTURES) as client:
            for i,(key,expected) in enumerate(rows):
                req=json.loads((out/'calls/generation'/key/'request.json').read_text(encoding='utf-8'))
                assert c.digest(req)==expected
                p=recovery/f'future-{i}.json'
                if not p.exists():
                    raw=await client.futures.with_raw_response.retrieve(request=FutureRetrieveRequest(request_id=f'{SAMPLER}:{i}'),timeout=30,max_retries=0)
                    assert raw.status_code==200
                    c.save(p,await raw.json())
                d=json.loads(p.read_text());assert d['type']=='sample' and len(d['sequences'])==1
    rest.holder.run_coroutine_threadsafe(fetch()).result()
    from frozen.harmony import make_renderer
    renderer=make_renderer(cfg['reasoning'],current_date=cfg['render_date'])
    mappings=[]
    for i,(key,_) in enumerate(rows):
        jobdir=out/'calls/generation'/key; req=json.loads((jobdir/'request.json').read_text(encoding='utf-8'))
        d=json.loads((recovery/f'future-{i}.json').read_text()); seq=d['sequences'][0]; ids=seq['tokens']
        assert len(ids)<=cfg['max_tokens']
        c.save(jobdir/'sample.json',d)
        parsed,term=renderer.parse_response(ids)
        message=renderer.to_openai_message(parsed) if term.is_stop_sequence else {'role':'assistant','content':renderer.tokenizer.decode(ids)}
        valid=bool(term.is_stop_sequence and not message.get('tool_calls') and not parsed.get('unparsed_tool_calls'))
        cost=(len(req['prompt_tokens'])*cfg['tinker_input_per_million']+len(ids)*cfg['tinker_output_per_million'])/1e6
        old=jobdir/'original-serialization-failure.json'
        if not old.exists():shutil.copy2(jobdir/'result.json',old)
        c.save(jobdir/'result.json',dict(valid=valid,message=message,finish_reason='stop' if valid else 'incomplete_or_tool',
             stop_reason=seq['stop_reason'],prompt_tokens=len(req['prompt_tokens']),completion_tokens=len(ids),cost_upper_usd=cost,
             recovered_from=f'{SAMPLER}:{i}',at=c.utc()))
        ledger.finish('generation',key,'completed' if valid else 'review',cost)
        mappings.append(dict(job_id=key,request_id=f'{SAMPLER}:{i}',valid=valid,original_input_sha256=c.digest(req),response_sha256=c.digest(d)))
    c.save(recovery/'recovery.json',dict(mappings=mappings,no_generation_calls=True,mapping_basis='Single-client FIFO admission and request counter; retries disabled; all 58 original futures retrieved',budget=ledger.summary(),at=c.utc()))
    print(json.dumps({'recovered':len(rows),'valid':sum(r['valid'] for r in mappings),'budget':ledger.summary()}))


if __name__=='__main__':main()
