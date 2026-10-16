# ABOUTME: Retain one user-accepted wait-call judgment without another model request.
# ABOUTME: Preserve stopped-owner receipts and original validation failure before continuation.
import time
import psutil
from scratch.gptoss_arena_judge.campaign import ROOT,read,save,digest,verdict,validate_job_events,status


def main():
    for name in ('full-owner.json','finish-owner.json'):
        r=read(ROOT/name)
        assert not (psutil.pid_exists(r['pid']) and abs(psutil.Process(r['pid']).create_time()-r['birth'])<.1)
    jid='4e5c95bc63dba6df301a73a97ba2ddbe8f6ed77c8e7924acd92cb2c4e739410d'
    d=ROOT/'jobs'/jid
    assert not (d/'completed.json').exists()
    failure=read(d/'failure.json');assert failure['message']=='Unexpected tool/error item in judge output'
    save(d/'accepted-tool-exception.json',dict(job=jid,at=time.time(),authorization='User: prohibited call record that as its response and continue',events_sha256=digest(d/'events.jsonl'),final_sha256=digest(d/'final.txt'),reason='Retain existing final and disclose empty-recipient wait call; no rerun'))
    turn=validate_job_events(d)[0];answer=read(d/'final.txt');score=verdict(answer['verdict'])
    assert score and answer['evaluation'].strip()
    job=next(j for j in read(ROOT/'jobs.json') if j['id']==jid)
    assert read(d/'started.json')['request_hash']==jid
    save(d/'completed.json',dict(**job,score=score,usage=turn['usage'],seconds=failure['at']-read(d/'started.json')['at'],final_sha256=digest(d/'final.txt'),backend='codex-chatgpt',model='gpt-6-luna',reasoning_effort='low',recovered_without_model_call=True,accepted_tool_exception=True))
    (d/'failure.json').rename(d/'original-tool-validation-failure.json')
    archive=ROOT/f'rate-limit-stop-accepted-wait-{time.time_ns()}';archive.mkdir()
    for name in ('full-owner.json','full-owner.log','full-owner.err','full-progress.json','admission-hold.json','full-source-sha.txt','finish-owner.json','finish-owner.log','finish-owner.err','finish-failure.json'):
        p=ROOT/name
        if p.exists(): p.rename(archive/name)
    save(archive/'recovery.json',dict(at=time.time(),job=jid,model_calls=0,remaining_unstarted=83,timeout_reviews=2))
    print(status())


if __name__=='__main__': main()
