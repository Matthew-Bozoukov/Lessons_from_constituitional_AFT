# ABOUTME: Retain already completed subscription judgments rejected solely for a transport notice.
# ABOUTME: No model calls; archive the stopped owner and every original validator failure.
import time
import psutil
from scratch.gptoss_arena_judge.campaign import ROOT, read, rows, save, digest, validate_events, verdict, status


def recover():
    owner=read(ROOT/'full-owner.json')
    assert not psutil.pid_exists(owner['pid']) or abs(psutil.Process(owner['pid']).create_time()-owner['birth'])>.1
    recovered=[]
    for job in read(ROOT/'jobs.json'):
        d=ROOT/'jobs'/job['id']
        if not (d/'failure.json').exists(): continue
        failure=read(d/'failure.json')
        assert failure['message']=='Unexpected tool/error item in judge output',failure
        turn=validate_events(rows(d/'events.jsonl'))[0]
        final=read(d/'final.txt')
        score=verdict(final['verdict'])
        assert score and final['evaluation'].strip()
        assert read(d/'started.json')['request_hash']==job['id']
        save(d/'completed.json',dict(**job,score=score,usage=turn['usage'],
             seconds=failure['at']-read(d/'started.json')['at'],final_sha256=digest(d/'final.txt'),
             backend='codex-chatgpt',model='gpt-6-luna',reasoning_effort='low',
             recovered_without_model_call=True))
        (d/'failure.json').rename(d/'transport-validation-failure.json')
        recovered.append(job['id'])
    snapshot=status()
    assert not snapshot['failures'] and snapshot['incomplete_started']==0
    archive=ROOT/f'transport-validation-{int(time.time())}'
    archive.mkdir()
    for name in ('full-owner.json','full-progress.json','full-owner.log','full-owner.err','admission-hold.json','full-source-sha.txt'):
        (ROOT/name).rename(archive/name)
    save(archive/'recovery.json',dict(at=time.time(),jobs=recovered,model_calls=0))
    print(snapshot)


if __name__=='__main__': recover()
