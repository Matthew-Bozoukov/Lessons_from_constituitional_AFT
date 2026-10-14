# ABOUTME: Apply the explicit user authorization to retry the last two failed judgments once.
# ABOUTME: Archive original attempts and stopped-owner evidence without changing prompts or settings.
import time
import psutil
from scratch.gptoss_arena_judge.campaign import ROOT,read,save,digest,status


def main():
    ids={'d342eae8c3028c78c97f5956c79110dfd4ad9c1790d19af2a1652372c822795d','41a475e8b7efeed00f6301cdca38c3ba2e78ca1de459d23fb2ddb6a6d82e0ce4'}
    for name in ('full-owner.json','finish-owner.json'):
        r=read(ROOT/name)
        assert not (psutil.pid_exists(r['pid']) and abs(psutil.Process(r['pid']).create_time()-r['birth'])<.1)
    assert {p.parent.name for p in (ROOT/'jobs').glob('*/review-needed.json')}==ids
    completed={p.parent.name:digest(p) for p in (ROOT/'jobs').glob('*/completed.json')}
    assert len(completed)==4492
    archive=ROOT/f'rate-limit-stop-final-two-retry-{time.time_ns()}';archive.mkdir()
    save(archive/'completed-hashes.json',completed)
    for jid in sorted(ids):
        d=ROOT/'jobs'/jid
        assert not (d/'completed.json').exists()
        assert read(d/'failure.json')['message']=='Judge timeout; retained start and output, no retry'
        dest=d/'user-authorized-retry-original';dest.mkdir()
        for name in ('started.json','process.json','command.json','events.jsonl','stderr.txt','final.txt','failure.json','review-needed.json'):
            p=d/name
            if p.exists(): p.rename(dest/name)
        save(dest/'authorization.json',dict(at=time.time(),user_instruction='retry',scope='One deliberate retry of each of the two unresolved judgments',prompt_sha256=digest(d/'prompt.txt'),identity_sha256=digest(d/'identity.json')))
    for name in ('full-owner.json','full-owner.log','full-owner.err','full-progress.json','full-source-sha.txt','finish-owner.json','finish-owner.log','finish-owner.err','finish-failure.json','review-needed.json'):
        p=ROOT/name
        if p.exists(): p.rename(archive/name)
    assert all(digest(ROOT/'jobs'/jid/'completed.json')==sha for jid,sha in completed.items())
    save(archive/'recovery.json',dict(at=time.time(),retry_ids=sorted(ids),preserved_completed=4492,authorization='User: retry',model_calls_during_preparation=0))
    print(status())


if __name__=='__main__': main()
