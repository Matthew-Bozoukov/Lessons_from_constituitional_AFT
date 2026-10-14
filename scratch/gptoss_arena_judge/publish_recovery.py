# ABOUTME: Recover publication using the existing repository Hugging Face configuration.
# ABOUTME: Retain the failed finalizer evidence and perform no model calls or rescoring.
import time
import psutil
from dotenv import dotenv_values
from scratch.gptoss_arena_judge.campaign import ROOT,read,save
from scratch.gptoss_arena_judge.finish import publish
import os


def main():
    previous=read(ROOT/'finish-owner.json')
    assert not (psutil.pid_exists(previous['pid']) and abs(psutil.Process(previous['pid']).create_time()-previous['birth'])<.1)
    values=dotenv_values('C:/Users/nikak/source/repos/LASR/teaching_claude_why_replication/.env')
    assert values.get('HF_ORG')=='dougalldeepmind'
    for key in ('HF_ORG','HF_TOKEN','HUGGINGFACE_API_KEY'):
        if values.get(key): os.environ[key]=values[key]
    archive=ROOT/f'rate-limit-stop-publication-env-{time.time_ns()}';archive.mkdir()
    for name in ('finish-owner.json','finish-failure.json','finish-owner.log','finish-owner.err'):
        p=ROOT/name
        if p.exists(): p.rename(archive/name)
    save(ROOT/'finish-owner.json',dict(pid=os.getpid(),birth=psutil.Process().create_time(),phase='publication-only'))
    try: publish()
    except BaseException as exc:
        save(ROOT/'finish-failure.json',dict(at=time.time(),error=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__': main()
