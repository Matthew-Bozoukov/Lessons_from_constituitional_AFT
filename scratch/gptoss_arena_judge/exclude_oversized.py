# ABOUTME: Apply the user-approved whole-question exclusion without deleting judge evidence.
# ABOUTME: Requires stopped owners and preserves completion hashes before scheduler continuation.
import shutil
import time
import psutil
from scratch.gptoss_arena_judge.campaign import ROOT, read, save, digest, status


def main():
    for name in ('full-owner.json','finish-owner.json'):
        owner=read(ROOT/name)
        assert not (psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['birth'])<.1), name
    assert not (ROOT/'exclusions.json').exists()
    jobs=read(ROOT/'jobs.json')
    excluded=[j for j in jobs if j['uid']=='c41838dac4504b21']
    assert len(excluded)==6
    assert not any((ROOT/'jobs'/j['id']/'completed.json').exists() for j in excluded)
    completed={j['id']:digest(ROOT/'jobs'/j['id']/'completed.json') for j in jobs if (ROOT/'jobs'/j['id']/'completed.json').exists()}
    assert len(completed)==3468
    archive=ROOT/f'rate-limit-stop-input-exclusion-{time.time_ns()}'
    archive.mkdir()
    save(archive/'completed-hashes.json',completed)
    for name in ('full-owner.json','full-owner.log','full-owner.err','full-progress.json','admission-hold.json','full-source-sha.txt','finish-owner.json','finish-owner.log','finish-owner.err','finish-failure.json'):
        path=ROOT/name
        if path.exists():
            shutil.copy2(path,archive/name)
            for attempt in range(50):
                try:
                    path.unlink()
                    break
                except PermissionError:
                    time.sleep(.2)
            assert not path.exists(), name
    save(ROOT/'exclusions.json',dict(at=time.time(),authorization='User: exclude them; continue other judgments',reason='CLI input character limit; entire matched question excluded without modifying answers or imputing scores',jobs=excluded))
    assert all(digest(ROOT/'jobs'/i/'completed.json')==sha for i,sha in completed.items())
    save(archive/'recovery.json',dict(at=time.time(),preserved_completed=len(completed),excluded=6,remaining=1026))
    print(status())


if __name__=='__main__': main()
