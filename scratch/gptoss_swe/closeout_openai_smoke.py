# ABOUTME: Durable local closeout for the authorized paired smoke; judges completed rollouts once.
# ABOUTME: Never launches inference, resets claims, retries outcomes, or touches unrelated containers.
import json
from pathlib import Path
import subprocess
import sys
import time
from scratch.gptoss_swe.openai_smoke import ROOT, TARGETS, save


def main():
    with (ROOT/'closeout-v2-started.json').open('x') as stream:
        json.dump(dict(at=time.time()),stream)
    try:
        deadline=time.monotonic()+6*3600
        while time.monotonic()<deadline:
            for arm in TARGETS:
                root=ROOT/arm
                for kind in ('odcv', 'swe-pending' if arm=='base' else 'swe-recovery'):
                    assert not (root/(kind+'-failure.json')).exists(), f'{arm} {kind} failed; retain state'
                if (root/'odcv-finished.json').exists() and not (root/'odcv-judge-started.json').exists():
                    with (root/'odcv-judge.log').open('w',encoding='utf-8') as log:
                        subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.openai_smoke',
                            'judge-odcv','--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
            if all((ROOT/a/('swe-pending-finished.json' if a=='base' else 'swe-recovery-finished.json')).exists() and (ROOT/a/'odcv-judge-finished.json').exists() for a in TARGETS):
                break
            time.sleep(15)
        else:
            raise TimeoutError('Closeout wait exceeded six hours; no inference restarted')
        with (ROOT/'publication.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.package_openai_smoke'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        names=subprocess.check_output(['docker','ps','-a','--filter','label=lasr.campaign=gptoss-openai-smoke-20261009',
                                       '--format','{{.Names}}'],text=True).split()
        cleanup=[]
        for name in names:
            detail=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            assert not detail['State']['Running'], 'Expected stopped stage container: '+name
            assert name.startswith('lasr-gptoss-openai-')
            cleanup.append(dict(name=name,id=detail['Id'],state=detail['State']['Status']))
            subprocess.run(['docker','rm',name],check=True)
        remaining=subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign=gptoss-openai-smoke-20261009'],text=True).strip()
        assert not remaining
        save(ROOT/'cleanup.json',dict(at=time.time(),removed=cleanup,owned_remaining=[]))
        (ROOT/'keep_awake.stop').touch()
        save(ROOT/'complete.json',dict(at=time.time(),publication=json.loads((ROOT/'publication.json').read_text())))
    except BaseException as error:
        save(ROOT/'closeout-v2-failure.json',dict(at=time.time(),error=repr(error)))
        raise


if __name__=='__main__': main()
