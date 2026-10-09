# ABOUTME: Close out terminal partial smokes without restarting inference or altering budgets.
# ABOUTME: Waits for existing owners, grades saved valid patches once, publishes and removes own stopped containers.
import json
import subprocess
import sys
import time
from scratch.gptoss_swe.openai_smoke import ROOT,save


def main():
    with (ROOT/'finalize-started.json').open('x') as f:json.dump(dict(at=time.time()),f)
    try:
        deadline=time.monotonic()+4*3600
        while time.monotonic()<deadline:
            names=['lasr-gptoss-openai-control-swe-pending-driver','lasr-gptoss-openai-control-swe-shim','lasr-gptoss-openai-retained-grader-1']
            active=[subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',n],text=True).strip()=='true' for n in names]
            if not any(active):break
            time.sleep(15)
        else:raise TimeoutError('Existing owners did not finish; no inference restarted')
        with (ROOT/'retained-grading-final.log').open('w',encoding='utf-8') as log:
            subprocess.run(['docker','start','-a','lasr-gptoss-openai-retained-grader-1'],stdout=log,stderr=subprocess.STDOUT,check=True)
            assert subprocess.check_output(['docker','inspect','--format','{{.State.ExitCode}}','lasr-gptoss-openai-retained-grader-1'],text=True).strip()=='0'
        with (ROOT/'publication.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8','-m','scratch.gptoss_swe.package_openai_smoke'],stdout=log,stderr=subprocess.STDOUT,check=True)
        names=subprocess.check_output(['docker','ps','-a','--filter','label=lasr.campaign=gptoss-openai-smoke-20261009','--format','{{.Names}}'],text=True).split()
        removed=[]
        for name in names:
            item=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            assert name.startswith('lasr-gptoss-openai-') and not item['State']['Running']
            removed.append(dict(name=name,id=item['Id'],state=item['State']['Status']))
            subprocess.run(['docker','rm',name],check=True)
        assert not subprocess.check_output(['docker','ps','-aq','--filter','label=lasr.campaign=gptoss-openai-smoke-20261009'],text=True).strip()
        save(ROOT/'cleanup.json',dict(at=time.time(),removed=removed,owned_remaining=[]))
        (ROOT/'keep_awake.stop').touch()
        save(ROOT/'finalized.json',dict(at=time.time(),publication=json.loads((ROOT/'publication.json').read_text()),scope='Partial SWE smoke, complete ODCV smoke; no full benchmark completion claim'))
    except BaseException as e:
        save(ROOT/'finalize-failure.json',dict(at=time.time(),error=repr(e)));raise

if __name__=='__main__':main()
