# ABOUTME: Run only untouched base smoke tasks after a recorded local network failure.
# ABOUTME: Preserves interrupted paid attempts, original ledger, model settings, and all failure evidence.
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from scratch.gptoss_swe.openai_smoke import ROOT,TARGETS,IMAGE,environment,port,start_shim,stop_shim,save


def main(linux=False, arm='base'):
    root=ROOT/arm
    old=json.loads((root/'swe-recovered/metadata/state.json').read_text())
    pending=[i for i,t in old['tasks'].items() if t['status']=='pending' and not t['attempts']]
    assert len(pending)==(8 if arm=='base' else 2)
    assert sum(t['status']=='invalid' for t in old['tasks'].values())==2
    assert old['halt']=='infrastructure failure circuit breaker'
    if arm=='base':
        assert any('Network is unreachable' in p.read_text(errors='replace') for p in (root/'swe-recovered/rollouts').glob('*/*/agent.log'))
    else:
        assert 'Could not decode tokens: Invalid utf-8 sequence' in (root/'swe-shim.log').read_text(errors='replace')
    if linux:
        from omegaconf import OmegaConf
        from scratch.gptoss_swe.run import run
        cfg=OmegaConf.load(root/'swe.yaml')
        cfg.instance_ids=pending
        cfg.output_root='/work/'+(root/'swe-pending').as_posix()
        cfg.campaign+='-untouched-pending';cfg.run_name+='-untouched-pending'
        OmegaConf.save(cfg,root/'swe-pending.yaml')
        target=SimpleNamespace(base_url=f'http://host.docker.internal:{port(arm,"swe")}/v1',model_name='openai/gpt-oss-120b',spec=SimpleNamespace(hf_path=TARGETS[arm]))
        print(json.dumps(run(target,cfg,(root/'swe-pending').resolve())),flush=True)
        return
    with (root/'swe-pending-started.json').open('x') as f:
        json.dump(dict(at=time.time(),pending=pending,interrupted=[i for i,t in old['tasks'].items() if t['status']=='invalid'],original_state=old,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f,indent=2)
    name=f'lasr-gptoss-openai-{arm}-swe-shim'
    assert subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',name],text=True).strip()=='false'
    subprocess.run(['docker','rename',name,name+'-network-failed'],check=True)
    for n in ('swe-shim.json','swe-shim.log','swe-identity.json'):
        p=root/n
        if p.exists(): p.rename(root/(n+'.network-failed'))
    shim=None
    try:
        env=environment();shim=start_shim(arm,'swe',env)
        args=['docker','run','--name',f'lasr-gptoss-openai-{arm}-swe-pending-driver','--label','lasr.campaign=gptoss-openai-smoke-20261009','--add-host','host.docker.internal:host-gateway','-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock']
        for vol,path in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),('harness','src/eval/capabilities/swebench_mini/envs/harness')]:
            args+=['-v',f'lasr-gptoss-{vol}-env:/work/{path}/.venv']
        args+=['-e','TINKER_API_KEY',IMAGE,'scratch/swebench_cpu_env/.venv/bin/python','-m','scratch.gptoss_swe.continue_openai_pending','--linux','--arm',arm]
        subprocess.run(args,env=env,check=True)
        save(root/'swe-pending-finished.json',dict(at=time.time()))
    except BaseException as e:
        save(root/'swe-pending-failure.json',dict(at=time.time(),error=repr(e)));raise
    finally:
        if shim:stop_shim(arm,'swe',shim)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--linux',action='store_true');p.add_argument('--arm',choices=list(TARGETS),default='base');a=p.parse_args();main(a.linux,a.arm)
