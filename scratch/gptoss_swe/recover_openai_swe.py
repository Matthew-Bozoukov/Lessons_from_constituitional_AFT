# ABOUTME: Recover only the pre-inference disabled-top-k startup failure in the paired smoke.
# ABOUTME: Keeps original failures and original empty budgets; refuses any previously sampled work.
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from scratch.gptoss_swe.openai_smoke import ROOT, TARGETS, IMAGE, environment, port, start_shim, stop_shim, save


def linux(arm):
    from omegaconf import OmegaConf
    from scratch.gptoss_swe.run import run
    cfg=OmegaConf.load(ROOT/arm/'swe.yaml')
    cfg.output_root='/work/'+(ROOT/arm/'swe-recovered').as_posix()
    cfg.campaign+='-startup-recovered'
    cfg.run_name+='-startup-recovered'
    OmegaConf.save(cfg,ROOT/arm/'swe-recovery.yaml')
    target=SimpleNamespace(base_url=f'http://host.docker.internal:{port(arm,"swe")}/v1',
        model_name='openai/gpt-oss-120b',spec=SimpleNamespace(hf_path=TARGETS[arm]))
    print(json.dumps(run(target,cfg,(ROOT/arm/'swe-recovered').resolve())),flush=True)


def host(arm):
    root=ROOT/arm
    assert json.loads((root/'swe-budget.json').read_text())['requests']=={}
    assert not list((root/'swe-traces').glob('*/request.json'))
    state=json.loads((root/'swe/metadata/state.json').read_text())
    assert all(t['status'] in ('pending','invalid') for t in state['tasks'].values())
    logs=list((root/'swe/rollouts').glob('*/*/agent.log'))
    assert len(logs)==2
    assert all("sampling['top_k'] > 0" in p.read_text() and 'AssertionError' in p.read_text() for p in logs)
    with (root/'swe-recovery-started.json').open('x') as f:
        json.dump(dict(at=time.time(),reason='Old Qwen validator rejected disabled top_k before any sampling',
            original_root=str(root/'swe'),recovered_root=str(root/'swe-recovered'),
            original_state=state,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f,indent=2)
    backup=root/'swe-startup-failure'
    backup.mkdir()
    for suffix in ('-shim.json','-shim.log','-identity.json','-failure.json','-budget.json'):
        p=root/('swe'+suffix)
        if p.exists(): shutil.copyfile(p,backup/p.name)
    name='lasr-gptoss-openai-'+arm+'-swe-shim'
    instance=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
    assert not instance['State']['Running']
    assert instance['Config']['Labels']['lasr.campaign']=='gptoss-openai-smoke-20261009'
    subprocess.run(['docker','rename',name,name+'-startup-failed'],check=True)
    shim=None
    try:
        env=environment()
        shim=start_shim(arm,'swe',env)
        args=['docker','run','--name','lasr-gptoss-openai-'+arm+'-swe-recovery-driver',
            '--label','lasr.campaign=gptoss-openai-smoke-20261009','--add-host','host.docker.internal:host-gateway',
            '-v',f'{Path.cwd()}:/work','-v','/var/run/docker.sock:/var/run/docker.sock']
        for vol,path in [('cpu','scratch/swebench_cpu_env'),('agent','src/eval/capabilities/swebench_mini/envs/agent'),
                         ('harness','src/eval/capabilities/swebench_mini/envs/harness')]:
            args+=['-v',f'lasr-gptoss-{vol}-env:/work/{path}/.venv']
        args+=['-e','TINKER_API_KEY',IMAGE,'scratch/swebench_cpu_env/.venv/bin/python',
               '-m','scratch.gptoss_swe.recover_openai_swe','--linux','--arm',arm]
        subprocess.run(args,env=env,check=True)
        save(root/'swe-recovery-finished.json',dict(at=time.time(),authoritative_output=str(root/'swe-recovered')))
    except BaseException as error:
        save(root/'swe-recovery-failure.json',dict(at=time.time(),error=repr(error)))
        raise
    finally:
        if shim: stop_shim(arm,'swe',shim)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=list(TARGETS),required=True)
    parser.add_argument('--linux',action='store_true')
    args=parser.parse_args()
    (linux if args.linux else host)(args.arm)
