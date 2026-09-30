# ABOUTME: Continue an already launched control training through export, transport, two ODCV runs and comparison.
# ABOUTME: Sequential and fail-fast; preserves successful stages and never silently rerolls a partial evaluation.
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='scratch/gptoss_control/control_refresh.yaml')
    args=p.parse_args()
    os.chdir(ROOT)
    cfg=OmegaConf.load(args.config)
    out=ROOT/cfg.output
    # This process neither launches nor restarts training. The active training driver owns it.
    while not (out/'trained_adapter.json').exists():
        log=(out/'train_console.log').read_text(encoding='utf-8',errors='replace')
        if 'Traceback (most recent call last)' in log:
            raise RuntimeError('Training stopped with an error; inspect before any recovery')
        time.sleep(10)
    def run(stage,script,*argv):
        print('Starting',stage,flush=True)
        with (out/(stage+'_console.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,script,*argv],cwd=ROOT,
                env={**os.environ,'HF_HUB_DISABLE_XET':'1','PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} failed; inspect its owned console log')
        print('Completed',stage,flush=True)
    if not (out/'published_adapter.json').exists():
        run('export','scratch/gptoss_control/run.py','export','--config',args.config)
    if not (out/'adapter_transport/passed.json').exists():
        checkpoint=json.loads((out/'trained_adapter.json').read_text())['sampler']
        run('transport','scratch/gptoss_control/transport_smoke.py',checkpoint,
            '--output',str(out/'adapter_transport'),'--port','18332')
    for variant in ['original','fixed']:
        if not (out/('published_eval_'+variant+'.json')).exists():
            run('eval_'+variant,'scratch/gptoss_control/refresh_control.py','eval-'+variant,'--config',args.config)
    if not (out/'published_comparison.json').exists():
        run('comparison','scratch/gptoss_control/compare_prompt_runs.py','--config',args.config,'--publish')
    print('Control refresh campaign complete',flush=True)


if __name__=='__main__':
    main()
