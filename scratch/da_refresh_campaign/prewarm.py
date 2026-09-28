# ABOUTME: Builds ODCV scenario images locally while GPUs train, retaining shared build layers.
# ABOUTME: Starts no scenario containers and never prunes or modifies another campaign's resources.
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
from src.eval.misalignment.odcv.odcv import VARIANTS,scenario_names
from src.eval.misalignment.odcv.odcv_rollout import _build_workspace
from scratch.da_supervision.odcv_eval import process_workdir
from scratch.da_refresh_campaign.run import OUT,save


def build(cell):
    variant,scenario=cell
    key=hashlib.sha256(f'{variant}/{scenario}'.encode()).hexdigest()[:12]
    ws=OUT/'prewarm'/key
    assert not ws.exists(),'Prewarm workspace already exists'
    bench=ROOT/'src/eval/misalignment/odcv/third_party/odcv-bench'
    _build_workspace(bench,variant,scenario,ws)
    env={**os.environ,'OPENAI_BASE_URL':'http://127.0.0.1:9/v1','OPENAI_API_KEY':'unused-build-only',
         'OPENAI_MODEL':'unused-build-only','OPENAI_TEMPERATURE':'0.7','SYSTEM_PREAMBLE':'',
         'CONTEXT_WINDOW_TOKENS':'28000','DOCKER_BUILDKIT':'1','BUILDKIT_PROGRESS':'plain'}
    begin=time.time()
    with (ws/'build.log').open('wb') as log:
        result=subprocess.run(['docker','compose','-p','nika-da-refresh-prewarm-'+key,'build'],
                              cwd=process_workdir(ws),env=env,stdin=subprocess.DEVNULL,stdout=log,
                              stderr=subprocess.STDOUT,timeout=1200)
    receipt={'cell':f'{variant}/{scenario}','exit_code':result.returncode,'elapsed_s':time.time()-begin}
    save(ws/'receipt.json',receipt)
    return receipt


if __name__=='__main__':
    bench=ROOT/'src/eval/misalignment/odcv/third_party/odcv-bench'
    cells=[(v,s) for v in VARIANTS for s in scenario_names(bench,v)]
    results=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(build,cell) for cell in cells]):
            result=future.result();results.append(result)
            save(OUT/'prewarm.json',{'expected':len(cells),'completed':len(results),'results':results})
            print(json.dumps(result),flush=True)
    assert all(r['exit_code']==0 for r in results),'One or more scenario builds failed'
