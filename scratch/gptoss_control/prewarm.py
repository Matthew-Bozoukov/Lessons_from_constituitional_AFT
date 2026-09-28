# ABOUTME: Build the exact solo ODCV scenario images while Tinker trains, without running agents.
# ABOUTME: Uses two workers, owned workspace/image names, and no pruning or provider credentials.
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.misalignment.odcv.odcv import VARIANTS, scenario_names
from src.eval.misalignment.odcv.odcv_rollout import _build_workspace, _compose

os.chdir(ROOT)
bench=ROOT/'src/eval/misalignment/odcv/third_party/odcv-bench'
out=ROOT/'output/gptoss_control/prewarm'
out.mkdir(parents=True,exist_ok=True)
# Build-only placeholders: Compose requires interpolation even though no service starts.
env={**os.environ,'OPENAI_BASE_URL':'http://host.docker.internal:18321/v1',
    'OPENAI_API_KEY':'unused-build-placeholder','OPENAI_MODEL':'openai/gpt-oss-120b',
    'OPENAI_TEMPERATURE':'0.7','SYSTEM_PREAMBLE':'','CONTEXT_WINDOW_TOKENS':'28000',
    'REQUIRE_EXACT_TOKEN_COUNT':'1','TEAM_SPEC_FILE':'','DOCKER_BUILDKIT':'1','BUILDKIT_PROGRESS':'plain'}
for name in list(env):
    if ('KEY' in name or 'TOKEN' in name) and name not in {'OPENAI_API_KEY','CONTEXT_WINDOW_TOKENS','REQUIRE_EXACT_TOKEN_COUNT'}:
        env.pop(name)
def build(job):
    variant,scenario=job
    ident=hashlib.sha256(f'{variant}/{scenario}'.encode()).hexdigest()[:12]
    ws=out/ident
    if not ws.resolve().is_relative_to(out.resolve()): raise ValueError('Workspace escaped owned root')
    _build_workspace(bench,variant,scenario,ws)
    result=_compose('gptoss-control-prewarm-'+ident,ws,env,['build'],timeout=1200)
    (ws/'build.log').write_text(result.stdout or '',encoding='utf-8')
    return {'variant':variant,'scenario':scenario,'workspace':ident,'exit_code':result.returncode}
jobs=[(v,s) for v in VARIANTS for s in scenario_names(bench,v)]
records=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    for future in as_completed([pool.submit(build,j) for j in jobs]):
        record=future.result()
        records.append(record)
        (out/'manifest.json').write_text(json.dumps({'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'builds':records},indent=2),encoding='utf-8')
        print(f'prewarm {len(records)}/{len(jobs)} exit={record["exit_code"]} {record["variant"]}/{record["scenario"]}',flush=True)
if any(r['exit_code'] for r in records): raise RuntimeError('Some ODCV image builds failed; inspect owned logs')
