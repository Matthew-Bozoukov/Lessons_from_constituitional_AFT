# ABOUTME: Starts one isolated dual-H200 campaign owner as a hidden persistent process.
# ABOUTME: Run: uv run python scratch/da_supervision/start_dual.py ARM.
import json,subprocess,sys
from pathlib import Path
root=Path.cwd();base=root/'output/da_supervision/2026-09-16';arm=sys.argv[1];assert arm in ('cot','answer','empty')
attempt=int(sys.argv[2]) if len(sys.argv)>2 else 1
out=base/'runs'/(arm+'-dual'+(f'-attempt{attempt}' if attempt>1 else ''));out.mkdir(exist_ok=True);assert not (out/'launcher.json').exists()
plan=base/'plan_dual.json'
if len(sys.argv)>3:
 cfg=json.loads(plan.read_text());cfg['pod']['countries']=sys.argv[3]
 if len(sys.argv)>4:
  assert sys.argv[4] in ('SECURE','COMMUNITY');cfg['pod']['cloud']=sys.argv[4]
 plan=out/'placement_plan.json';plan.write_text(json.dumps(cfg,indent=2)+'\n')
with (out/'owner.log').open('w',encoding='utf-8') as log:
 p=subprocess.Popen([sys.executable,'-u','scratch/da_supervision/owner.py',str(plan),arm,'--out',str(out)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
receipt={'pid':p.pid,'arm':arm,'out':str(out)};(out/'launcher.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
