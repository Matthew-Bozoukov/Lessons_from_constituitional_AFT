# ABOUTME: Fetch immutable limit-ending successes and their exact histories for a CPU-only cutoff audit.
# ABOUTME: Preserve original scores and hash-check complete source archives before replaying any commands.
import concurrent.futures,hashlib,json,tarfile
from pathlib import Path
import requests
OUT=Path('output/gptoss_patch_at_65000_20261013');OUT.mkdir(exist_ok=True)
REPO='dougalldeepmind/2026-10-09-gptoss120b-base-control-da15-swe-smoke20';REV='09bc27d232fb52c6a6d46aa54f09c12e6ac4e9b5'
class Reader:
 def __init__(self,raw):self.raw=raw;self.sha=hashlib.sha256()
 def read(self,n=-1):
  b=self.raw.read(n);self.sha.update(b);return b

def fetch(arm):
 prior=json.loads(Path(f'output/gptoss_mxfp4_cost_audit_20261013/context64/{arm}.json').read_text())
 selected=[r for r in prior['tasks'] if r['resolved'] and r['ending']=='LimitsExceeded']; wanted={(r['task'],r['attempt']) for r in selected}
 for r in selected:(OUT/arm/r['task']).mkdir(parents=True,exist_ok=True)
 source=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/rollouts/{arm}.tar.gz'
 with requests.get(source,stream=True,timeout=(30,120)) as res:
  res.raise_for_status();reader=Reader(res.raw)
  with tarfile.open(fileobj=reader,mode='r|gz') as tf:
   for e in tf:
    p=e.name.split('/')
    if e.name==f'{arm}/swe/metadata/state.json':
     state=json.load(tf.extractfile(e));(OUT/arm/'original-state.json').write_text(json.dumps(state,indent=2))
    if len(p)<6 or p[1:3]!=['swe','rollouts'] or tuple(p[3:5]) not in wanted:continue
    if p[-1] not in ('checkpoint.traj.json','request.json','scaffold.json') and not p[-1].endswith('.traj.json'):continue
    if 'http' in p or 'tokenization' in p:continue
    b=tf.extractfile(e).read();(OUT/arm/p[3]/p[-1]).write_bytes(b)
  while reader.read(1024*1024):pass
 assert reader.sha.hexdigest()==prior['archive_sha256']
 for r in selected:
  r['prediction']=state['tasks'][r['task']]['attempts'][-1]['prediction']
  r['source']=dict(repo=REPO,revision=REV,archive_sha256=reader.sha.hexdigest())
 (OUT/arm/'selected.json').write_text(json.dumps(selected,indent=2))
 print(json.dumps(dict(arm=arm,selected=[r['task'] for r in selected])),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,['base','control','da15']))
