# ABOUTME: Destroys the explicitly requested, receipted Vast CPU and verifies absence.
# ABOUTME: Preserves receipts and archives the registry before marking it empty.
import os,json
from pathlib import Path
from datetime import datetime,timezone
from vastai import VastAI
c=VastAI(api_key=os.environ['VAST_API_KEY'].strip())
rp=Path.home()/'.lasr/swebench/cpu.json'
r=json.loads(rp.read_text());receipt=json.loads(Path(r['receipt']).read_text())
assert r['instance_id']==receipt['instance_id']==55053931
before=c.show_instances();matches=[x for x in before if x['id']==55053931]
assert len(matches)==1 and matches[0]['label']==receipt['label']
audit={'checked_at':datetime.now(timezone.utc).isoformat(),'instance_id':55053931,'label':receipt['label'],'before_status':matches[0]['actual_status'],'authorization':'User explicitly requested teardown on 2026-10-12'}
out=Path('output/2026-10-12-vast-cpu-teardown.json');out.write_text(json.dumps(audit,indent=2))
audit['destroy_response']=c.destroy_instance(id=55053931)
after=c.show_instances();audit['remaining_instance_ids']=[x['id'] for x in after];audit['volumes']=c.show_volumes()
audit['provider_absent']=55053931 not in audit['remaining_instance_ids']
out.write_text(json.dumps(audit,indent=2))
assert audit['provider_absent'],'Provider still lists CPU; registry retained'
assert json.loads(rp.read_text())==r,'Registry changed concurrently'
rp.with_name('cpu-before-destroy-55053931-20261012.json').write_text(json.dumps(r,indent=2))
tmp=rp.with_suffix('.tmp');tmp.write_text('{"state":"empty"}\n');tmp.replace(rp)
print(json.dumps(audit,indent=2))
