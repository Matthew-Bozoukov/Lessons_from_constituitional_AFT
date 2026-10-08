# ABOUTME: Predeclare ten base-only local SWE tasks and cache their immutable Docker images.
# ABOUTME: Selection uses only repository/task identifiers, never prior outcomes or timings.
import json
from pathlib import Path
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf

ROOT = Path('output/gptoss_shared_smoke')
rows = json.loads((ROOT/'cache/metadata/swebench_lite_test.json').read_text())
images = json.loads((ROOT/'cache/metadata/images.json').read_text())
eligible = sorted(x['instance_id'] for x in rows if x['repo'] != 'psf/requests')
selected = sorted(random.Random(0).sample(eligible, 10))
selection = dict(ids=selected, rule='Random(0).sample(sorted(non-requests instance IDs),10); no outcome selection',
                 excluded='requests requires external HTTPBin; no local fixture for this smoke')
path = ROOT/'selection.json'
if path.exists():
    assert json.loads(path.read_text()) == selection
else:
    path.write_text(json.dumps(selection, indent=2))
cfg = OmegaConf.load('scratch/gptoss_swe/pilot.yaml')
cfg.campaign = 'gptoss-shared-interface-smoke-20261008'
cfg.run_name = 'gptoss120b-base-shared-smoke'
cfg.output_root = '/work/output/gptoss_shared_smoke/swe'
cfg.cached_campaign = '/work/output/gptoss_shared_smoke/cache'
cfg.instance_ids = selected
cfg.subset.n = 10
cfg.workers = 2
cfg.tinker.budget_usd = 15
cfg.tinker.budget_ledger = '/work/output/gptoss_shared_smoke/inference-budget.json'
cfg.worker.tool_concurrency = 2
cfg.worker.min_available_memory_gib = 2
OmegaConf.save(cfg, ROOT/'swe.yaml')

def pull(iid):
    image = images[iid]
    with (ROOT/(iid+'-pull.log')).open('w') as log:
        subprocess.run(['docker','pull',image['digest']], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['docker','tag',image['digest'],image['name']], stdout=log, stderr=subprocess.STDOUT, check=True)
        actual = json.loads(subprocess.check_output(['docker','image','inspect',image['name']]))[0]
        # Docker Desktop's containerd image store exposes the manifest digest as
        # Id; classic Docker exposed the config digest. The immutable RepoDigest
        # below is identical on both and is the content-addressed pull authority.
        assert actual['Id'] in (image['id'], image['digest'].split('@')[1]), (iid,actual['Id'])
        assert image['digest'] in actual['RepoDigests']
        subprocess.run(['docker','run','--rm','--network','none','--entrypoint','/bin/bash',image['name'],
                        '-lc','cd /testbed && git rev-parse HEAD && test -d /opt/miniconda3'], stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Ready:',iid,flush=True)
    return dict(instance_id=iid, **image)

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(pull,selected))
    (ROOT/'images-verified.json').write_text(json.dumps(verified,indent=2))
