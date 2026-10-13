# ABOUTME: Run the official harness with image inventory scoped to this audit's selected instances.
# ABOUTME: Avoid blocked whole-host Docker image enumeration; tests and grading are unchanged.
import json,sys
from pathlib import Path
from swebench.harness import run_evaluation,docker_utils,reporting
from swebench.harness.test_spec.test_spec import make_test_spec
from docker.models.images import ImageCollection

request=json.loads(Path(sys.argv[1]).read_text())['harness']
assert request['cache_level']=='instance' and not request['clean']
rows=json.loads(Path(request['dataset_name']).read_text())
selected=[r for r in rows if r['instance_id'] in request['instance_ids']]
keys=[make_test_spec(r,namespace=request['namespace']).instance_image_key for r in selected]
def scoped_collection(self,*args,**kwargs):
 assert not args and set(kwargs)<= {'all'},'Unexpected inventory query'
 return [self.get(key) for key in keys]
# run_instances also enumerates ImageCollection directly, rather than the
# list_images helper. Scope that bookkeeping in this subprocess as well.
ImageCollection.list=scoped_collection
def scoped_images(client):
 return {tag for key in keys for tag in client.images.get(key).tags}
docker_utils.list_images=scoped_images
run_evaluation.list_images=scoped_images
reporting.list_images=scoped_images
run_evaluation.main(**request)
