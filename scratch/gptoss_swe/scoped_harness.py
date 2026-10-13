# ABOUTME: Bound official grading image bookkeeping to the selected cached instance images.
# ABOUTME: Preserve the pinned HTTPBin fixture, test commands, patches and official scoring.
import argparse
import json
from pathlib import Path
import runpy
import sys


def install_scope(request):
    from docker.models.images import ImageCollection
    from swebench.harness import docker_utils, reporting, run_evaluation
    from swebench.harness.test_spec.test_spec import make_test_spec
    assert request['cache_level'] == 'instance' and not request['clean']
    rows = json.loads(Path(request['dataset_name']).read_text())
    wanted = set(request['instance_ids'])
    selected = [r for r in rows if r['instance_id'] in wanted]
    assert len(selected) == len(wanted), 'Selected dataset rows missing or duplicated'
    keys = [make_test_spec(r, namespace=request['namespace']).instance_image_key for r in selected]

    def collection(images, *args, **kwargs):
        assert not args and set(kwargs) <= {'all'}, 'Unexpected image enumeration'
        return [images.get(key) for key in keys]

    def tags(client):
        return {tag for key in keys for tag in client.images.get(key).tags}

    ImageCollection.list = collection
    docker_utils.list_images = tags
    reporting.list_images = tags
    run_evaluation.list_images = tags
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    keys = install_scope(request['harness'])
    Path('image-inventory-scope.json').write_text(json.dumps(keys, indent=2))
    if request['fixture'] is not None:
        sys.argv = ['swebench_local_httpbin.py', '--request', str(args.request)]
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'swebench_local_httpbin.py'), run_name='__main__')
    else:
        from swebench.harness.run_evaluation import main as grade
        grade(**request['harness'])


if __name__ == '__main__':
    main()
