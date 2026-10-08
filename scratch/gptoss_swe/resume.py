# ABOUTME: Validate a fully drained SWE run before carrying its paid outcomes into a larger worker pool.
# ABOUTME: No live, invalid, mismatched, or already retried task may be silently resumed here.
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil


def validate_snapshot(state, manifest, config, target):
    assert manifest['target'] == target, 'Resume target changed'
    old = manifest['config']
    for key in ('protocol', 'dataset_revision', 'dataset_sha256', 'sampling', 'worker', 'tinker',
                'qualification_instance', 'cached_campaign', 'campaign'):
        assert old[key] == config[key], 'Resume recipe changed: '+key
    assert state['halt'] is None and state['deadline'] is not None, 'Run was not deliberately drained'
    assert set(state['tasks']) == set(manifest['selected_ids']), 'Resume task set changed'
    assert len(state['tasks']) == 300
    for iid, task in state['tasks'].items():
        assert task['status'] in ('valid', 'pending'), 'Task still live or invalid: '+iid
        if task['status'] == 'pending':
            assert not task['attempts'], 'Pending task has previous attempts: '+iid
        else:
            assert len(task['attempts']) == 1 and task['attempts'][0]['valid'], iid
            assert task['attempts'][0]['prediction']['instance_id'] == iid
    result = deepcopy(state)
    result['deadline'] = None  # Restore the original unlimited campaign lifetime after admission-only drain.
    return result


def carry_forward(source, destination, config, target):
    source, destination = Path(source), Path(destination)
    state = json.loads((source/'metadata/state.json').read_text())
    manifest = json.loads((source/'metadata/manifest.json').read_text())
    result = validate_snapshot(state, manifest, config, target)
    proof = json.loads((source/'metadata/live-qualification.json').read_text())
    assert proof['passed'] is True and proof['n_graded'] == 1
    records = {}
    for folder in ('rollouts', 'results'):
        for path in (source/folder).rglob('*'):
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            dest = destination/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert not dest.exists(), 'Refuse overwrite: '+str(dest)
            shutil.copyfile(path, dest)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
            records[rel.as_posix()] = digest
    shutil.copytree(source/'metadata', destination/'metadata/prior-run',
                    ignore=shutil.ignore_patterns('token-slots'))
    shutil.copyfile(source/'run_meta.json', destination/'metadata/prior-run/launch-run_meta.json')
    shutil.copyfile(source/'metadata/live-qualification.json', destination/'metadata/live-qualification.json')
    receipt = dict(source=str(source), preserved_valid=sum(t['status']=='valid' for t in state['tasks'].values()),
                   copied_sha256=records, original_deadline=state['deadline'], workers=config['workers'])
    return result, receipt
