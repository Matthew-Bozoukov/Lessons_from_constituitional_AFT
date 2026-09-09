# ABOUTME: Exercises real training-output archives and corruption/missing-artifact teardown gates offline.
# ABOUTME: No GPU rental or network access; fixtures include full adapter provenance and partial checkpoints.
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scratch.nonmoral.result_backup import may_terminate_training, pack_script, verify_archive


def fixture(tmp_path, *, weights=True):
    run = tmp_path / 'output/train/run1'
    adapter = run / 'adapter'
    adapter.mkdir(parents=True)
    identity = {'repo': 'test/data', 'revision': 'a'*40}
    meta = {'dataset': identity, 'base_model_revision': 'b'*40}
    (run / 'run_meta.json').write_text(json.dumps(meta))
    for name, content in {'adapter_config.json':'{}', 'training_meta.json':json.dumps(meta),
                          'train_config.yaml':'model: test', 'tokenizer_config.json':'{}',
                          'tokenizer.json':'{}'}.items():
        (adapter / name).write_text(content)
    if weights:
        (adapter / 'adapter_model.safetensors').write_bytes(b'fixture weights\x00\xff')
    logs = tmp_path / 'output/nonmoral-paired-supervision'
    logs.mkdir()
    (logs / 'arm_0.log').write_text('complete training log')
    (run / 'checkpoint-1').mkdir()
    (run / 'checkpoint-1/optimizer.pt').write_bytes(b'resume state')
    # Fixed output roots must never include credentials or weight-download caches.
    (tmp_path / '.env').write_text('SECRET=never include')
    return [{'data_repo':identity['repo'], 'data_revision':identity['revision'],
             'base_model_revision':meta['base_model_revision']}]


def pack(tmp_path):
    result = subprocess.run([sys.executable, '-c', pack_script(str(tmp_path))],
                            capture_output=True, text=True, check=True)
    manifest = json.loads(result.stdout)
    return Path(manifest['path']), manifest


def test_full_backup_retains_logs_checkpoint_and_required_adapter(tmp_path):
    arms = fixture(tmp_path)
    path, manifest = pack(tmp_path)
    result = verify_archive(path, manifest, arms)
    assert result['verified_completed_arms'] == 1
    assert result['files'] == 9  # Six adapter files, run metadata, full log, checkpoint.
    assert may_terminate_training({'training_started':True, 'local_backup':result})


def test_transfer_corruption_blocks_teardown(tmp_path):
    arms = fixture(tmp_path)
    path, manifest = pack(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[len(raw)//2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='size/hash'):
        verify_archive(path, manifest, arms)
    assert not may_terminate_training({'training_started':True, 'backup_error':'corrupted'})


def test_successful_copy_without_completed_weights_is_not_enough(tmp_path):
    arms = fixture(tmp_path, weights=False)
    path, manifest = pack(tmp_path)
    with pytest.raises(ValueError, match='missing local weights'):
        verify_archive(path, manifest, arms)
    # A failed run can still preserve everything it produced, explicitly partial.
    result = verify_archive(path, manifest, [])
    assert result['verified_completed_arms'] == 0
    assert not may_terminate_training({'training_started':True})
    assert may_terminate_training({'training_started':False})


def test_wrong_dataset_revision_does_not_validate_completed_arm(tmp_path):
    arms = fixture(tmp_path)
    path, manifest = pack(tmp_path)
    arms[0]['data_revision'] = 'c'*40
    with pytest.raises(ValueError, match='missing local weights'):
        verify_archive(path, manifest, arms)


@pytest.mark.parametrize('transfer_ok', [True, False])
def test_real_driver_orders_backup_before_teardown_and_blocks_failed_fetch(tmp_path, monkeypatch, transfer_ok):
    from scratch.nonmoral import train_pair as driver
    events, clock = [], [1000.0]
    monkeypatch.setattr(driver, 'load_dotenv', lambda *a: None)
    monkeypatch.setattr(driver, 'hf_org', lambda: 'dougalldeepmind')
    monkeypatch.setattr(driver, 'hf_api', lambda: SimpleNamespace(
        dataset_info=lambda repo, revision: SimpleNamespace(sha=revision, private=False)))
    monkeypatch.setattr(driver.time, 'time', lambda: clock[0])
    monkeypatch.setattr(driver.time, 'sleep', lambda seconds: clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(driver, 'MAX_LIFETIME_S', 120)
    monkeypatch.setattr(driver, 'RECOVERY_RESERVE_S', 60)
    monkeypatch.setattr(driver.runpod, 'start_watchdog', lambda *a: object())
    monkeypatch.setattr(driver.runpod, 'call', lambda *a: {'costPerHr':1})
    monkeypatch.setattr(driver.runpod, 'wait_bootstrapped', lambda *a,**kw: True)
    monkeypatch.setattr(driver.runpod, 'active_pods', lambda: [])
    monkeypatch.setattr(driver.runpod, 'terminate', lambda pod: events.append('terminate') or True)
    def up(*args, **kwargs):
        kwargs['on_provisioned']('owned-test-pod')
        return 'host: test@host:22\n'
    monkeypatch.setattr(driver.runpod, 'up', up)
    class Remote:
        def __init__(self, *a, **kw):
            pass
        def _ssh(self, command, **kwargs):
            if 'torch.cuda' in command:
                return 'H200 H200'
            if 'print(json.dumps(r))' in command:
                return json.dumps({'complete':True,'metadata':{},'arms':[
                    {'index':i,'bytes':100,'tail':'done','exit':0} for i in range(2)]})
            return ''
    monkeypatch.setattr(driver, 'SshExec', Remote)
    def fetch(*args, **kwargs):
        events.append('backup')
        if not transfer_ok:
            raise OSError('simulated interrupted download')
        return {'verified':True}
    monkeypatch.setattr(driver, 'fetch_training_outputs', fetch)
    monkeypatch.setitem(sys.modules,'account_snapshot',SimpleNamespace(snapshot=lambda: {}))
    plan = {'approved_for_training':True,'base_model_revision':'b'*40,
            'arms':[{'data_repo':'test/data'+str(i),'data_revision':'a'*40} for i in range(2)]}
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    driver.run(plan_path,tmp_path/'run')
    state = json.loads((tmp_path/'run/status.json').read_text())
    if transfer_ok:
        assert events == ['backup','terminate']
        assert state['local_backup']['verified'] and state['terminated']
    else:
        assert events and set(events) == {'backup'}
        assert state['phase'] == 'artifact_recovery_required' and not state['terminated']
        assert clock[0] >= 1120  # Owner stays alive through the emergency deadline.
