# ABOUTME: Offline contracts for the broader LoRA mode of the existing overnight eval owner.
# ABOUTME: Checks frozen budgets/protocol, watchdog ordering and verified-log teardown without rentals.
import hashlib
import io
import json
import shlex
from pathlib import Path
import tarfile
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from scratch.nonmoral import overnight_baseline as owner


@pytest.fixture
def plan_file(tmp_path):
    config = tmp_path/'common.yaml'
    cfg = OmegaConf.load(owner.ROOT/'scratch/nonmoral/odcv-paired.yaml')
    cfg.passes = 3
    OmegaConf.save(cfg, config)
    plan = dict(target='dougalldeepmind/2026-09-09-qwen36-lora-nonmoral-broader-7-mix',
                target_revision='a'*40, base_model='Qwen/Qwen3.6-27B',
                base_revision=owner.BASE_REVISION, output_dir=str(tmp_path/'new_eval'),
                run_name='odcv-broader-offline-test', eval_output_root='C:/nm-eval', eval_config=str(config),
                eval_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                expected_cells=80, passes=3, gpu_cap_usd=15, backup_reserve_usd=2,
                judge_cap_usd=5, max_gpu_hourly_usd=4.5, storage_hourly_reserve_usd=0.1)
    path = tmp_path/'plan.yaml'
    OmegaConf.save(OmegaConf.create(plan), path)
    return path


def test_plan_contract_rejects_drift_and_reuse(plan_file):
    plan = owner.load_plan(plan_file)
    cfg = Path(plan['eval_config'])
    cfg.write_text(cfg.read_text()+'\n# changed\n')
    with pytest.raises(ValueError, match='SHA256'):
        owner.load_plan(plan_file)
    plan['eval_config_sha256'] = hashlib.sha256(cfg.read_bytes()).hexdigest()
    plan['judge_cap_usd'] = 6
    OmegaConf.save(OmegaConf.create(plan), plan_file)
    with pytest.raises(ValueError, match='allocation'):
        owner.load_plan(plan_file)
    plan['judge_cap_usd'] = 5
    OmegaConf.save(OmegaConf.create(plan), plan_file)
    out = Path(plan['output_dir'])
    out.mkdir()
    (out/'prior.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh'):
        owner.load_plan(plan_file)


def test_plan_rejects_long_or_relative_eval_paths(plan_file):
    plan = owner.load_plan(plan_file)
    for path in ('output/eval', 'C:/a/deeply/nested/evaluation/output/directory'):
        plan['eval_output_root'] = path
        OmegaConf.save(OmegaConf.create(plan), plan_file)
        with pytest.raises(ValueError, match='short absolute Windows'):
            owner.load_plan(plan_file)


def test_recovery_retries_then_succeeds_within_same_deadline(tmp_path, monkeypatch):
    clock = [100.0]
    state = dict(rented_at_unix=100, max_lifetime_s=120, host='host', pod_id='owned',
                 evaluation_started=True)
    calls = []
    def fetch(*args, **kwargs):
        calls.append(kwargs['timeout'])
        if len(calls) == 1:
            raise RuntimeError('transient SSH disconnect')
        return {'verified': True}
    monkeypatch.setattr(owner, 'fetch_eval_logs', fetch)
    def sleep(seconds):
        clock[0] += seconds
    assert owner.recover_eval_logs(state,'',tmp_path,lambda **kw: state.update(kw),
                                   now=lambda: clock[0],sleep=sleep)
    assert calls == [110.0, 95.0]
    assert state['recovery_attempts'] == 2 and state['ordinary_teardown_blocked'] is False
    assert state['recovery_deadline_unix'] == 220 and clock[0] == 115


@pytest.mark.parametrize('backup_succeeds', [True, False])
def test_owned_dispatch_arms_first_and_only_closes_after_backup(plan_file, monkeypatch, backup_succeeds):
    plan = owner.load_plan(plan_file)
    events = []
    clock = [1000.0]
    original_recover = owner.recover_eval_logs
    def recover(state, identity, out, save):
        clock[0] = state['rented_at_unix']
        def sleep(seconds):
            clock[0] += seconds
        return original_recover(state, identity, out, save, now=lambda: clock[0], sleep=sleep)
    monkeypatch.setattr(owner, 'recover_eval_logs', recover)
    monkeypatch.setattr(owner, 'docker_preflight', lambda: None)
    monkeypatch.setattr(owner, 'require_network_capacity', lambda *a, **k: None)
    monkeypatch.setattr(owner, 'require_lf_shell_scripts', lambda *a: None)
    monkeypatch.setattr(owner, 'hf_org', lambda: 'dougalldeepmind')
    monkeypatch.setattr(owner.runpod, 'default_keypair', lambda: ('public', 'private'))
    monkeypatch.setattr(owner, 'resolve_target', lambda t: SimpleNamespace(
        hf_path=t, revision=plan['target_revision'], base_model=plan['base_model'],
        base_revision=plan['base_revision'], adapter=True, mode='think'))
    monkeypatch.setattr(owner.runpod, 'start_watchdog', lambda *a: (
        events.append('watchdog') or SimpleNamespace(pid=42, terminate=lambda: events.append('cancel'))))
    monkeypatch.setattr(owner.runpod, 'call', lambda *a: {'costPerHr': 3.49})
    def provision(*a, **kw):
        kw['on_provisioned']('owned')
        events.append('provision_return')
        return SimpleNamespace(id='owned', host='root@host:22')
    monkeypatch.setattr(owner.runpod, 'provision_eval_pod', provision)
    monkeypatch.setattr(owner.runpod, 'wait_bootstrapped', lambda *a, **kw: events.append('bootstrap') or True)
    def dispatch(path, config, host, identity, timeout):
        events.append('eval')
        cfg = OmegaConf.load(config)
        assert cfg.passes == 3 and cfg.judge_budget.cap_usd == 5
        assert cfg.output_root == 'C:/nm-eval'
        assert timeout < (15-2)/4.6*3600
    monkeypatch.setattr(owner, 'dispatch_frozen', dispatch)
    def backup(*a, **kw):
        events.append('backup')
        if not backup_succeeds:
            raise RuntimeError('hash mismatch')
        return {'verified': True}
    monkeypatch.setattr(owner, 'fetch_eval_logs', backup)
    monkeypatch.setattr(owner.runpod, 'terminate', lambda p: events.append('terminate') or True)
    monkeypatch.setattr(owner.runpod, 'active_pods', lambda: [])
    if backup_succeeds:
        owner.main(plan_path=plan_file)
        assert events == ['watchdog', 'provision_return', 'bootstrap', 'eval', 'backup', 'terminate', 'cancel']
    else:
        with pytest.raises(RuntimeError, match='backup failed'):
            owner.main(plan_path=plan_file)
        assert events[:4] == ['watchdog', 'provision_return', 'bootstrap', 'eval']
        assert len(events[4:]) > 1 and set(events[4:]) == {'backup'}
    status = json.loads((Path(plan['output_dir'])/'broader_eval_status.json').read_text())
    assert status['total_allocation_usd'] == 20
    assert status['max_lifetime_s']*4.6/3600 < 15
    if not backup_succeeds:
        assert clock[0] >= status['rented_at_unix']+status['max_lifetime_s']
        assert status['emergency_watchdog_handoff'] is True
    with pytest.raises(ValueError, match='fresh'):
        owner.main(plan_path=plan_file)


def test_remote_archive_hash_and_allowlist(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for name in owner.LOG_FILES:
            data = b'ordinary log output\n'
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    payload = archive.getvalue()
    manifest = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    def ssh(self, command, **kw):
        if command.startswith('pkill '):
            return ''
        if command.startswith('pgrep '):
            return 'down'
        script = shlex.split(command)[2]
        assert "'boot.log', 'output/serve/vllm.log'" in script
        assert '.env' not in script and 'run.sh' not in script
        return json.dumps(manifest)
    monkeypatch.setattr(owner.SshExec, '_ssh', ssh)
    monkeypatch.setattr(owner, 'ssh_argv', lambda *a: (['ssh'], 'host'))
    def transfer(*a, **kw):
        kw['stdout'].write(payload)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(owner.subprocess, 'run', transfer)
    receipt = owner.fetch_eval_logs('host', '', tmp_path, True)
    assert receipt['verified'] and set(receipt['file_sha256']) == set(owner.LOG_FILES)
    manifest['sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='verification failed'):
        owner.fetch_eval_logs('host', '', tmp_path, True)
    first_partial = next(tmp_path.glob('remote_logs_*.tar.partial'))
    with pytest.raises(RuntimeError, match='verification failed'):
        owner.fetch_eval_logs('host', '', tmp_path, True)
    assert len(list(tmp_path.glob('remote_logs_*.tar.partial'))) == 2
    assert first_partial.read_bytes() == payload


def test_log_fetch_consumes_one_deadline_across_remote_operations(tmp_path, monkeypatch):
    clock = [0.0]
    observed = []
    monkeypatch.setattr(owner.time, 'monotonic', lambda: clock[0])
    def ssh(self, command, **kw):
        observed.append(kw['timeout'])
        if command.startswith('pkill '):
            clock[0] += 3
            return ''
        if command.startswith('pgrep '):
            clock[0] += 4
            return 'down'
        clock[0] += 2
        return json.dumps(dict(bytes=1, sha256='0'*64))
    monkeypatch.setattr(owner.SshExec, '_ssh', ssh)
    monkeypatch.setattr(owner, 'ssh_argv', lambda *a: (['ssh'], 'host'))
    def transfer(*a, **kw):
        observed.append(kw['timeout'])
        kw['stdout'].write(b'x')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(owner.subprocess, 'run', transfer)
    with pytest.raises(RuntimeError, match='verification failed'):
        owner.fetch_eval_logs('host','',tmp_path,True,timeout=10)
    assert observed == [10, 7, 3, 1]
