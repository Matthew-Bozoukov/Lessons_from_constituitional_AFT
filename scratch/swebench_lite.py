# ABOUTME: Persistent, bounded full SWE-bench Lite campaign on a prepared CPU host and RunPod replicas.
# ABOUTME: Preparation and grading rent nothing; only explicit run/resume can allocate inference GPUs.
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from dotenv import load_dotenv
from omegaconf import OmegaConf
import psutil

from src.infra import runpod
from src.infra.endpoints.vllm import resolve_target
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_run_dir
from src.naming import eval_name, today
from scratch.swebench_lite_state import State, atomic, digest, lock, read
from scratch.swebench_lite_worker import stop_process

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / 'src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python'


def account():
    return runpod.graphql('query { myself { clientBalance currentSpendPerHr } }')['myself']


def preflight(cfg):
    ready = Path(cfg.readiness)
    proof = read(ready.parent / 'cpu-readiness-backup-verified.json')
    result = read(ready / 'results/readiness.json')
    assert digest(ready / 'results/readiness.json') == proof['readiness_sha256']
    assert result['status'] == 'ready' and result['gold']['passed'] and result['no_fix']['passed']
    assert result['revision'] == cfg.dataset_revision
    assert digest(ready / 'metadata/swebench_lite_test.json') == cfg.dataset_sha256
    assert digest(REPO / 'scratch/swebench_local_httpbin.py') == result['gold']['wrapper_sha256']
    rows = read(ready / 'metadata/swebench_lite_test.json')
    assert len(rows) == len({r['instance_id'] for r in rows}) == 300
    images = read(ready / 'metadata/images.json')
    assert set(images) == {r['instance_id'] for r in rows}
    inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', *[i['digest'] for i in images.values()]]))
    assert len(inspected) == 300
    assert [i['Id'] for i in inspected] == [i['id'] for i in images.values()]
    # The unmodified grader uses cached latest tags; ensure these tags still denote
    # precisely the digest the agent sees, and do not pull while a campaign runs.
    tags = json.loads(subprocess.check_output(['docker', 'image', 'inspect', *[i['name'] for i in images.values()]]))
    assert [i['Id'] for i in tags] == [i['id'] for i in images.values()]
    assert shutil.disk_usage('/var/lib/docker').free / 2**30 >= cfg.min_free_gib
    assert psutil.cpu_count() >= 32 and psutil.virtual_memory().total >= 115 * 2**30
    receipt = read(cfg.receipt)
    available = datetime.fromisoformat(receipt['stop_at']).timestamp() - time.time()
    assert available > cfg.campaign_seconds + cfg.cpu_finish_reserve_seconds, 'CPU expiry too close; do not rent GPUs'
    assert Path(cfg.ssh_key).is_file() and Path(cfg.ssh_key + '.pub').is_file()
    assert os.environ.get('USER_PREFIX') and os.environ.get('HF_ORG')
    spec = resolve_target(cfg.target, revision=cfg.target_revision)
    assert spec.base_model == cfg.base and spec.base_revision == cfg.base_revision and spec.mode == cfg.mode
    assert spec.adapter and spec.lora_rank == 64
    runpod.validate_scheduled_provision()
    quote = runpod.gpu_price(cfg.gpu)
    assert quote and quote <= cfg.max_hourly_usd, 'GPU quote missing or exceeds configured hourly ceiling'
    balance = account()
    # The inference fixture must still pass health checks, without repeating gold runs.
    from scratch.swebench_cpu_prepare import ensure_fixture
    fixture = ensure_fixture(OmegaConf.load(REPO / 'scratch/swebench_cpu.yaml'), receipt)
    assert fixture == read(ready / 'metadata/httpbin_fixture.json')
    return {'status': 'ready_to_launch', 'tasks': 300, 'model': cfg.target, 'revision': cfg.target_revision,
            'replicas': cfg.replicas, 'workers': cfg.replicas * cfg.workers_per_replica,
            'gpu_quote_usd_hour': quote, 'balance': balance, 'cpu_stop_at': receipt['stop_at'],
            'recommended_budget_usd': cfg.recommended_budget_usd,
            'paid_gpu_test': 'pending; first launch begins with one calibration replica'}


def sources():
    paths = [p for p in (REPO / 'scratch').glob('swebench_lite*.py')]
    paths += [REPO / p for p in ['src/infra/runpod.py', 'src/infra/endpoints/vllm.py', 'src/eval/run_eval.py',
                               'src/model_profile.py', 'configs/models/qwen36.yaml', 'src/naming.py',
                               'src/infra/huggingface.py', 'src/utils.py', 'pyproject.toml',
                               'src/eval/capabilities/swebench_mini/agent.py', 'src/eval/capabilities/stats.py',
                               'scratch/swebench_local_httpbin.py', 'scratch/swebench_cpu_env/uv.lock',
                               'src/eval/capabilities/swebench_mini/envs/agent/uv.lock',
                               'src/eval/capabilities/swebench_mini/envs/harness/uv.lock']]
    return {str(p.relative_to(REPO)): digest(p) for p in paths}


def initialize(cfg, config_path, budget):
    root = Path(cfg.root)
    if (root / 'metadata/manifest.json').exists():
        raise RuntimeError('Campaign exists; use resume, never overwrite it')
    spec = resolve_target(cfg.target, revision=cfg.target_revision)
    repo = hf_repo_id(eval_name('swebench_mini', spec.model_key))
    assert not hf_api().repo_exists(repo, repo_type='dataset'), 'HF destination exists; refusing to overwrite another campaign'
    campaign = uuid.uuid4().hex
    rows = read(Path(cfg.readiness) / 'metadata/swebench_lite_test.json')
    manifest = {'campaign': campaign, 'repo': repo, 'created': datetime.now(timezone.utc).isoformat(),
                'date': today(), 'model_key': spec.model_key, 'config': OmegaConf.to_container(cfg),
                'source_hashes': sources(), 'budget_usd': budget, 'dataset_tasks': 300,
                'protocol': 'mini-swe-agent 2.2.1; official 250 steps, inert local dollar limit; network none; '
                            'digest-pinned cached images; 2 CPU/4GiB/512 PID agent container caps; infrastructure retries only (max two attempts); '
                            'requests local HTTPBin fixture for grading; full denominator 300',
                'limitations': read(Path(cfg.readiness) / 'results/readiness.json')['benchmark_limitations']}
    deployment = Path('/srv/lasr/lite-deployment.json')
    if deployment.exists():
        manifest['deployment'] = read(deployment)
    for sub in ('metadata', 'rollouts', 'results'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in ('swebench_lite_test.json', 'images.json', 'httpbin_fixture.json', 'httpbin-ca.pem'):
        shutil.copyfile(Path(cfg.readiness) / 'metadata' / name, root / 'metadata' / name)
    shutil.copyfile(config_path, root / 'metadata/config.yaml')
    for name, sha in manifest['source_hashes'].items():
        dest = root / 'metadata/source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dest)
        assert digest(dest) == sha
    atomic(root / 'metadata/manifest.json', manifest)
    atomic(root / 'metadata/state.json', {'tasks': {r['instance_id']: {'status': 'pending', 'attempts': []} for r in rows},
           'pods': [], 'deadline': 0, 'last_upload': 0, 'phase': 'prepared', 'calibrated': False, 'halt': None})
    return manifest


def owned(pod, manifest):
    env = pod.get('env') or {}
    return env.get('LASR_POD_OWNER') == runpod.POD_OWNER and env.get('LASR_CAMPAIGN') == manifest['campaign']


def cleanup_grading(root, manifest):
    tasks = read(Path(root) / 'metadata/state.json')['tasks']
    expected = {'sweb.eval.' + iid.lower() + '.lite_' + manifest['campaign'] for iid in tasks}
    present = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
    names = sorted(set(present) & expected)
    if names:
        subprocess.run(['docker', 'rm', '-f', *names], check=True, timeout=90)


def fence(cfg, manifest):
    # Provider identity, not a name-prefix sweep; teammates' pods are never touched.
    for pod in runpod.active_pods():
        if owned(pod, manifest):
            runpod.teardown(pod['id'])
    assert not any(owned(p, manifest) for p in runpod.active_pods())
    root = str(Path(cfg.root))
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmd = proc.info['cmdline'] or []
        if proc.pid != os.getpid() and ('scratch.swebench_lite_task' in cmd or 'scratch.swebench_lite_worker' in cmd):
            if any(root in a for a in cmd):
                proc.kill()
                proc.wait(timeout=15)
    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=lasr_campaign=' + manifest['campaign']], text=True).split()
    if ids:
        subprocess.run(['docker', 'rm', '-f', *ids], check=True, timeout=90)
    cleanup_grading(cfg.root, manifest)
    State(cfg.root).recover()


def guard(cfg):
    """Independent systemd timer; remains alive when the coordinator's cgroup exits."""
    path = Path(cfg.root) / 'metadata/manifest.json'
    if not path.exists():
        return
    manifest = read(path)
    active = subprocess.run(['systemctl', 'is-active', '--quiet', 'lasr-swebench-lite.service']).returncode == 0
    state = read(Path(cfg.root) / 'metadata/state.json')
    for pod in runpod.active_pods():
        if owned(pod, manifest):
            deadline = float((pod.get('env') or {}).get('LASR_POD_DEADLINE', 0))
            if not active or time.time() >= min(deadline, state['deadline']):
                runpod.teardown(pod['id'])
    if not active:
        try:
            with lock(Path(cfg.root) / '.coordinator.lock', nonblocking=True):
                cleanup_grading(cfg.root, manifest)
                ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                      'label=lasr_campaign=' + manifest['campaign']], text=True).split()
                if ids:
                    subprocess.run(['docker', 'rm', '-f', *ids], check=True, timeout=90)
        except BlockingIOError:
            pass  # An explicit CPU-only grade command owns the lock outside the service.


def publish(cfg):
    root = Path(cfg.root)
    manifest = read(root / 'metadata/manifest.json')
    with lock(root / '.publish.lock', nonblocking=True), tempfile.TemporaryDirectory(prefix='lite-upload-') as temp:
        snapshot = Path(temp)
        # JSON checkpoints are atomically replaced. Logs may be prefixes; completed
        # attempts are immutable. Copy state last, under the lease lock, so it never
        # points to a completed attempt absent from this snapshot.
        with lock(root / '.state.lock'):
            for sub in ('rollouts', 'results', 'metadata'):
                shutil.copytree(root / sub, snapshot / sub, ignore=shutil.ignore_patterns('*.tmp', '__pycache__'))
        repo = manifest['repo']
        api = hf_api()
        if api.repo_exists(repo, repo_type='dataset'):
            old = read(hf_download(repo, 'metadata/manifest.json', repo_type='dataset'))
            assert old['campaign'] == manifest['campaign'], 'HF campaign identity mismatch'
        fields = {'experiment': 'Full 300-task SWE-bench Lite no-DA control; partial until valid coverage and grading finish',
                  'date_generated': manifest['date'], 'constitution': 'none',
                  'source_repo': 'teaching_claude_why_replication; exact sources and SHA256 in metadata/source',
                  'models': cfg.target + '@' + cfg.target_revision + '; base ' + cfg.base + '@' + cfg.base_revision,
                  'generation_config': json.dumps(manifest['config']),
                  'schema': 'rollouts/: every attempt JSON and readable transcript; results/: official reports and score; metadata/: pins, ledger, code',
                  'provenance': manifest['protocol'] + '; ' + manifest['limitations']}
        push_run_dir(snapshot, repo, fields, front_matter={'tags': ['eval-run', 'eval:swebench_mini',
                     'model:' + manifest['model_key'], 'mode:' + cfg.mode, 'swebench-lite']})
        commit = api.dataset_info(repo).sha
        saved = read(hf_download(repo, 'metadata/state.json', repo_type='dataset', revision=commit))
        assert saved == read(snapshot / 'metadata/state.json'), 'HF state round-trip mismatch'
        with State(root).edit() as state:
            state['last_upload'] = time.time()
            state['hf_commit'] = commit
        print('HF checkpoint:', repo, commit, flush=True)


def checkpoint(cfg, config_path, *, required=False):
    try:
        subprocess.run([sys.executable, '-m', 'scratch.swebench_lite', 'publish', '--config', str(config_path)],
                       check=True, timeout=cfg.upload_timeout_seconds)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        print('HF checkpoint failed:', type(exc).__name__, flush=True)
        if required:
            raise
        return False


def reserved_cost(state):
    return sum(p['ceiling_hourly'] * (p.get('ended', p['expires']) - p['created']) / 3600 for p in state['pods'])


def replica(cfg, config_path, slot, allowed_path, expires, manifest):
    state = State(cfg.root)
    name = f"{os.environ['USER_PREFIX']}-swe-lite-{manifest['campaign'][:8]}-{slot}"
    record = {'slot': slot, 'name': name, 'created': time.time(), 'expires': expires,
              'ceiling_hourly': cfg.max_hourly_usd, 'id': None, 'status': 'allocating'}
    with state.edit() as data:
        data['pods'].append(record)
    guard = proc = None
    pod_id = None

    def update(**values):
        with state.edit() as data:
            next(p for p in data['pods'] if p['slot'] == slot).update(values)

    def allocated(identifier):
        nonlocal pod_id, guard
        pod_id = identifier
        update(id=identifier, status='booting')
        guard = runpod.start_watchdog(identifier, max(1, math.ceil(expires - time.time())),
                                      Path(cfg.root) / 'metadata' / f'watchdog-{slot}.log')
        actual = next(p for p in runpod.active_pods() if p['id'] == identifier)
        cost = actual.get('costPerHr')
        if cost is None or float(cost) > cfg.max_hourly_usd:
            raise RuntimeError('Allocated hourly cost missing or over ceiling')
        update(actual_hourly=float(cost))

    try:
        pod = runpod.provision_eval_pod(cfg.target, name=name, gpu=cfg.gpu, disk_gb=cfg.disk_gb,
            pubkey_path=cfg.ssh_key + '.pub', identity=cfg.ssh_key, eval='swebench_mini',
            revisions={cfg.target: cfg.target_revision, cfg.base: cfg.base_revision},
            terminate_at=datetime.fromtimestamp(expires, timezone.utc).isoformat(),
            env={'LASR_POD_OWNER': runpod.POD_OWNER, 'LASR_POD_DEADLINE': str(expires),
                 'LASR_CAMPAIGN': manifest['campaign']}, on_provisioned=allocated)
        assert pod.reachable, 'SSH failed'
        assert runpod.wait_bootstrapped(pod.id, timeout_s=min(cfg.boot_seconds, max(1, int(expires - time.time()))))
        if read(state.path).get('halt') or time.time() >= expires - cfg.cleanup_reserve_seconds:
            raise RuntimeError('Campaign stopped during boot')
        update(status='serving')
        log_path = Path(cfg.root) / 'metadata' / f'replica-{slot}.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'scratch.swebench_lite_worker', '--config', str(config_path),
                                     '--server', pod.host, '--replica', str(slot), '--allowed', str(allowed_path)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(worker_pid=proc.pid)
            while proc.poll() is None:
                live = read(state.path)
                if time.time() >= expires - cfg.cleanup_reserve_seconds or live.get('halt'):
                    stop_process(proc)
                    raise TimeoutError('Replica deadline or campaign halt')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f'Replica process failed ({proc.returncode}); see {log_path}')
    except BaseException as exc:
        update(error=type(exc).__name__ + ': ' + str(exc))
        with state.edit() as data:
            data['halt'] = f'replica {slot} failed; inspect saved logs before resume'
        raise
    finally:
        if proc:
            stop_process(proc)
        # Includes the ambiguous POST case: reconcile only this nonce AND exact name.
        matches = [p['id'] for p in runpod.active_pods() if owned(p, manifest) and p.get('name') == name]
        if pod_id and pod_id not in matches:
            matches.append(pod_id)
        for identifier in matches:
            runpod.teardown(identifier)
        if matches:
            update(status='terminated', ended=time.time())
        else:
            # A timed-out create may appear later: reserve its entire provider TTL.
            update(status='allocation-unconfirmed')
        if guard:
            guard.terminate()  # only after verified teardown and account/balance sweep


def phase(cfg, config_path, ids, count, seconds, manifest):
    state = State(cfg.root)
    data = read(state.path)
    remaining = manifest['budget_usd'] - reserved_cost(data)
    seconds = min(seconds, int(remaining * 3600 / (count * cfg.max_hourly_usd)))
    assert seconds > cfg.cleanup_reserve_seconds + 300, 'Remaining budget too small to allocate safely'
    expires = min(time.time() + seconds, data['deadline'])
    assert expires - time.time() > cfg.cleanup_reserve_seconds + 300
    allowed = Path(cfg.root) / 'metadata' / f'allowed-{len(data["pods"])}.json'
    atomic(allowed, ids)
    first_slot = len(data['pods'])
    last_upload = 0
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(replica, cfg, config_path, first_slot + i, allowed, expires, manifest) for i in range(count)]
        while not all(f.done() for f in futures):
            if time.time() - last_upload >= cfg.upload_every_seconds:
                checkpoint(cfg, config_path)
                last_upload = time.time()
            live = read(state.path)
            if time.time() - live['last_upload'] > cfg.upload_stale_seconds or time.time() > expires:
                with state.edit() as data:
                    data['halt'] = 'Backup stale or campaign deadline'
            if shutil.disk_usage('/var/lib/docker').free / 2**30 < cfg.min_free_gib:
                with state.edit() as data:
                    data['halt'] = 'CPU disk reserve reached'
            time.sleep(2)
        failures = [str(f.exception()) for f in futures if f.exception()]
    if failures:
        raise RuntimeError('; '.join(failures))
    checkpoint(cfg, config_path, required=True)


def grade(cfg):
    root = Path(cfg.root)
    state = read(root / 'metadata/state.json')
    manifest = read(root / 'metadata/manifest.json')
    assert not any(owned(p, manifest) for p in runpod.active_pods()), 'Release GPUs before grading'
    preds = {iid: t['attempts'][-1]['prediction'] for iid, t in state['tasks'].items() if t['status'] == 'valid'}
    # Empty valid submissions are unambiguously unresolved; upstream skips their tests.
    nonempty = {iid: p for iid, p in preds.items() if str(p.get('model_patch', '')).strip()}
    grading = root / 'results/grading'
    grading.mkdir(parents=True, exist_ok=True)
    predictions = grading / 'predictions.jsonl'
    predictions.write_text(''.join(json.dumps(p | {'instance_id': iid}) + '\n' for iid, p in preds.items()))
    request = {'fixture': read(root / 'metadata/httpbin_fixture.json'), 'harness': {
        'dataset_name': str(root / 'metadata/swebench_lite_test.json'), 'split': 'test',
        'instance_ids': list(nonempty), 'predictions_path': str(predictions),
        'max_workers': cfg.grading_workers, 'force_rebuild': False, 'cache_level': 'instance', 'clean': False,
        'open_file_limit': 16384, 'run_id': 'lite_' + manifest['campaign'], 'timeout': cfg.grading_timeout_seconds,
        'namespace': 'swebench', 'rewrite_reports': False, 'modal': False, 'report_dir': '.'}}
    atomic(grading / 'request.json', request)
    if nonempty:
        try:
            with (grading / 'harness.log').open('a') as log:
                subprocess.run([str(HARNESS), str(REPO / 'scratch/swebench_local_httpbin.py'), '--request', str(grading / 'request.json')],
                               cwd=grading, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=cfg.cpu_finish_reserve_seconds - cfg.cleanup_reserve_seconds)
        finally:
            cleanup_grading(root, manifest)
    reports = list(grading.glob('*.lite_' + manifest['campaign'] + '.json'))
    report = read(reports[0]) if reports else {}
    if nonempty and not reports:
        raise RuntimeError('Official grader produced no report')
    resolved = set(report.get('resolved_ids', [])) & set(preds)
    completed = set(report.get('completed_ids', [])) | (set(preds) - set(nonempty))
    patch_failed = set()
    for iid in set(nonempty) - completed:
        logs = list((grading / 'logs').glob('**/' + iid + '/run_instance.log'))
        if logs and '>>>>> Patch Apply Failed' in logs[0].read_text(errors='replace'):
            patch_failed.add(iid)
    completed |= patch_failed
    # Patch-apply failures are candidate failures, not transport failures. Other harness
    # errors remain ungraded until inspected/retried; never relaunch the model for them.
    valid_coverage = len(preds) == 300 and set(preds) <= completed
    from src.eval.capabilities.stats import wilson_ci
    result = {'status': 'complete' if valid_coverage else 'incomplete', 'n_total': 300,
              'n_valid_rollouts': len(preds), 'n_graded': len(completed), 'n_resolved': len(resolved),
              'resolved_ids': sorted(resolved), 'pass_at_1': len(resolved) / 300 if valid_coverage else None,
              'observed_resolved_fraction_300': len(resolved) / 300,
              'wilson_95': wilson_ci(len(resolved), 300) if valid_coverage else None,
              'infrastructure_invalid_ids': sorted(set(state['tasks']) - set(preds)),
              'ungraded_ids': sorted(set(preds) - completed),
              'patch_apply_failed_ids': sorted(patch_failed),
              'budget_upper_bound_usd': reserved_cost(state), 'limitations': manifest['limitations'],
              'task_results': {iid: {'rollout_status': t['status'], 'resolved': iid in resolved,
                                     'graded': iid in completed, 'attempts': len(t['attempts'])} for iid, t in state['tasks'].items()}}
    atomic(root / 'results/results.json', result)
    with State(root).edit() as data:
        data['phase'] = result['status']
    print(json.dumps({k: v for k, v in result.items() if k not in ('task_results', 'resolved_ids')}, indent=2))


def execute(cfg, config_path, action, budget):
    root = Path(cfg.root)
    assert os.environ.get('INVOCATION_ID'), 'Launch through lasr-swebench-lite.service so the independent reaper can supervise it'
    with lock(root / '.coordinator.lock', nonblocking=True):
        plan = preflight(cfg)
        print(json.dumps(plan, indent=2), flush=True)
        if action == 'run':
            assert budget is not None and 0 < budget <= cfg.recommended_budget_usd, 'Explicit positive --budget-usd required, at most configured recommendation'
            assert float(plan['balance']['clientBalance']) >= budget, 'RunPod balance below campaign cap'
            manifest = initialize(cfg, config_path, budget)
        else:
            manifest = read(root / 'metadata/manifest.json')
            assert manifest['config'] == OmegaConf.to_container(cfg) and manifest['source_hashes'] == sources(), 'Resume protocol/code drift'
            if budget is not None:
                assert budget == manifest['budget_usd'], 'Resume cannot silently reset the cumulative budget'
        state = State(root)
        fence(cfg, manifest)
        with state.edit() as data:
            data['halt'] = None
            data['breaker_failures_baseline'] = sum(a.get('valid') is False for t in data['tasks'].values() for a in t['attempts'])
            data['deadline'] = time.time() + cfg.campaign_seconds
            data['phase'] = 'calibration' if not data['calibrated'] else 'fleet'

        def halted(signum, frame):
            with state.edit() as data:
                data['halt'] = f'signal {signum}'
        signal.signal(signal.SIGTERM, halted)
        signal.signal(signal.SIGINT, halted)
        checkpoint(cfg, config_path, required=True)  # verify writable canonical HF before first rental
        try:
            if not read(state.path)['calibrated']:
                rows = read(root / 'metadata/swebench_lite_test.json')
                pilot = [next(r['instance_id'] for r in rows if r['repo'] == repo) for repo in cfg.pilot_repos]
                previous = read(state.path)
                pending = [iid for iid in pilot if previous['tasks'][iid]['status'] != 'valid']
                assert all(len(previous['tasks'][iid]['attempts']) < cfg.max_infrastructure_attempts for iid in pending), 'Pilot infrastructure retry allowance exhausted'
                started = time.time()
                if pending:
                    calibration_remaining = cfg.calibration_budget_usd - reserved_cost(previous)
                    phase(cfg, config_path, pending, 1, min(cfg.calibration_seconds,
                          int(calibration_remaining * 3600 / cfg.max_hourly_usd)), manifest)
                elif previous['pods']:
                    started = min(p['created'] for p in previous['pods'])
                data = read(state.path)
                assert all(data['tasks'][iid]['status'] == 'valid' for iid in pilot), 'Calibration did not finish validly'
                for iid in pilot:
                    attempt = data['tasks'][iid]['attempts'][-1]
                    traj = read(root / 'rollouts' / iid / attempt['id'] / iid / (iid + '.traj.json'))
                    assert any(m.get('tool_calls') for m in traj.get('messages', []) if m.get('role') == 'assistant'), 'Pilot lacks valid tool calls'
                elapsed = time.time() - started
                attempts = [data['tasks'][iid]['attempts'][-1] for iid in pilot]
                # Active attempt time excludes downtime between explicit resumes.
                gpu_seconds_per_task = sum(a['finished'] - a['started'] for a in attempts) / (len(pilot) * cfg.workers_per_replica)
                # Charge cold boot once PER replica, not once per six-task batch.
                boot_seconds = max(0, min(a['started'] for a in attempts) - started)
                proposed_replicas = min(cfg.replicas, max(1, math.ceil((300 - len(pilot)) * gpu_seconds_per_task / cfg.target_generation_seconds)))
                projected = ((300 - len(pilot)) * gpu_seconds_per_task + proposed_replicas * boot_seconds) / 3600 * cfg.max_hourly_usd
                atomic(root / 'metadata/calibration.json', {'tasks': pilot, 'elapsed_seconds': elapsed,
                       'gpu_seconds_per_task': gpu_seconds_per_task,
                       'boot_seconds': boot_seconds, 'proposed_replicas': proposed_replicas,
                       'projected_remaining_gpu_usd': projected, 'estimate_only': True})
                assert projected + reserved_cost(data) <= manifest['budget_usd'], 'Pilot projects over budget; stop before fleet'
                grade(cfg)
                assert not read(root / 'results/results.json')['ungraded_ids'], 'Pilot grading failed'
                with state.edit() as data:
                    data['calibrated'] = True
                    data['phase'] = 'fleet'
            data = read(state.path)
            todo = [iid for iid, t in data['tasks'].items() if t['status'] != 'valid' and len(t['attempts']) < cfg.max_infrastructure_attempts]
            if todo:
                measured = read(root / 'metadata/calibration.json')['gpu_seconds_per_task']
                count = min(cfg.replicas, math.ceil(len(todo) / cfg.workers_per_replica),
                            max(1, math.ceil(len(todo) * measured / cfg.target_generation_seconds)))
                atomic(root / 'metadata/fleet_plan.json', {'replicas': count, 'ceiling': cfg.replicas,
                       'inference_seconds_estimate': len(todo) * measured / count, 'estimate_only': True})
                phase(cfg, config_path, todo, count, cfg.campaign_seconds, manifest)
        finally:
            fence(cfg, manifest)
            checkpoint(cfg, config_path)
        try:
            grade(cfg)
        finally:
            checkpoint(cfg, config_path, required=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'run', 'resume', 'grade', 'publish', 'status', 'guard'])
    parser.add_argument('--config', default='scratch/swebench_lite.yaml')
    parser.add_argument('--budget-usd', type=float)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = OmegaConf.load(config_path)
    load_dotenv(cfg.credentials)
    os.chdir(REPO)
    if args.action == 'plan':
        print(json.dumps(preflight(cfg), indent=2))
    elif args.action == 'publish':
        publish(cfg)
    elif args.action == 'guard':
        guard(cfg)
    elif args.action == 'status':
        data = read(Path(cfg.root) / 'metadata/state.json')
        print(json.dumps({'phase': data['phase'], 'halt': data['halt'], 'valid': sum(t['status'] == 'valid' for t in data['tasks'].values()),
                          'total': len(data['tasks']), 'pods': data['pods'], 'last_upload': data['last_upload'],
                          'cost_upper_bound_usd': reserved_cost(data)}, indent=2))
    elif args.action == 'grade':
        with lock(Path(cfg.root) / '.coordinator.lock', nonblocking=True):
            try:
                grade(cfg)
            finally:
                checkpoint(cfg, config_path, required=True)
    else:
        execute(cfg, config_path, args.action, args.budget_usd)


if __name__ == '__main__':
    main()
