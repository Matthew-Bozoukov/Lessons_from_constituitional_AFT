# ABOUTME: Persistent, bounded full SWE-bench Lite campaign on a prepared CPU host and RunPod replicas.
# ABOUTME: Preparation and grading rent nothing; only explicit run/resume can allocate inference GPUs.
from src.eval.capabilities.swebench_mini.fleet_host import receipt_deadline, deadline_value
import argparse
import hashlib
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
import threading

from dotenv import load_dotenv
from omegaconf import OmegaConf
import psutil

from src.infra import runpod
from src.infra.endpoints.vllm import resolve_target, SshExec, POD_VENV
from src.infra.huggingface import hf_api, hf_download, hf_repo_id, push_run_dir
from src.naming import eval_name, today
from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, digest, lock, read
from src.eval.capabilities.swebench_mini.fleet_worker import stop_process
from src.eval.capabilities.swebench_mini import fleet_session as session

REPO = Path(__file__).resolve().parents[4]
ALLOCATION_LOCK = threading.Lock()
HARNESS = REPO / 'src/eval/capabilities/swebench_mini/envs/harness/.venv/bin/python'


def recipe_settings(cfg):
    keys = ('base', 'base_revision', 'mode', 'gpu', 'replicas', 'workers_per_replica',
            'serving', 'max_response_tokens', 'max_task_tokens', 'agent_cpus', 'agent_memory',
            'agent_pids', 'agent_environment', 'grading_workers', 'grading_timeout_seconds', 'dataset_revision',
            'fallback_gpus', 'cuda_versions', 'cpu_worker_limit', 'allocation_fallback_after_seconds',
            'allocation_fallback_after_attempts')
    keys += ('model_request_timeout_seconds', 'model_request_attempts')
    keys += tuple(k for k in ('task_seconds', 'task_admission_seconds', 'rental_seconds') if k in cfg)
    keys += tuple(k for k in ('tool_concurrency', 'tool_queue_timeout_seconds', 'cpu_qualification_path', 'task_scheduling') if k in cfg)
    return {key: OmegaConf.to_container(cfg[key]) if OmegaConf.is_config(cfg[key]) else cfg[key] for key in keys}


def validate_recipe(cfg, recipe):
    qualified = recipe.get('qualification', {})
    assert (recipe.get('validated_full_run') or
            (qualified.get('protocol_reviewed') and qualified.get('cpu_qualified') and
             qualified.get('recovery_tests_passed'))), 'Recipe needs a complete graded run or explicit protocol and infrastructure qualification'
    assert recipe['settings'] == recipe_settings(cfg), 'Frozen recipe mismatch; explicit recalibration required'
    # New target/base artifacts are parameters; implementation changes require review.
    assert recipe['source_hashes'] == sources(), 'Frozen recipe code drift'
    return recipe


def priority_order(rows, profile, dataset_revision):
    """Freeze a runtime-only prior; no tasks are added, dropped or scored here."""
    ids = [r['instance_id'] for r in rows]
    assert len(ids) == len(set(ids)), 'Duplicate dataset task IDs'
    assert profile['policy'] == 'historical-longest-first-v1', 'Unknown scheduling policy'
    assert profile['dataset_revision'] == dataset_revision, 'Timing profile dataset mismatch'
    seconds = profile['seconds_by_instance']
    assert set(seconds) == set(ids), 'Timing profile must cover exactly the requested tasks'
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
               for v in seconds.values()), 'Invalid historical task duration'
    return sorted(ids, key=lambda iid: (-seconds[iid], iid))


def scheduling_plan(cfg, rows):
    reference = cfg.get('task_scheduling')
    if not reference:  # Preserve historical configurations and their queue order.
        return {'policy': 'dataset-order', 'instance_ids': [r['instance_id'] for r in rows]}
    path = hf_download(reference.repo, reference.path, repo_type='dataset', revision=reference.revision)
    assert digest(path) == reference.sha256, 'Timing profile checksum mismatch'
    profile = read(path)
    order = priority_order(rows, profile, cfg.dataset_revision)
    return {'policy': profile['policy'], 'reference': OmegaConf.to_container(reference),
            'profile': profile, 'instance_ids': order}


def prepare_target(cfg, args):
    assert args.target and args.write_config and args.root, 'prepare requires --target, --root, --write-config'
    assert not Path(args.write_config).exists(), 'Refusing to overwrite an existing configuration'
    revision = hf_api().model_info(args.target, revision=args.target_revision).sha
    target = resolve_target(args.target, revision=revision)
    cfg.target, cfg.target_revision = args.target, revision
    cfg.base, cfg.base_revision, cfg.mode = target.base_model, target.base_revision, target.mode
    cfg.root, cfg.calibrate = str(Path(args.root).resolve()), False
    validate_recipe(cfg, read(cfg.recipe_path))
    assert not (Path(cfg.root) / 'metadata/manifest.json').exists(), 'Choose a fresh run directory'
    OmegaConf.save(cfg, args.write_config)
    print('Prepared fixed-fleet config (no rental):', args.write_config)


def account():
    return runpod.graphql('query { myself { clientBalance currentSpendPerHr } }')['myself']


def qualify_shell(cfg):
    """One-time CPU-only probe of every cached agent image, no model/provider calls."""
    images = read(Path(cfg.readiness) / 'metadata/images.json')
    env = OmegaConf.to_container(cfg.agent_environment)
    def probe(item):
        iid, image = item
        name = 'lasr-shell-probe-' + uuid.uuid4().hex
        cmd = ['docker', 'run', '--name', name, '--rm', '--network', 'none', '--pull', 'never',
               '--cpus', '1', '--memory', '1g', '--pids-limit', '128', '--entrypoint', 'bash']
        for key, value in env.items():
            cmd += ['-e', key + '=' + value]
        cmd += [image['digest'], '-c', 'python -c "import sys,json; assert sys.prefix == \'/opt/miniconda3/envs/testbed\', sys.prefix; print(json.dumps(dict(prefix=sys.prefix,version=sys.version)))"']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            return iid, {'image_id': image['id'], 'returncode': result.returncode,
                         'output': result.stdout.strip(), 'stderr': result.stderr[-1000:]}
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=30)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = dict(pool.map(probe, images.items()))
    proof = {'status': 'passed' if len(results) == 300 and all(r['returncode'] == 0 for r in results.values()) else 'failed',
             'environment': env, 'dataset_revision': cfg.dataset_revision, 'images': results}
    path = Path(cfg.readiness) / 'results/agent-shell.json'
    atomic(path, proof)
    print('Agent shell qualification:', proof['status'], len(results), path, flush=True)
    assert proof['status'] == 'passed', 'Agent image shell activation failed; do not rent GPUs'


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
    scheduling_plan(cfg, rows)  # Validate/download before any inference rental.
    images = read(ready / 'metadata/images.json')
    shell = read(ready / 'results/agent-shell.json')
    assert shell['status'] == 'passed' and shell['environment'] == OmegaConf.to_container(cfg.agent_environment)
    assert shell['dataset_revision'] == cfg.dataset_revision
    assert {iid: r['image_id'] for iid, r in shell['images'].items()} == {iid: r['id'] for iid, r in images.items()}
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
    workers = cfg.replicas * cfg.workers_per_replica
    assert workers <= cfg.cpu_worker_limit, 'CPU worker safety limit exceeded'
    assert cfg.agent_memory == '4g', 'Requalify CPU RAM sizing when changing per-agent memory'
    if workers > 40:
        capacity = read(cfg.cpu_qualification_path)
        assert capacity['qualified_workers'] >= workers and capacity['status'] == 'passed'
        assert capacity['cpu_count'] <= psutil.cpu_count()
        assert capacity['min_host_memory_gib'] <= psutil.virtual_memory().total / 2**30
        assert capacity['tool_concurrency'] == cfg.tool_concurrency
        assert capacity['agent_cpus'] == cfg.agent_cpus and capacity['agent_memory'] == cfg.agent_memory
        assert capacity['dataset_revision'] == cfg.dataset_revision
        assert capacity['images_sha256'] == digest(ready / 'metadata/images.json')
        from src.eval.capabilities.swebench_mini.fleet_task import resource_shell
        assert capacity['resource_policy_sha256'] == hashlib.sha256(resource_shell(cfg.agent_cpus, 120).encode()).hexdigest()
        assert capacity['agent_pids'] == cfg.agent_pids
    else:
        assert workers * 4 + cfg.cpu_memory_reserve_gib <= psutil.virtual_memory().total / 2**30
        assert cfg.agent_cpus * workers <= psutil.cpu_count() * 1.5, 'Excessive CPU oversubscription'
    assert 0 < cfg.max_response_tokens <= cfg.max_task_tokens
    assert cfg.serving.concurrency >= cfg.workers_per_replica
    if not cfg.calibrate:
        validate_recipe(cfg, read(cfg.recipe_path))
    receipt = read(cfg.receipt)
    available = deadline_value(receipt_deadline(receipt)) - time.time()
    assert available > cfg.allocation_min_remaining_seconds + cfg.cpu_finish_reserve_seconds + 120, 'CPU expiry cannot accommodate boot, one full task and grading; do not rent GPUs'
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
    from src.eval.capabilities.swebench_mini.fixture import ensure_fixture
    fixture = ensure_fixture(OmegaConf.load(REPO / 'scratch/swebench_cpu.yaml'), receipt)
    assert fixture == read(ready / 'metadata/httpbin_fixture.json')
    return {'status': 'ready_to_launch', 'tasks': 300, 'model': cfg.target, 'revision': cfg.target_revision,
            'replicas': cfg.replicas, 'workers': cfg.replicas * cfg.workers_per_replica,
            'gpu_quote_usd_hour': quote, 'balance': balance, 'cpu_stop_at': receipt['stop_at'],
            'recommended_budget_usd': cfg.recommended_budget_usd,
            'paid_gpu_test': 'one calibration replica' if cfg.calibrate else 'fixed fleet; CPU/recovery qualified, no new paid calibration'}


def sources():
    paths = list((REPO / 'src/eval/capabilities/swebench_mini').glob('fleet*.py'))
    paths += [REPO / 'src/eval/capabilities/swebench_mini/fixture.py']
    paths += [REPO / p for p in ['src/infra/runpod.py', 'src/infra/endpoints/vllm.py', 'src/eval/run_eval.py',
                               'src/model_profile.py', 'configs/models/qwen36.yaml', 'src/naming.py',
                               'src/infra/huggingface.py', 'src/utils.py', 'pyproject.toml',
                               'src/eval/capabilities/swebench_mini/agent.py', 'src/eval/capabilities/stats.py',
                               'scratch/swebench_local_httpbin.py', 'scratch/swebench_timeout_recovery.py', 'scratch/swebench_cpu_env/uv.lock',
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
    schedule = scheduling_plan(cfg, rows)
    manifest = {'campaign': campaign, 'repo': repo, 'created': datetime.now(timezone.utc).isoformat(),
                'date': today(), 'model_key': spec.model_key, 'config': OmegaConf.to_container(cfg),
                'source_hashes': sources(), 'budget_usd': budget, 'dataset_tasks': 300,
                'protocol': 'mini-swe-agent 2.2.1; official 250 steps, inert local dollar limit; network none; '
                            f'digest-pinned cached images; {cfg.agent_cpus} CPU/{cfg.agent_memory}/{cfg.agent_pids} PID agent container caps; infrastructure retries only (max {cfg.max_infrastructure_attempts} attempts); '
                            'quota-derived test/BLAS thread limits; in-container command timeout with descendant cleanup; '
                            'requests local HTTPBin fixture for grading; full denominator 300',
                'limitations': read(Path(cfg.readiness) / 'results/readiness.json')['benchmark_limitations']}
    manifest['protocol'] += (f'; response cap {cfg.max_response_tokens}, task completion-token cap {cfg.max_task_tokens}; '
                             f'HTTP timeout {cfg.model_request_timeout_seconds}s, request attempts {cfg.model_request_attempts}; '
                             'token-limit outcomes terminate unresolved without model rerolls; '
                             'agent shell environment ' + json.dumps(OmegaConf.to_container(cfg.agent_environment)))
    manifest['protocol'] += f'; task wall-clock cap {cfg.get("task_seconds")}; finite per-pod emergency lease; token/step limits unchanged'
    if cfg.get('following_configs') or cfg.get('fleet_owner_root'):
        manifest['shared_fleet'] = {'owner_root': cfg.get('fleet_owner_root') or cfg.root,
            'budget_scope': 'One cumulative fleet budget, not an independent allowance per arm',
            'following_configs': list(cfg.get('following_configs', []))}
    deployment = Path('/srv/lasr/lite-deployment.json')
    if deployment.exists():
        manifest['deployment'] = read(deployment)
    for sub in ('metadata', 'rollouts', 'results'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    atomic(root / 'metadata/task-schedule.json', schedule)
    manifest['task_schedule_sha256'] = digest(root / 'metadata/task-schedule.json')
    for name in ('swebench_lite_test.json', 'images.json', 'httpbin_fixture.json', 'httpbin-ca.pem'):
        shutil.copyfile(Path(cfg.readiness) / 'metadata' / name, root / 'metadata' / name)
    shutil.copyfile(Path(cfg.readiness) / 'results/agent-shell.json', root / 'metadata/agent-shell.json')
    shutil.copyfile(config_path, root / 'metadata/config.yaml')
    for name, sha in manifest['source_hashes'].items():
        dest = root / 'metadata/source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dest)
        assert digest(dest) == sha
    atomic(root / 'metadata/manifest.json', manifest)
    atomic(root / 'metadata/state.json', {'tasks': {iid: {'status': 'pending', 'attempts': []} for iid in schedule['instance_ids']},
           'pods': [], 'deadline': 0, 'last_upload': 0, 'phase': 'prepared', 'calibrated': not cfg.calibrate, 'halt': None})
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
        if proc.pid != os.getpid() and ('src.eval.capabilities.swebench_mini.fleet_task' in cmd or 'src.eval.capabilities.swebench_mini.fleet_worker' in cmd):
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
    rejected_names = {p['name'] for p in state.get('pods', [])
                      if p['status'] in ('allocation-unconfirmed', 'rejected-reconciled')}
    for pod in runpod.active_pods():
        if owned(pod, manifest):
            deadline = float((pod.get('env') or {}).get('LASR_POD_DEADLINE', 0))
            if not active or pod.get('name') in rejected_names or time.time() >= min(deadline, deadline_value(state['deadline'])):
                runpod.teardown(pod['id'])
    if not active:
        try:
            with lock(Path(cfg.root) / '.coordinator.lock', nonblocking=True):
                for arm in session.members(cfg):
                    arm_path = Path(arm.root) / 'metadata/manifest.json'
                    if not arm_path.exists():
                        continue
                    arm_manifest = read(arm_path)
                    cleanup_grading(arm.root, arm_manifest)
                    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter',
                          'label=lasr_campaign=' + arm_manifest['campaign']], text=True).split()
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
                # Copy contents: hard links do not freeze upstream in-place writes.
                shutil.copytree(root / sub, snapshot / sub, copy_function=shutil.copy2,
                                ignore=shutil.ignore_patterns('*.tmp', '__pycache__'))
        repo = manifest['repo']
        api = hf_api()
        if api.repo_exists(repo, repo_type='dataset'):
            old = read(hf_download(repo, 'metadata/manifest.json', repo_type='dataset'))
            assert old['campaign'] == manifest['campaign'], 'HF campaign identity mismatch'
        fields = {'experiment': 'Full 300-task SWE-bench Lite; partial until valid coverage and grading finish',
                  'date_generated': manifest['date'], 'constitution': 'none',
                  'source_repo': 'teaching_claude_why_replication; exact sources and SHA256 in metadata/source',
                  'models': cfg.target + '@' + cfg.target_revision + '; base ' + cfg.base + '@' + cfg.base_revision,
                  'generation_config': json.dumps(manifest['config']),
                  'schema': 'rollouts/: every attempt JSON and readable transcript; results/: official reports and score; metadata/: pins, ledger, code',
                  'provenance': manifest['protocol'] + '; ' + manifest['limitations']}
        push_run_dir(snapshot, repo, fields, atomic_commit=True, front_matter={'tags': ['eval-run', 'eval:swebench_mini',
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
        subprocess.run([sys.executable, '-m', 'src.eval.capabilities.swebench_mini.fleet', 'publish', '--config', str(config_path)],
                       check=True, timeout=cfg.upload_timeout_seconds)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        print('HF checkpoint failed:', type(exc).__name__, flush=True)
        if required:
            raise
        return False


def reserved_cost(state):
    return sum(p['ceiling_hourly'] * (p.get('ended', p['expires']) - p['created']) / 3600
               for p in state['pods'] if not (
                   p.get('reservation_released') and p.get('status') == 'rejected-reconciled'
                   and p.get('id') is None and p.get('reconciliation_inventory_checks') == 2
                   and 'RunPod GraphQL rejected the request;' in p.get('error', '')))


def final_accounting(cfg):
    root = Path(cfg.root)
    state = read(root / 'metadata/state.json')
    manifest = read(root / 'metadata/manifest.json')
    inventory = runpod.active_pods()
    assert not any(owned(p, manifest) for p in inventory), 'Owned GPU still billing'
    ids = {p['id'] for p in state['pods'] if p.get('id')}
    receipt = read(cfg.receipt)
    report = {'observed_at': time.time(), 'owned_gpus_remaining': [],
              'gpu_conservative_ledger_usd': reserved_cost(state),
              'cpu_hourly_quote_usd': receipt['offer']['dph_total'],
              'cpu_storage_transfer_invoiced_usd': None,
              'billing_note': 'Provider rows may lag. CPU/storage/transfer are separate; shared account balance changes are not run cost.'}
    elapsed = max(0, time.time() - datetime.fromisoformat(manifest['created']).timestamp())
    report['cpu_campaign_elapsed_quote_usd'] = elapsed / 3600 * receipt['offer']['dph_total']
    report['cpu_quote_note'] = 'Elapsed campaign wall time times the receipted host quote; excludes earlier preparation and retention after completion, not an invoice.'
    try:
        rows = runpod.call('GET', '/billing/pods', params={
            'startTime': manifest['created'][:10] + 'T00:00:00Z', 'endTime': datetime.now(timezone.utc).isoformat(),
            'bucketSize': 'hour', 'grouping': 'podId'})
        assert isinstance(rows, list), 'Unexpected provider billing schema'
        rows = [r for r in rows if r.get('podId') in ids]
        report.update(provider_gpu_recorded_usd=sum(float(r['amount']) for r in rows),
                      provider_rows=rows, provider_missing_pod_ids=sorted(ids - {r['podId'] for r in rows}))
    except Exception as exc:
        report.update(provider_gpu_recorded_usd=None, billing_error=type(exc).__name__ + ': ' + str(exc))
    atomic(root / 'metadata/final-accounting.json', report)
    metrics = cache_audit(root)
    atomic(root / 'metadata/cache-audit.json', metrics)
    return report


def cache_audit(root):
    replicas = []
    for path in (Path(root) / 'metadata').glob('metrics-*.jsonl'):
        samples = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # A crash can truncate the last telemetry line.
            metric = {}
            for entry in row.get('metrics', []):
                name = entry.split('{')[0]
                metric[name] = metric.get(name, 0) + float(entry.rsplit(' ', 1)[1])
            if metric:
                samples.append(metric)
        if not samples:
            continue
        def delta(key):
            values = [s[key] for s in samples if key in s]
            return sum(b-a if b >= a else b for a, b in zip(values, values[1:]))
        queries, hits = delta('vllm:prefix_cache_queries_total'), delta('vllm:prefix_cache_hits_total')
        replicas.append({'slot': path.stem.removeprefix('metrics-'), 'samples': len(samples),
            'prefix_queries': queries, 'prefix_hits': hits,
            'max_kv_fraction': max(s.get('vllm:kv_cache_usage_perc', 0) for s in samples),
            'preemptions': delta('vllm:num_preemptions_total')})
    queries = sum(r['prefix_queries'] for r in replicas)
    return {'replicas': replicas, 'prefix_hit_fraction': sum(r['prefix_hits'] for r in replicas)/queries if queries else None,
            'preemptions': sum(r['preemptions'] for r in replicas),
            'limitations': 'Sampled counter deltas; cache hit rate is not an end-to-end speedup.'}


def verify_final_files(cfg, revision):
    """Verify every rollout/result against the immutable uploaded tree, not just a success response."""
    root = Path(cfg.root)
    manifest = read(root / 'metadata/manifest.json')
    tree = {f.path: f for f in hf_api().list_repo_tree(manifest['repo'], repo_type='dataset',
            recursive=True, revision=revision) if hasattr(f, 'blob_id')}
    checked = 0
    for sub in ('rollouts', 'results'):
        for path in (root / sub).rglob('*'):
            if (not path.is_file() or path.name.endswith('.tmp') or
                    any(part in {'.git', '.cache', '__pycache__'} for part in path.relative_to(root).parts)):
                continue
            name = path.relative_to(root).as_posix()
            assert name in tree, 'Missing final HF artifact: ' + name
            payload = path.read_bytes()
            remote = tree[name]
            checksum = (hashlib.sha256(payload).hexdigest() if remote.lfs else
                        hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest())
            assert checksum == (remote.lfs.sha256 if remote.lfs else remote.blob_id), 'HF artifact hash mismatch: ' + name
            checked += 1
    return checked


def reconcile_rejections(cfg, manifest):
    """Reconcile explicit provider rejections; transport timeouts keep their TTL reserve."""
    state = State(cfg.root)
    candidates = [p for p in read(state.path)['pods']
                  if p['status'] in ('allocation-unconfirmed', 'rejected-reconciled')
                  and not p.get('reservation_released') and p.get('id') is None
                  and 'RunPod GraphQL rejected the request;' in p.get('error', '')
                  and time.time() - p['created'] >= cfg.allocation_reconcile_seconds]
    if not candidates:
        return
    live = runpod.active_pods()
    late = {}
    # Any late create is fenced before releasing a reservation. Never adopt a pod
    # whose create response was lost: it has no confirmed watchdog callback.
    for pod in live:
        if owned(pod, manifest) and pod.get('name') in {p['name'] for p in candidates}:
            late[pod['name']] = pod['id']
            runpod.teardown(pod['id'])
    present = {p.get('name') for p in runpod.active_pods() if owned(p, manifest)}
    with state.edit() as data:
        for p in data['pods']:
            if p['slot'] in {p['slot'] for p in candidates} and p['name'] not in present:
                if p['name'] in late:
                    p.update(status='terminated', id=late[p['name']], ended=time.time(),
                             reconciliation='Late create observed and reaped; elapsed ceiling remains charged')
                else:
                    p.setdefault('ended', min(time.time(), p['expires']))
                    p.update(status='rejected-reconciled', reservation_released=True,
                             reservation_released_at=time.time(), reconciliation_inventory_checks=2,
                             reconciliation='Explicit GraphQL rejection; absent in two provider inventory sweeps; no late create observed')


def pending_tasks(data, allowed, cfg):
    data = session.view(cfg, data)
    return any(iid in allowed and t['status'] in ('pending', 'invalid')
               and len(t['attempts']) < cfg.max_infrastructure_attempts for iid, t in data['tasks'].items())


def allocation_gpu(cfg, failures, elapsed, fallback_failures=None):
    if failures < cfg.allocation_fallback_after_attempts or elapsed < cfg.allocation_fallback_after_seconds:
        return cfg.gpu
    options = list(cfg.fallback_gpus) + [cfg.gpu]
    offset = failures - cfg.allocation_fallback_after_attempts if fallback_failures is None else fallback_failures
    return options[offset % len(options)]


def cancel_idle_startup(state, slot, allowed, cfg):
    # Serialize against the worker's ready transition, before it can claim tasks.
    with state.edit() as data:
        pod = next(p for p in data['pods'] if p['slot'] == slot)
        if pod.get('ready_at') or pending_tasks(data, allowed, cfg):
            return False
        pod['idle_startup_cancellation'] = {'at': time.time(), 'reason': 'No unleased eligible tasks'}
        return True


def price_ceiling(cfg, gpu):
    price = runpod.gpu_price(gpu)
    if not price or price > cfg.max_hourly_usd:
        raise RuntimeError('GPU quote unavailable or exceeds authorized hourly ceiling: ' + gpu)
    return min(cfg.max_hourly_usd, price * cfg.gpu_price_margin)


def replica(cfg, config_path, slot, allowed_path, expires, manifest):
    state = State(cfg.root)
    name = f"{os.environ['USER_PREFIX']}-swe-lite-{manifest['campaign'][:8]}-{slot}"
    ceiling = price_ceiling(cfg, cfg.gpu)
    # A more expensive fallback must not monopolize the other lanes' reservations.
    # Provider expiry, detached watchdog, worker and ledger all use this same value.
    if cfg.get('rental_reservation_usd') is not None:
        expires = min(expires, time.time() + cfg.rental_reservation_usd * 3600 / ceiling)
    record = {'slot': slot, 'name': name, 'created': time.time(), 'expires': expires, 'gpu': cfg.gpu,
              'ceiling_hourly': ceiling, 'id': None, 'status': 'allocating'}
    boot_deadline = record['created'] + cfg.boot_seconds
    with state.edit() as data:
        reserve = ceiling * (expires - record['created']) / 3600
        if reserved_cost(data) + reserve > manifest['budget_usd']:
            raise RuntimeError('Allocation deferred: cumulative budget reservation unavailable')
        data['pods'].append(record)
    guard = proc = executor = None
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
        if cost is None or float(cost) > ceiling:
            raise RuntimeError('Allocated hourly cost missing or over ceiling')
        update(actual_hourly=float(cost))

    def provisioned(identifier):
        # Serialize only the create request, never SSH, weight download or serving.
        # Avoid simultaneous creates; the observed provider failures may be a scheduler race.
        ALLOCATION_LOCK.release()
        allocated(identifier)

    try:
        released = False
        def on_created(identifier):
            nonlocal released
            released = True
            provisioned(identifier)
        ALLOCATION_LOCK.acquire()
        try:
            if read(state.path).get('halt') or time.time() >= expires - cfg.allocation_min_remaining_seconds:
                update(status='not-requested', ended=record['created'])
                raise RuntimeError('Allocation cancelled before provider request')
            boot_arms = session.members(cfg)
            boot_arm = boot_arms[cfg.get('primary_arm', 0) % len(boot_arms)]
            pod = runpod.provision_eval_pod(boot_arm.target, name=name, gpu=cfg.gpu, disk_gb=cfg.disk_gb,
                pubkey_path=cfg.ssh_key + '.pub', identity=cfg.ssh_key, eval='swebench_mini',
                revisions={boot_arm.target: boot_arm.target_revision, cfg.base: cfg.base_revision},
                terminate_at=datetime.fromtimestamp(expires, timezone.utc).isoformat(),
                cuda_versions=cfg.cuda_versions,
                boot_deadline=boot_deadline,
                env={'LASR_POD_OWNER': runpod.POD_OWNER, 'LASR_POD_DEADLINE': str(expires),
                     'LASR_CAMPAIGN': manifest['campaign']}, on_provisioned=on_created)
        finally:
            if not released:
                ALLOCATION_LOCK.release()
        assert pod.reachable, 'SSH failed'
        assert runpod.wait_bootstrapped(pod.id, timeout_s=max(1, int(min(boot_deadline, expires) - time.time())))
        assert time.time() < boot_deadline, 'Cold bootstrap exceeded its total allowance'
        if read(state.path).get('halt') or time.time() >= expires - cfg.cleanup_reserve_seconds:
            raise RuntimeError('Campaign stopped during boot')
        if cancel_idle_startup(state, slot, read(allowed_path), cfg):
            return  # finally still verifies teardown and closes the reservation.
        executor = SshExec(pod.host, port=cfg.port_base + slot, identity=cfg.ssh_key)
        runtime = executor._ssh(f'{POD_VENV}/bin/python -', timeout=90, stdin_text='''
import json, torch, importlib.metadata
assert torch.cuda.is_available(), 'CUDA runtime unavailable'
p = torch.cuda.get_device_properties(0)
assert p.total_memory >= 85 * 2**30, 'Unexpected GPU memory capacity'
x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
print(json.dumps({'gpu': p.name, 'memory_gib': p.total_memory / 2**30,
                  'torch': torch.__version__, 'vllm': importlib.metadata.version('vllm'),
                  'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}))
''')
        atomic(Path(cfg.root) / 'metadata' / f'gpu-runtime-{slot}.json', json.loads(runtime.strip().splitlines()[-1]))
        update(status='serving')
        log_path = Path(cfg.root) / 'metadata' / f'replica-{slot}.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'src.eval.capabilities.swebench_mini.fleet_worker', '--config', str(config_path),
                                     '--server', pod.host, '--replica', str(slot), '--allowed', str(allowed_path),
                                     '--expires', str(expires), '--primary-arm', str(cfg.get('primary_arm', 0))],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(worker_pid=proc.pid)
            while proc.poll() is None:
                live = session.view(cfg)
                own_record = next(p for p in live['pods'] if p['slot'] == slot)
                if not own_record.get('ready_at') and cancel_idle_startup(state, slot, read(allowed_path), cfg):
                    return  # Stop only this unready worker; healthy replicas drain.
                if time.time() >= boot_deadline and not own_record.get('ready_at'):
                    stop_process(proc)
                    raise TimeoutError('Total bootstrap/serving startup allowance exceeded')
                if time.time() >= expires - cfg.cleanup_reserve_seconds or live.get('halt'):
                    stop_process(proc)
                    raise TimeoutError('Replica deadline or campaign halt')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f'Replica process failed ({proc.returncode}); see {log_path}')
    except BaseException as exc:
        update(error=type(exc).__name__ + ': ' + str(exc))
        # A missing replica must not cancel healthy replicas. Global budget,
        # deadline and task-failure breakers remain coordinator-owned.
        raise
    finally:
        cleanup_error = None
        if proc:
            try:
                stop_process(proc)
                session.recover_worker(cfg, slot)
            except Exception as exc:
                cleanup_error = exc
                with state.edit() as data:
                    data['halt'] = 'Replica process/container cleanup failed; GPU teardown still attempted'
        if executor:
            try:
                log = executor._ssh('cat /workspace/output/serve/vllm.log', timeout=15)
                (Path(cfg.root) / 'metadata' / f'vllm-{slot}.log').write_text(log)
            except Exception as exc:
                update(log_capture_error=type(exc).__name__)
        # Includes the ambiguous POST case: reconcile only this nonce AND exact name.
        matches = [p['id'] for p in runpod.active_pods() if owned(p, manifest) and p.get('name') == name]
        if pod_id and pod_id not in matches:
            matches.append(pod_id)
        for identifier in matches:
            runpod.teardown(identifier)
        if matches:
            update(status='terminated', ended=time.time())
        elif next(p for p in read(state.path)['pods'] if p['slot'] == slot)['status'] != 'not-requested':
            # A timed-out create may appear later: reserve its entire provider TTL.
            update(status='allocation-unconfirmed')
        if guard:
            guard.terminate()  # only after verified teardown and account/balance sweep
        if cleanup_error:
            raise cleanup_error


def phase(cfg, config_path, ids, count, seconds, manifest):
    state = State(cfg.root)
    data = read(state.path)
    remaining = manifest['budget_usd'] - reserved_cost(data)
    seconds = min(seconds, int(remaining * 3600 / (count * price_ceiling(cfg, cfg.gpu))))
    assert seconds > cfg.allocation_min_remaining_seconds, 'Remaining budget cannot cover boot plus a full task; no rental'
    # `seconds` is a per-rental safety lease, never a shared batch cutoff. Late
    # arrivals and replacements get their own lease, bounded by CPU expiry/cost.
    assert deadline_value(data['deadline']) - time.time() > cfg.allocation_min_remaining_seconds
    allowed = Path(cfg.root) / 'metadata' / f'allowed-{len(data["pods"])}.json'
    atomic(allowed, ids)
    next_slot = max((p['slot'] for p in data['pods']), default=-1) + 1
    last_upload = 0
    last_reconcile = 0
    retry_at = {i: 0 for i in range(count)}
    attempts = {i: 0 for i in range(count)}
    failures = {i: 0 for i in range(count)}
    fallback_failures = {i: 0 for i in range(count)}
    fallback_attempt = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {}
        while True:
            live = session.view(cfg)
            for lane, future in list(futures.items()):
                if not future.done():
                    continue
                del futures[lane]
                if future.exception():
                    failures[lane] += 1
                    if fallback_attempt.get(lane):
                        fallback_failures[lane] += 1
                    print(f'Fleet slot {lane}: {future.exception()}; peers continue', flush=True)
                    retry_at[lane] = time.time() + min(cfg.allocation_retry_max_seconds,
                        cfg.allocation_retry_seconds * 2 ** min(attempts[lane] - 1, 4))
            if (not live.get('halt') and pending_tasks(live, ids, cfg)
                    and deadline_value(live['deadline']) - time.time() > cfg.allocation_min_remaining_seconds):
                for lane in range(count):
                    if lane in futures or time.time() < retry_at[lane]:
                        continue
                    attempts[lane] += 1
                    elapsed = time.time() - started
                    selected = allocation_gpu(cfg, failures[lane], elapsed, fallback_failures[lane])
                    fallback_attempt[lane] = (failures[lane] >= cfg.allocation_fallback_after_attempts
                                              and elapsed >= cfg.allocation_fallback_after_seconds)
                    replica_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
                    replica_cfg.gpu = selected
                    replica_cfg.primary_arm = lane % len(session.members(cfg))
                    replica_cfg.rental_reservation_usd = remaining / count
                    expires = min(time.time() + seconds, deadline_value(live['deadline']))
                    print(f'Fleet slot {lane}: attempt {attempts[lane]} on {selected}', flush=True)
                    futures[lane] = pool.submit(replica, replica_cfg, config_path, next_slot, allowed, expires, manifest)
                    next_slot += 1
            if not futures and (live.get('halt') or not pending_tasks(live, ids, cfg)
                               or deadline_value(live['deadline']) - time.time() <= cfg.allocation_min_remaining_seconds):
                break
            if time.time() - last_upload >= cfg.upload_every_seconds:
                session.checkpoints(cfg, config_path)
                last_upload = time.time()
            if time.time() - last_reconcile >= cfg.allocation_reconcile_seconds:
                reconcile_rejections(cfg, manifest)
                last_reconcile = time.time()
            live = read(state.path)
            awaiting_reconciliation = any(p.get('id') is None and not p.get('reservation_released')
                and 'RunPod GraphQL rejected the request;' in p.get('error', '')
                for p in live['pods'])
            if (not futures and not awaiting_reconciliation and
                    manifest['budget_usd'] - reserved_cost(live) < price_ceiling(cfg, cfg.gpu) * cfg.allocation_min_remaining_seconds / 3600):
                with state.edit() as data:
                    data['halt'] = 'budget cannot fund another complete rental'
                break
            if time.time() > deadline_value(live['deadline']):
                with state.edit() as data:
                    data['halt'] = 'CPU lifetime exhausted'
            if shutil.disk_usage('/var/lib/docker').free / 2**30 < cfg.min_free_gib:
                with state.edit() as data:
                    data['halt'] = 'CPU disk reserve reached'
            if psutil.virtual_memory().available / 2**30 < cfg.min_available_memory_gib:
                with state.edit() as data:
                    data['halt'] = 'CPU memory reserve reached'
            time.sleep(2)
    session.checkpoints(cfg, config_path)  # A remote outage must not prevent local grading.


def grade(cfg):
    root = Path(cfg.root)
    state = read(root / 'metadata/state.json')
    manifest = read(root / 'metadata/manifest.json')
    assert not any(owned(p, session.owner_manifest(cfg)) for p in runpod.active_pods()), 'Release GPUs before grading'
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
    if cfg.get('fleet_owner_root') or cfg.get('following_configs'):
        result['budget_upper_bound_usd'] = None
        result['shared_fleet_accounting'] = 'metadata/final-accounting.json; shared session total, not additive per model'
    atomic(root / 'results/results.json', result)
    (root / 'results/results.md').write_text(
        f"SWE-bench Lite: {result['status']}\n\n"
        f"{len(resolved)}/300 resolved; {len(completed)}/300 graded.\n\n"
        f"Limitations: {manifest['limitations']}\n")
    with State(root).edit() as data:
        data['phase'] = result['status']
    print(json.dumps({k: v for k, v in result.items() if k not in ('task_results', 'resolved_ids')}, indent=2))


def execute(cfg, config_path, action, budget):
    if cfg.get('following_configs'):
        return session.execute(cfg, config_path, action, budget)
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
        reconcile_rejections(cfg, manifest)
        with state.edit() as data:
            data['halt'] = None
            data['breaker_failures_baseline'] = sum(a.get('valid') is False for t in data['tasks'].values() for a in t['attempts'])
            supervisor_path = root / 'metadata/supervisor.json'
            job_deadline = (read(supervisor_path)['deadline'] if supervisor_path.exists()
                            else receipt_deadline(read(cfg.receipt)))
            data['deadline'] = job_deadline - cfg.cpu_finish_reserve_seconds if job_deadline is not None else None
            assert deadline_value(data['deadline']) - time.time() >= cfg.allocation_min_remaining_seconds, 'Insufficient authorized time for a full task and cleanup'
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
                pilot = [r['instance_id'] for repo in cfg.pilot_repos
                         for r in [r for r in rows if r['repo'] == repo][:cfg.pilot_tasks_per_repo]]
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
                tool_tasks = []
                for iid in pilot:
                    attempt = data['tasks'][iid]['attempts'][-1]
                    traj = read(root / 'rollouts' / iid / attempt['id'] / iid / (iid + '.traj.json'))
                    if any(m.get('tool_calls') for m in traj.get('messages', []) if m.get('role') == 'assistant'):
                        tool_tasks.append(iid)
                assert tool_tasks, 'Pilot lacks valid tool calls across all cases'
                elapsed = time.time() - started
                # Use only this configuration's new attempts for throughput; historical
                # compatible outputs remain valid but had a different batch size.
                measured_ids = pending or pilot
                attempts = [data['tasks'][iid]['attempts'][-1] for iid in measured_ids]
                # Active attempt time excludes downtime between explicit resumes.
                gpu_seconds_per_task = sum(a['finished'] - a['started'] for a in attempts) / (len(attempts) * cfg.workers_per_replica)
                # Charge cold boot once PER replica, not once per six-task batch.
                boot_seconds = max(0, min(a['started'] for a in attempts) - started)
                proposed_replicas = cfg.replicas
                projected = ((300 - len(pilot)) * gpu_seconds_per_task + proposed_replicas * boot_seconds) / 3600 * cfg.max_hourly_usd
                atomic(root / 'metadata/calibration.json', {'tasks': pilot, 'elapsed_seconds': elapsed,
                       'gpu_seconds_per_task': gpu_seconds_per_task,
                       'measured_tasks': measured_ids, 'workers_per_replica': cfg.workers_per_replica,
                       'observed_batch_seconds': max(a['finished'] for a in attempts) - min(a['started'] for a in attempts),
                       'boot_seconds': boot_seconds, 'proposed_replicas': proposed_replicas,
                       'projected_remaining_gpu_usd': projected, 'estimate_only': True})
                assert projected + reserved_cost(data) <= manifest['budget_usd'], 'Pilot projects over budget; stop before fleet'
                grade(cfg)
                assert not read(root / 'results/results.json')['ungraded_ids'], 'Pilot grading failed'
                with state.edit() as data:
                    data['calibrated'] = True
                    data['phase'] = 'fleet'
                atomic(root / 'metadata/frozen_recipe.json', {'settings': recipe_settings(cfg),
                       'source_hashes': sources(), 'calibration': read(root / 'metadata/calibration.json'),
                       'validated_full_run': False, 'source_hf_repo': manifest['repo']})
            data = read(state.path)
            todo = [iid for iid, t in data['tasks'].items() if t['status'] != 'valid' and len(t['attempts']) < cfg.max_infrastructure_attempts]
            if todo:
                recipe = (read(root / 'metadata/frozen_recipe.json') if cfg.calibrate
                          else validate_recipe(cfg, read(cfg.recipe_path)))
                calibration = recipe['calibration']
                measured = (calibration['gpu_seconds_per_task'] * calibration['workers_per_replica']
                            / cfg.workers_per_replica) if calibration.get('gpu_seconds_per_task') is not None else None
                count = min(cfg.replicas, math.ceil(len(todo) / cfg.workers_per_replica))
                atomic(root / 'metadata/fleet_plan.json', {'replicas': count, 'ceiling': cfg.replicas,
                       'inference_seconds_estimate': len(todo) * measured / count if measured is not None else None, 'estimate_only': True,
                       'assumption': 'Historical mean task duration held constant; new GPU throughput unmeasured'})
                phase(cfg, config_path, todo, count, cfg.rental_seconds, manifest)
        finally:
            fence(cfg, manifest)
            checkpoint(cfg, config_path)
        try:
            grade(cfg)
            if read(root / 'results/results.json')['status'] == 'complete':
                recipe_path = root / 'metadata/frozen_recipe.json'
                recipe = read(recipe_path) if recipe_path.exists() else read(cfg.recipe_path)
                recipe['validated_full_run'] = True
                recipe['coverage'] = {'tasks': 300, 'graded': 300, 'hf_repo': manifest['repo']}
                recipe.setdefault('qualification', {})['performance_benchmark_validated'] = False
                # Complete coverage never silently certifies an unmeasured fleet.
                recipe['validation_hf_repo'] = manifest['repo']
                atomic(recipe_path, recipe)
                atomic(cfg.recipe_path, recipe)
        finally:
            checkpoint(cfg, config_path, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['qualify-shell', 'prepare', 'plan', 'run', 'resume', 'grade', 'publish', 'status', 'guard', 'supervise', 'launch', 'stop'])
    parser.add_argument('--config', default='configs/eval/swebench_mini/lite.yaml')
    parser.add_argument('--budget-usd', type=float)
    parser.add_argument('--target')
    parser.add_argument('--target-revision')
    parser.add_argument('--next-target')
    parser.add_argument('--next-target-revision')
    parser.add_argument('--root')
    parser.add_argument('--write-config')
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    cfg = OmegaConf.load(config_path)
    if cfg.get('fleet_owner_root') and args.action in ('run', 'resume', 'supervise', 'launch', 'stop'):
        raise ValueError('Shared-fleet child: use the owner launch.yaml at ' + cfg.fleet_owner_root)
    load_dotenv(cfg.credentials)
    os.chdir(REPO)
    if args.action in ('supervise', 'launch', 'stop'):
        from src.eval.capabilities.swebench_mini.fleet_supervisor import dispatch
        return dispatch(cfg, config_path, args)
    if args.action == 'qualify-shell':
        qualify_shell(cfg)
    elif args.action == 'prepare':
        prepare_target(cfg, args)
    elif args.action == 'plan':
        print(json.dumps(preflight(cfg), indent=2))
    elif args.action == 'publish':
        publish(cfg)
    elif args.action == 'guard':
        guard(cfg)
    elif args.action == 'status':
        data = session.view(cfg)
        print(json.dumps({'phase': data['phase'], 'halt': data['halt'], 'valid': sum(t['status'] == 'valid' for t in data['tasks'].values()),
                          'total': len(data['tasks']), 'pods': data['pods'], 'last_upload': data['last_upload'],
                          'cost_upper_bound_usd': reserved_cost(data),
                          'arms': [{'target': arm.target, 'root': arm.root,
                                    'valid': sum(t['status'] == 'valid' for t in read(State(arm.root).path)['tasks'].values()),
                                    'last_upload': read(State(arm.root).path)['last_upload']}
                                   for arm in session.members(cfg) if State(arm.root).path.exists()]}, indent=2))
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
