# ABOUTME: Failure-path tests for durable Lite supervision and shared CPU admission, without paid infrastructure.
# ABOUTME: Linux tests exercise real process locks; providers, publication and inference are mocked explicitly.
import os
import unittest
if os.name != 'posix':
    raise unittest.SkipTest('Fleet coordinator runs on Linux')

from datetime import datetime, timezone, timedelta
import hashlib
import multiprocessing
import signal
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from concurrent.futures import Future
from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini import fleet, fleet_supervisor as supervisor
from src.eval.capabilities.swebench_mini.fleet_state import atomic, read, tool_slot
from src.eval.capabilities.swebench_mini.fleet_host import check_instance


def hold_slot(path, ready):
    with tool_slot(path, 1, min_available_gib=0):
        ready.set()
        time.sleep(30)


class AdmissionTests(unittest.TestCase):
    def test_cross_process_limit_and_crash_release(self):
        with tempfile.TemporaryDirectory() as path:
            ready = multiprocessing.Event()
            process = multiprocessing.Process(target=hold_slot, args=(path, ready))
            process.start()
            try:
                self.assertTrue(ready.wait(10))
                with self.assertRaises(TimeoutError):
                    with tool_slot(path, 1, wait_seconds=.1, min_available_gib=0):
                        self.fail('Two processes acquired one slot')
            finally:
                process.kill()
                process.join(5)
            with tool_slot(path, 1, wait_seconds=1, min_available_gib=0):
                pass

    def test_low_memory_never_admits_work(self):
        with tempfile.TemporaryDirectory() as path, patch(
                'src.eval.capabilities.swebench_mini.fleet_state.available_memory_bytes',
                return_value=0):
            with self.assertRaises(TimeoutError):
                with tool_slot(path, 4, wait_seconds=.1):
                    self.fail('Low-memory work admitted')


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create({'max_recovery_cycles': 4, 'max_infrastructure_attempts': 3})
        self.control = {'deadline': time.time()+30000, 'cycles': 0}
        self.state = {'tasks': {'a': {'status': 'valid', 'attempts': [{}]},
                                'b': {'status': 'invalid', 'attempts': [{}]}}}

    def test_valid_model_outcomes_are_never_retry_candidates(self):
        self.assertEqual(supervisor.decide(self.state, self.control, self.cfg), 'infer')
        self.state['tasks']['b']['status'] = 'valid'
        self.control['deadline'] = 0
        self.assertEqual(supervisor.decide(self.state, self.control, self.cfg), 'finish')

    def test_explicit_stop_overrides_recovery_and_completion(self):
        for reason in ('signal 15', 'user stop'):
            self.state['halt'] = reason
            self.assertEqual(supervisor.decide(self.state, self.control, self.cfg), 'stopped')
        self.state.pop('halt')
        self.control['cancelled'] = True
        self.assertEqual(supervisor.decide(self.state, self.control, self.cfg), 'stopped')

    def test_restart_cannot_refresh_cycle_or_time_allowances(self):
        with tempfile.TemporaryDirectory() as path:
            file = Path(path)/'control.json'
            self.control['cycles'] = 4
            atomic(file, self.control)
            self.assertEqual(supervisor.decide(self.state, read(file), self.cfg), 'recovery_exhausted')
            self.control.update(cycles=0, deadline=time.time()+100)
            atomic(file, self.control)
            self.assertEqual(supervisor.decide(self.state, read(file), self.cfg), 'cpu_lifetime_insufficient')

    def test_resource_halt_and_task_attempt_exhaustion_are_bounded(self):
        self.state['halt'] = 'CPU memory reserve reached'
        self.assertEqual(supervisor.decide(self.state, self.control, self.cfg), 'needs_attention')
        self.state['halt'] = None
        self.state['tasks']['b']['attempts'] = [{}, {}, {}]
        self.assertEqual(supervisor.decide(self.state, self.control, self.cfg), 'attempts_exhausted')

    def test_publication_failure_retries_without_inference_or_ledger_reset(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            atomic(root/'metadata/supervisor.json', self.control)
            atomic(root/'metadata/state.json', {'hf_commit': 'abc', 'pods': [{'charged': 5}]})
            atomic(root/'metadata/manifest.json', {'repo': 'org/results', 'campaign': 'test'})
            atomic(root/'results/results.json', {'status': 'complete', 'n_graded': 300})
            cfg = OmegaConf.create({'root': path, 'publication_attempts': 2, 'publication_backoff_seconds': 0})
            with patch.object(fleet, 'publish', side_effect=[RuntimeError('Hub down'), None, None]) as upload, \
                 patch.object(fleet, 'execute') as inference, \
                 patch.object(fleet, 'hf_download', return_value=root/'results/results.json'), \
                 patch.object(fleet, 'verify_final_files', return_value=900), \
                 patch.object(fleet.runpod, 'active_pods', return_value=[]):
                self.assertTrue(supervisor.publish_until_verified(cfg, root/'metadata/supervisor.json'))
                self.assertEqual(upload.call_count, 3)
                inference.assert_not_called()
            self.assertEqual(read(root/'metadata/state.json')['pods'], [{'charged': 5}])
            self.assertEqual(read(root/'metadata/supervisor.json')['verified_files'], 900)

    def test_exhausted_publication_retains_pending_state(self):
        with tempfile.TemporaryDirectory() as path:
            control = Path(path)/'supervisor.json'
            atomic(control, self.control)
            cfg = OmegaConf.create({'publication_attempts': 2, 'publication_backoff_seconds': 0})
            with patch.object(fleet, 'publish', side_effect=RuntimeError('Hub down')) as publish:
                self.assertFalse(supervisor.publish_until_verified(cfg, control))
                self.assertEqual(publish.call_count, 2)
            self.assertEqual(read(control)['cycles'], 0)
            self.assertEqual(read(control)['status'], 'publication_retry')

    def test_supervisor_recovers_crash_without_repeating_completed_task(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            atomic(root/'receipt.json', {'stop_at': (datetime.now(timezone.utc)+timedelta(hours=10)).isoformat()})
            cfg = OmegaConf.create({'root': path, 'receipt': str(root/'receipt.json'),
                'recommended_budget_usd': 180, 'max_recovery_cycles': 4,
                'max_infrastructure_attempts': 3, 'recovery_backoff_seconds': 0})
            calls = []
            def execute(config, config_path, action, budget):
                calls.append((action, budget))
                if len(calls) == 1:
                    atomic(root/'metadata/manifest.json', {'campaign': 'test'})
                    atomic(root/'metadata/state.json', self.state)
                    raise RuntimeError('Synthetic coordinator failure after task a committed')
                state = read(root/'metadata/state.json')
                self.assertEqual(state['tasks']['a'], self.state['tasks']['a'])
                state['tasks']['b']['status'] = 'valid'
                atomic(root/'metadata/state.json', state)
            def grade(config):
                atomic(root/'results/results.json', {'status': 'complete'})
            def publish(config, path):
                atomic(path, read(path) | {'status': 'complete'})
                return True
            old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                with patch.object(fleet, 'execute', side_effect=execute), patch.object(fleet, 'fence'), \
                     patch.object(fleet, 'grade', side_effect=grade), patch.object(fleet, 'final_accounting'), \
                     patch.object(supervisor, 'publish_until_verified', side_effect=publish):
                    supervisor.supervise(cfg, root/'config.yaml', 180)
            finally:
                for signum, handler in old.items():
                    signal.signal(signum, handler)
            self.assertEqual(calls, [('run', 180), ('resume', 180)])
            self.assertEqual(read(root/'metadata/supervisor.json')['cycles'], 2)

    def test_production_timeline_recovers_after_first_batch_and_after_six_hours(self):
        """Reproduce the 02:15 launch / 04:40 partial finish using production limits."""
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
            cfg.root, cfg.receipt = path, str(root/'receipt.json')
            start = datetime.fromisoformat('2026-09-24T02:15:42+00:00').timestamp()
            cpu_stop = datetime.fromisoformat('2026-09-24T12:37:16+00:00').timestamp()
            atomic(cfg.receipt, {'stop_at': datetime.fromtimestamp(cpu_stop, timezone.utc).isoformat()})
            clock = [start]
            tasks = {str(n): {'status': 'valid' if n < 261 else 'pending',
                             'attempts': [{'prediction': n}] if n < 261 else []} for n in range(300)}
            calls = []
            def execute(config, config_path, action, budget):
                calls.append(action)
                if len(calls) == 1:
                    atomic(root/'metadata/manifest.json', {'campaign': 'test'})
                    atomic(root/'metadata/state.json', {'tasks': tasks, 'pods': [{'charged': 62.05}]})
                    clock[0] = start + 2*3600 + 25*60
                else:
                    state = read(root/'metadata/state.json')
                    self.assertEqual(state['tasks'], tasks)
                    self.assertEqual(state['pods'], [{'charged': 62.05}])
                    self.assertEqual(read(root/'metadata/supervisor.json')['deadline'], cpu_stop-120)
                    # The old six-hour job expiry must not reappear either.
                    self.assertEqual(supervisor.decide(state, read(root/'metadata/supervisor.json'),
                                     cfg, now=start+6*3600+15*60), 'infer')
                    for task in state['tasks'].values():
                        task['status'] = 'valid'
                    atomic(root/'metadata/state.json', state)
                    clock[0] += 3600
            def grade(config):
                atomic(root/'results/results.json', {'status': 'complete'})
            def publish(config, control):
                atomic(control, read(control) | {'status': 'complete'})
                return True
            old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                with patch.object(supervisor.time, 'time', side_effect=lambda: clock[0]), \
                     patch.object(supervisor.time, 'sleep'), patch.object(fleet, 'execute', side_effect=execute), \
                     patch.object(fleet, 'fence'), patch.object(fleet, 'grade', side_effect=grade), \
                     patch.object(fleet, 'final_accounting'), \
                     patch.object(supervisor, 'publish_until_verified', side_effect=publish):
                    supervisor.supervise(cfg, root/'config.yaml', 180)
            finally:
                for signum, handler in old.items():
                    signal.signal(signum, handler)
            self.assertEqual(calls, ['run', 'resume'])

    def test_terminal_incomplete_run_exits_nonzero_and_preserves_reason(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            control = self.control | {'cycles': 4, 'budget_usd': 180}
            atomic(root/'metadata/supervisor.json', control)
            atomic(root/'metadata/state.json', self.state)
            atomic(root/'metadata/manifest.json', {'campaign': 'test'})
            cfg = OmegaConf.create({'root': path, 'max_recovery_cycles': 4})
            def publish(config, file):
                atomic(file, read(file) | {'status': 'incomplete'})
                return True
            old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                with patch.object(fleet, 'fence'), patch.object(fleet, 'grade'), \
                     patch.object(fleet, 'final_accounting'), \
                     patch.object(supervisor, 'publish_until_verified', side_effect=publish):
                    with self.assertRaises(SystemExit) as raised:
                        supervisor.supervise(cfg, root/'config.yaml', 180)
            finally:
                for signum, handler in old.items():
                    signal.signal(signum, handler)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(read(root/'metadata/supervisor.json')['terminal_reason'], 'recovery_exhausted')

    def test_drained_replica_replaced_without_waiting_for_shared_batch_cutoff(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
            cfg.root = path
            start, clock, expiries = 100000., [100000.], []
            state_path = root/'metadata/state.json'
            atomic(state_path, {'deadline': start+8*3600, 'pods': [], 'halt': None,
                'tasks': {'a': {'status': 'pending', 'attempts': []},
                          'b': {'status': 'pending', 'attempts': []}}})
            def submit(fn, config, config_path, slot, allowed, expires, manifest):
                expiries.append(expires)
                state = read(state_path)
                state['tasks']['a' if len(expiries) == 1 else 'b']['status'] = 'valid'
                atomic(state_path, state)
                # First pod drains beyond the original 57-minute admission window.
                clock[0] += 100*60
                result = Future()
                result.set_result(None)
                return result
            pool = Mock()
            pool.submit.side_effect = submit
            with patch.object(fleet, 'ThreadPoolExecutor') as executor, \
                 patch.object(fleet.time, 'time', side_effect=lambda: clock[0]), \
                 patch.object(fleet.time, 'sleep'), patch.object(fleet, 'price_ceiling', return_value=3.35), \
                 patch.object(fleet, 'checkpoint'), patch.object(fleet, 'reconcile_rejections'), \
                 patch.object(fleet.shutil, 'disk_usage', return_value=Mock(free=200*2**30)), \
                 patch.object(fleet.psutil, 'virtual_memory', return_value=Mock(available=180*2**30)):
                executor.return_value.__enter__.return_value = pool
                fleet.phase(cfg, root/'config.yaml', ['a', 'b'], 1, cfg.rental_seconds, {'budget_usd': 180})
            self.assertEqual(len(expiries), 2)
            self.assertEqual(expiries[0], start+cfg.rental_seconds)
            self.assertEqual(expiries[1], start+100*60+cfg.rental_seconds)


class ProvenanceTests(unittest.TestCase):
    def test_unused_bootstrap_tears_down_without_starting_worker(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
            cfg.root = path
            manifest = {'campaign': 'only-ours', 'budget_usd': 100}
            task = {'status': 'running', 'attempts': [{'worker': '99-0'}]}
            atomic(root/'metadata/state.json', {'pods': [], 'halt': None, 'tasks': {'a': task}})
            atomic(root/'allowed.json', ['a'])
            live = {'id': 'ours', 'name': 'nika-swe-lite-only-our-0', 'costPerHr': 3.19,
                    'env': {'LASR_POD_OWNER': fleet.runpod.POD_OWNER, 'LASR_CAMPAIGN': 'only-ours'}}
            guard = Mock()
            def allocate(*args, **kwargs):
                kwargs['on_provisioned']('ours')
                return SimpleNamespace(id='ours', reachable=True)
            with patch.dict(os.environ, {'USER_PREFIX': 'nika'}), \
                 patch.object(fleet, 'price_ceiling', return_value=3.35), \
                 patch.object(fleet.runpod, 'provision_eval_pod', side_effect=allocate), \
                 patch.object(fleet.runpod, 'active_pods', return_value=[live]), \
                 patch.object(fleet.runpod, 'start_watchdog', return_value=guard), \
                 patch.object(fleet.runpod, 'wait_bootstrapped', return_value=True), \
                 patch.object(fleet.runpod, 'teardown') as teardown, \
                 patch.object(fleet.subprocess, 'Popen') as worker:
                fleet.replica(cfg, root/'config.yaml', 0, root/'allowed.json', time.time()+9000, manifest)
            worker.assert_not_called()
            teardown.assert_called_once_with('ours')
            guard.terminate.assert_called_once()
            state = read(root/'metadata/state.json')
            self.assertEqual(state['tasks']['a'], task)
            self.assertEqual(state['pods'][0]['status'], 'terminated')
            self.assertIn('idle_startup_cancellation', state['pods'][0])

    def test_startup_cancellation_serializes_with_ready_worker(self):
        from src.eval.capabilities.swebench_mini.fleet_state import State
        from src.eval.capabilities.swebench_mini import fleet_worker
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
            cfg.root = path
            OmegaConf.save(cfg, root/'config.yaml')
            state = State(root)
            atomic(state.path, {'pods': [{'slot': 1, 'ready_at': time.time()}],
                               'tasks': {'a': {'status': 'valid', 'attempts': [{}]}}})
            self.assertFalse(fleet.cancel_idle_startup(state, 1, ['a'], cfg))
            with state.edit() as data:
                data['pods'][0].pop('ready_at')
            self.assertTrue(fleet.cancel_idle_startup(state, 1, ['a'], cfg))
            target = SimpleNamespace(spec=SimpleNamespace(revision=cfg.target_revision,
                base_revision=cfg.base_revision, mode=cfg.mode), base_url='http://unused')
            worker_cfg = OmegaConf.create({'campaign_config': str(root/'config.yaml'), 'replica': 1})
            with patch.object(fleet_worker, 'ThreadPoolExecutor') as pool:
                result = fleet_worker.runner(target, worker_cfg, root)
            pool.assert_not_called()
            self.assertIn('cancelled', result['status'])
            self.assertNotIn('ready_at', read(state.path)['pods'][0])

    def test_accounting_filters_shared_account_and_retains_missing_bills(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            atomic(root/'metadata/state.json', {'pods': [
                {'id': 'ours', 'created': 100, 'ended': 3700, 'expires': 4000, 'ceiling_hourly': 4},
                {'id': 'lagged', 'created': 100, 'ended': 1900, 'expires': 4000, 'ceiling_hourly': 4}]})
            atomic(root/'metadata/manifest.json', {'campaign': 'test', 'created': datetime.now(timezone.utc).isoformat()})
            atomic(root/'receipt.json', {'offer': {'dph_total': .54}})
            rows = [{'podId': 'ours', 'amount': 3.5}, {'podId': 'teammate', 'amount': 99}]
            with patch.object(fleet.runpod, 'active_pods', return_value=[]), patch.object(fleet.runpod, 'call', return_value=rows):
                result = fleet.final_accounting(OmegaConf.create({'root': path, 'receipt': str(root/'receipt.json')}))
            self.assertEqual(result['gpu_conservative_ledger_usd'], 6)
            self.assertEqual(result['provider_gpu_recorded_usd'], 3.5)
            self.assertEqual(result['provider_missing_pod_ids'], ['lagged'])

    def test_boot_stages_share_one_deadline_and_watchdog_arms_first(self):
        pod = fleet.runpod
        events = []
        def endpoint(identifier, timeout_s):
            events.append(('endpoint', timeout_s))
            return '127.0.0.1', 22
        with patch.object(pod, 'plan_eval_pod', return_value=(['org/lora'], (['org/lora'], None), 'gpu', 200, None)), \
             patch.object(pod, '_bootstrap', return_value='true'), patch.object(pod, '_check_bash'), \
             patch.object(pod, 'provision_runpod', return_value='synthetic'), \
             patch.object(pod, '_ssh_endpoint', side_effect=endpoint), \
             patch.object(pod, '_wait_for_ssh', return_value=True) as ssh, \
             patch.object(pod.time, 'time', side_effect=[100, 140]):
            pod.provision_eval_pod('org/lora', name='test', boot_deadline=160,
                                  on_provisioned=lambda _: events.append(('watchdog',)))
            self.assertEqual(events, [('watchdog',), ('endpoint', 60)])
            self.assertEqual(ssh.call_args.kwargs['timeout_s'], 20)

    def test_single_entrypoint_dispatches_without_renting_on_caller(self):
        from src.eval.run_eval import main
        with patch.object(fleet, 'main') as launch, patch.object(fleet.runpod, 'provision_eval_pod') as rent:
            main(['--name', 'swebench_mini', '--target', 'org/lora', '--fleet', '--budget-usd', '180'])
            launch.assert_called_once_with(['launch', '--config', 'configs/eval/swebench_mini/lite.yaml',
                                           '--target', 'org/lora', '--budget-usd', '180.0'])
            rent.assert_not_called()

    def test_multiple_targets_and_missing_budget_are_refused(self):
        from src.eval.run_eval import main
        with patch.object(fleet, 'main') as launch:
            with self.assertRaises(SystemExit):
                main(['--name', 'swebench_mini', '--target', 'org/a', 'org/b', '--fleet', '--budget-usd', '180'])
            with self.assertRaises(SystemExit):
                main(['--name', 'swebench_mini', '--target', 'org/a', '--fleet'])
            launch.assert_not_called()

    def test_final_artifact_verification_detects_missing_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            atomic(root/'metadata/manifest.json', {'repo': 'org/results'})
            atomic(root/'results/results.json', {'n': 300})
            (root/'rollouts').mkdir()
            payload = (root/'results/results.json').read_bytes()
            f = SimpleNamespace(path='results/results.json', lfs=None,
                blob_id=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest())
            api = Mock()
            api.list_repo_tree.return_value = [f]
            with patch.object(fleet, 'hf_api', return_value=api):
                self.assertEqual(fleet.verify_final_files(OmegaConf.create({'root': path}), 'rev'), 1)
                atomic(root/'results/results.json', {'n': 299})
                with self.assertRaisesRegex(AssertionError, 'hash mismatch'):
                    fleet.verify_final_files(OmegaConf.create({'root': path}), 'rev')
                api.list_repo_tree.return_value = []
                with self.assertRaisesRegex(AssertionError, 'Missing final'):
                    fleet.verify_final_files(OmegaConf.create({'root': path}), 'rev')

    def test_cpu_resume_refuses_expiry_or_wrong_owner(self):
        now = datetime.now(timezone.utc)
        receipt = {'instance_id': 123, 'label': 'nika-test', 'stop_at': (now+timedelta(hours=6)).isoformat()}
        host = {'id': 123, 'label': 'nika-test', 'actual_status': 'stopped'}
        check_instance(receipt, host, now)
        with self.assertRaisesRegex(AssertionError, 'ownership'):
            check_instance(receipt, host | {'label': 'someone-else'}, now)
        with self.assertRaisesRegex(AssertionError, 'lifetime'):
            check_instance(receipt, host, now+timedelta(hours=6))


if __name__ == '__main__':
    unittest.main()


class PersistentLifetimeTests(unittest.TestCase):
    def test_unlimited_requires_explicit_authorization(self):
        from src.eval.capabilities.swebench_mini.fleet_host import receipt_deadline
        for receipt in ({}, {'stop_at': None}, {'lifetime': 'persistent'},
                        {'lifetime': 'persistent', 'lifetime_authorized_at': 'now', 'stop_at': 'old'}):
            with self.assertRaises(AssertionError):
                receipt_deadline(receipt)
        self.assertIsNone(receipt_deadline({'lifetime': 'persistent', 'lifetime_authorized_at': 'now', 'stop_at': None}))

    def test_persistent_still_has_boot_watchdog(self):
        from src.eval.capabilities.swebench_mini.fleet_host import watchdog_deadline
        receipt = {'lifetime': 'persistent', 'lifetime_authorized_at': 'now', 'stop_at': None,
                   'boot_deadline': '2026-09-24T12:00:00+00:00', 'ssh_verified': False}
        self.assertIsInstance(watchdog_deadline(receipt), float)
        receipt['ssh_verified'] = True
        self.assertIsNone(watchdog_deadline(receipt))

    def test_persistent_keeps_task_and_gpu_timeouts(self):
        from src.eval.capabilities.swebench_mini.fleet_worker import attempt_deadline
        cfg = OmegaConf.create({'task_seconds': 5400, 'cleanup_reserve_seconds': 180})
        now = time.time()
        self.assertEqual(attempt_deadline(cfg, {'deadline': None}, now+300), now+120)
        control = {'deadline': None, 'cycles': 0}
        state = {'tasks': {'a': {'status': 'pending', 'attempts': []}}}
        cfg.max_infrastructure_attempts = 3
        self.assertEqual(supervisor.decide(state, control, cfg, now=10**12), 'infer')
        control['cycles'] = 4
        self.assertEqual(supervisor.decide(state, control, cfg), 'recovery_exhausted')

    def test_registry_does_not_replace_another_host(self):
        from scratch.swebench_cpu_manage import register
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            receipt = d/'receipt.json'
            key = d/'key'
            key.touch()
            atomic(receipt, {'instance_id': 1, 'stop_at': '2026-09-24T12:00:00+00:00'})
            register(d/'registry.json', receipt, key)
            atomic(receipt, {'instance_id': 2, 'stop_at': '2026-09-24T12:00:00+00:00'})
            with self.assertRaises(AssertionError):
                register(d/'registry.json', receipt, key)


class HostRoutingTests(unittest.TestCase):
    def test_vm_prefers_fresh_direct_mapping_over_broken_proxy(self):
        from src.eval.capabilities.swebench_mini.fleet_host import ssh_endpoint
        self.assertEqual(ssh_endpoint({'public_ipaddr': '1.2.3.4', 'ports': {'22/tcp': [{'HostPort': '42'}]},
                                      'ssh_host': 'proxy', 'ssh_port': 10}), ('1.2.3.4', 42))
        self.assertEqual(ssh_endpoint({'ssh_host': 'proxy', 'ssh_port': 10}), ('proxy', 10))

    def test_first_fallback_is_h200_regardless_of_prior_primary_failures(self):
        cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
        for failures in (5, 6, 7, 25):
            self.assertEqual(fleet.allocation_gpu(cfg, failures, 601, 0), cfg.fallback_gpus[0])
        self.assertEqual(fleet.allocation_gpu(cfg, 26, 650, 1), cfg.fallback_gpus[1])
