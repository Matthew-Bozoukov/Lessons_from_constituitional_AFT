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
            self.assertEqual(supervisor.decide(self.state, read(file), self.cfg), 'inference_window_closed')

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
            old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                with patch.object(fleet, 'execute', side_effect=execute), patch.object(fleet, 'fence'), \
                     patch.object(fleet, 'grade', side_effect=grade), patch.object(fleet, 'final_accounting'), \
                     patch.object(supervisor, 'publish_until_verified', return_value=True):
                    supervisor.supervise(cfg, root/'config.yaml', 180)
            finally:
                for signum, handler in old.items():
                    signal.signal(signum, handler)
            self.assertEqual(calls, [('run', 180), ('resume', 180)])
            self.assertEqual(read(root/'metadata/supervisor.json')['cycles'], 2)


class ProvenanceTests(unittest.TestCase):
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
            check_instance(receipt, host, now+timedelta(hours=2))


if __name__ == '__main__':
    unittest.main()
