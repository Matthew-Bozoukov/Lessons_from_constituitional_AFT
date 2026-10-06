# ABOUTME: Offline queue tests cover durable submission, completion gates and failed-wave isolation.
# ABOUTME: All service, provider and Hugging Face boundaries are mocked; tests never rent or upload.
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
import httpx
import requests
from scratch.swebench_plain_campaign.queue import Queue, Runtime, launch_argv, read_only_call, sha


class ReadOnlyRetryTests(unittest.TestCase):
    def test_transient_read_recovers_with_bounded_attempts(self):
        operation = Mock(side_effect=[requests.ConnectionError('temporary'), requests.Timeout('temporary'), 'ok'])
        with patch('scratch.swebench_plain_campaign.queue.time.sleep') as sleep:
            self.assertEqual(read_only_call(operation), 'ok')
        self.assertEqual(operation.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])

    def test_exhausted_transient_read_raises_after_three_attempts(self):
        operation = Mock(side_effect=requests.ConnectionError('offline'))
        with patch('scratch.swebench_plain_campaign.queue.time.sleep'), self.assertRaises(requests.ConnectionError):
            read_only_call(operation)
        self.assertEqual(operation.call_count, 3)

    def test_assertions_auth_and_other_client_errors_never_retry(self):
        for error in [AssertionError('hash differs'), RuntimeError('auth missing')]:
            with self.subTest(error=error):
                operation = Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    read_only_call(operation)
                self.assertEqual(operation.call_count, 1)
        for status in (401, 403, 404):
            response = requests.Response()
            response.status_code = status
            operation = Mock(side_effect=requests.HTTPError(response=response))
            with self.assertRaises(requests.HTTPError):
                read_only_call(operation)
            self.assertEqual(operation.call_count, 1)

    def test_slow_first_failure_does_not_admit_another_retry(self):
        operation = Mock(side_effect=requests.Timeout('slow'))
        with patch('scratch.swebench_plain_campaign.queue.time.monotonic', side_effect=[0, 31]), self.assertRaises(requests.Timeout):
            read_only_call(operation)
        self.assertEqual(operation.call_count, 1)

    def test_httpx_server_error_is_retryable(self):
        response = httpx.Response(503, request=httpx.Request('GET', 'https://example.invalid'))
        error = httpx.HTTPStatusError('temporary', request=response.request, response=response)
        operation = Mock(side_effect=[error, 'ok'])
        with patch('scratch.swebench_plain_campaign.queue.time.sleep'):
            self.assertEqual(read_only_call(operation), 'ok')
        self.assertEqual(operation.call_count, 2)


class FakeRuntime:
    def __init__(self):
        self.service_state = dict(ActiveState='inactive', Result='success', ExecMainStatus='0', config=None)
        self.controls = {}
        self.launches = []
        self.verifications = []
        self.fail_verify = False

    def service(self):
        return self.service_state.copy()

    def inspect(self, wave):
        return self.controls.get(wave['root'], {})

    def launch(self, wave):
        self.launches.append(copy.deepcopy(wave))
        self.service_state.update(ActiveState='active', config=str(Path(wave['root']) / 'launch.yaml'))

    def verify(self, wave):
        self.verifications.append(wave['root'])
        if self.fail_verify:
            raise RuntimeError('HF mismatch or owned GPU')
        return {'arms': [dict(valid=300, graded=300)] * 2}


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = self.root / 'fleet.yaml'
        config.write_text('replicas: 12\nmax_replicas_per_arm: 6\n')
        self.plan = {'poll_seconds': 30, 'startup_grace_seconds': 120, 'waves': [
            {'root': str(self.root / f'wave{i}'), 'config': str(config), 'config_sha256': sha(config),
             'budget_usd': 360, 'targets': [{'repo': f'org/arm{i}{j}', 'revision': str(j + 1) * 40} for j in range(2)]}
            for i in range(2)]}
        self.state = {'status': 'running', 'waves': [{'status': 'pending'}, {'status': 'pending'}]}
        self.saved = []
        self.runtime = FakeRuntime()
        self.queue = Queue(self.plan, self.state, lambda v: self.saved.append(copy.deepcopy(v)), self.runtime)

    def complete_first(self):
        self.queue.step()
        self.runtime.service_state['ActiveState'] = 'inactive'
        self.runtime.controls[self.plan['waves'][0]['root']] = {'status': 'complete'}

    def test_submission_intent_is_durable_before_launch(self):
        original = self.runtime.launch
        def launch(wave):
            self.assertEqual(self.saved[-1]['waves'][0]['status'], 'submitting')
            original(wave)
        self.runtime.launch = launch
        self.assertEqual(self.queue.step(), 'waiting')
        self.assertEqual(len(self.runtime.launches), 1)
        self.queue.step()
        self.assertEqual(len(self.runtime.launches), 1)

    def test_wave_two_waits_for_inactive_then_rechecks_evidence(self):
        self.queue.step()
        self.runtime.controls[self.plan['waves'][0]['root']] = {'status': 'complete'}
        self.queue.step()
        self.assertEqual(len(self.runtime.launches), 1)
        self.runtime.service_state['ActiveState'] = 'inactive'
        self.queue.step()
        self.assertEqual(len(self.runtime.launches), 2)
        self.assertEqual(len(self.runtime.verifications), 2)
        self.assertEqual(self.state['waves'][0]['status'], 'complete')

    def test_hf_or_provider_failure_blocks_next_wave_permanently(self):
        self.complete_first()
        self.runtime.fail_verify = True
        self.assertEqual(self.queue.step(), 'blocked')
        self.runtime.fail_verify = False
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(len(self.runtime.launches), 1)

    def test_incomplete_or_failed_service_is_never_relaunched(self):
        self.queue.step()
        self.runtime.service_state.update(ActiveState='failed', Result='exit-code', ExecMainStatus='2')
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(len(self.runtime.launches), 1)

    def test_submission_network_error_is_never_retried(self):
        self.runtime.launch = Mock(side_effect=requests.Timeout('uncertain submission'))
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(self.runtime.launch.call_count, 1)

    def test_successful_exit_without_complete_wave_blocks(self):
        self.queue.step()
        self.runtime.service_state['ActiveState'] = 'inactive'
        self.runtime.controls[self.plan['waves'][0]['root']] = {'status': 'incomplete'}
        self.assertEqual(self.queue.step(), 'blocked')

    def test_reentrant_live_submission_is_adopted_without_launch(self):
        self.state['waves'][0].update(status='submitting', submitted_at=0)
        self.runtime.service_state.update(ActiveState='active', config=str(Path(self.plan['waves'][0]['root']) / 'launch.yaml'))
        self.queue.step()
        self.assertEqual(self.runtime.launches, [])
        self.assertEqual(self.state['waves'][0]['status'], 'submitted')

    def test_crash_before_any_launch_side_effect_can_replay_intent(self):
        self.state['waves'][0].update(status='submitting', submitted_at=0)
        self.assertEqual(self.queue.step(), 'waiting')
        self.assertEqual(len(self.runtime.launches), 1)

    def test_ambiguous_partial_launch_is_not_retried(self):
        root = Path(self.plan['waves'][0]['root'])
        root.mkdir()
        (root / 'launch.yaml').write_text('partial')
        self.state['waves'][0].update(status='submitting', submitted_at=0)
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(self.runtime.launches, [])

    def test_existing_ledger_never_reinitialized(self):
        (Path(self.plan['waves'][0]['root']) / 'metadata').mkdir(parents=True)
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(self.runtime.launches, [])

    def test_config_drift_blocks_launch(self):
        Path(self.plan['waves'][0]['config']).write_text('replicas: 99')
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(self.runtime.launches, [])

    def test_unrelated_live_service_is_not_replaced(self):
        self.runtime.service_state.update(ActiveState='active', config='/different/launch.yaml')
        self.assertEqual(self.queue.step(), 'blocked')
        self.assertEqual(self.runtime.launches, [])

    def test_completed_queue_is_idempotent(self):
        self.complete_first()
        self.queue.step()
        self.runtime.service_state['ActiveState'] = 'inactive'
        self.runtime.controls[self.plan['waves'][1]['root']] = {'status': 'complete'}
        self.assertEqual(self.queue.step(), 'complete')
        self.assertEqual(self.queue.step(), 'complete')
        self.assertEqual(len(self.runtime.launches), 2)

    def test_command_uses_one_paired_budget_and_external_config(self):
        argv = launch_argv(self.plan['waves'][0])
        self.assertEqual(argv.count('--budget-usd'), 1)
        self.assertEqual(argv[argv.index('--budget-usd') + 1], '360')
        self.assertIn('src.eval.run_eval', argv)
        self.assertIn('--next-target-revision', argv)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.cfg = OmegaConf.create({'root': str(root), 'target': 'org/a', 'target_revision': 'a' * 40,
                                     'credentials': str(root/'credentials.env')})
        self.child = OmegaConf.create({'root': str(root / 'next-arm'), 'fleet_owner_root': str(root),
                                      'target': 'org/b', 'target_revision': 'b' * 40})
        self.wave = {'root': str(root), 'budget_usd': 360, 'targets': [
            {'repo': c.target, 'revision': c.target_revision} for c in [self.cfg, self.child]]}
        self.paths = []
        for cfg in [self.cfg, self.child]:
            p = Path(cfg.root)
            (p / 'metadata').mkdir(parents=True)
            (p / 'results').mkdir()
            OmegaConf.save(cfg, p / 'launch.yaml')
            tasks = {str(i): {'status': 'valid'} for i in range(300)}
            self.write(p / 'metadata/state.json', {'tasks': tasks})
            self.write(p / 'metadata/manifest.json', {'config': OmegaConf.to_container(cfg), 'repo': cfg.target,
                'budget_usd': 360, 'campaign': cfg.target})
            self.write(p / 'metadata/supervisor.json', {'budget_usd': 360, 'status': 'complete',
                'verified_hf_revision': 'c' * 40, 'verified_files': 300})
            self.write(p / 'results/results.json', {'status': 'complete', 'n_total': 300, 'n_valid_rollouts': 300,
                'n_graded': 300, 'task_results': {k: {'graded': True, 'rollout_status': 'valid'} for k in tasks}})
            self.paths.append(p)
        self.fake = types.ModuleType('src.eval.capabilities.swebench_mini.fleet')
        self.fake.runpod = types.SimpleNamespace(active_pods=lambda: [])
        self.fake.owned = lambda pod, manifest: pod['campaign'] == manifest['campaign']
        self.fake.hf_download = lambda repo, *a, **k: self.paths[0 if repo == 'org/a' else 1] / 'results/results.json'
        self.fake.verify_final_files = lambda cfg, rev: 300
        self.session = types.ModuleType('src.eval.capabilities.swebench_mini.fleet_session')
        self.session.members = lambda cfg: [self.cfg, self.child]
        # `from package import module` may reuse package attributes established by
        # earlier fleet tests; replacing sys.modules alone is order-dependent.
        from src.eval.capabilities import swebench_mini
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.dict('sys.modules', {
            self.fake.__name__: self.fake, self.session.__name__: self.session}))
        self.patches.enter_context(patch.object(swebench_mini, 'fleet', self.fake, create=True))
        self.patches.enter_context(patch.object(swebench_mini, 'fleet_session', self.session, create=True))

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data))

    def mutate(self, relative, change):
        p = self.paths[1] / relative
        value = json.loads(p.read_text())
        change(value)
        self.write(p, value)

    def test_full_pair_verifies(self):
        self.assertEqual(len(Runtime().verify(self.wave)['arms']), 2)

    def test_credentials_loaded_before_external_reads(self):
        with patch('scratch.swebench_plain_campaign.queue.load_dotenv') as load:
            self.fake.hf_download = lambda repo, *a, **k: (
                self.assertEqual(load.call_args.args, (self.cfg.credentials,)) or
                self.paths[0 if repo == 'org/a' else 1] / 'results/results.json')
            Runtime().verify(self.wave)

    def test_provider_is_read_after_both_hf_verifications(self):
        events = []
        self.fake.verify_final_files = lambda cfg, rev: events.append('hf') or 300
        self.fake.runpod.active_pods = lambda: events.append('provider') or [{'campaign': 'org/a'}]
        with self.assertRaisesRegex(AssertionError, 'Campaign-owned GPUs remain'):
            Runtime().verify(self.wave)
        self.assertEqual(events, ['hf', 'hf', 'provider'])

    def test_incomplete_second_arm_rejected(self):
        self.mutate('results/results.json', lambda x: x.update(n_graded=299))
        with self.assertRaises(AssertionError):
            Runtime().verify(self.wave)

    def test_live_owned_gpu_rejected(self):
        self.fake.runpod.active_pods = lambda: [{'campaign': 'org/b'}]
        with self.assertRaises(AssertionError):
            Runtime().verify(self.wave)

    def test_unverified_hf_receipt_rejected(self):
        self.mutate('metadata/supervisor.json', lambda x: x.update(verified_files=0))
        with self.assertRaises(AssertionError):
            Runtime().verify(self.wave)

    def test_ledger_budget_mismatch_rejected(self):
        self.mutate('metadata/manifest.json', lambda x: x.update(budget_usd=720))
        with self.assertRaises(AssertionError):
            Runtime().verify(self.wave)


if __name__ == '__main__':
    unittest.main()
