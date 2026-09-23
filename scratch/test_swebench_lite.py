# ABOUTME: Offline checks for expensive failure paths, lease exclusivity, and immutable completed attempts.
# ABOUTME: Run on Linux: python -m unittest scratch.test_swebench_lite -v
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import shutil
import time
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from scratch.swebench_lite_state import State, atomic, classify, read, lock
from scratch import swebench_lite as fleet
from src.infra import runpod
from scratch.swebench_lite_task import install_token_limits


class TokenBudgetTests(unittest.TestCase):
    def model(self):
        class Stop(Exception):
            def __init__(self, *messages):
                self.messages = messages

        class Model:
            def _query(self, messages, **kwargs):
                return kwargs

            def _parse_actions(self, response):
                return ['execute']

        install_token_limits(Model, Stop, 10, 15)
        return Model(), Stop

    def response(self, used, reason='tool_calls'):
        return Mock(model_dump=lambda: {'usage': {'completion_tokens': used},
                    'choices': [{'finish_reason': reason, 'message': {'content': 'kept'}}]})

    def test_remaining_budget_bounds_next_request(self):
        model, stop = self.model()
        self.assertEqual(model._query([])['max_tokens'], 10)
        self.assertEqual(model._parse_actions(self.response(8)), ['execute'])
        self.assertEqual(model._query([])['max_tokens'], 7)
        with self.assertRaises(stop) as caught:
            model._parse_actions(self.response(7))
        self.assertEqual(caught.exception.messages[-1]['extra']['limit_reason'], 'task_token_limit')

    def test_truncation_keeps_response_and_exits_without_actions(self):
        model, stop = self.model()
        with self.assertRaises(stop) as caught:
            model._parse_actions(self.response(10, 'length'))
        messages = caught.exception.messages
        self.assertEqual(messages[0]['content'], 'kept')
        self.assertEqual(messages[0]['extra']['actions'], [])
        self.assertEqual(messages[-1]['extra']['submission'], '')
        self.assertTrue(classify({'info': {'exit_status': 'LimitsExceeded'}}, 0)['valid'])

    def test_model_budgets_are_not_shared_between_tasks(self):
        first, _ = self.model()
        second = first.__class__()
        first._parse_actions(self.response(8))
        self.assertEqual(second._query([])['max_tokens'], 10)

    def test_unvalidated_or_changed_recipe_rejected(self):
        cfg = OmegaConf.load('scratch/swebench_lite.yaml')
        recipe = {'validated_full_run': False, 'settings': fleet.recipe_settings(cfg), 'source_hashes': {}}
        with self.assertRaisesRegex(AssertionError, 'complete graded'):
            fleet.validate_recipe(cfg, recipe)
        recipe['validated_full_run'] = True
        with patch.object(fleet, 'sources', return_value={}):
            fleet.validate_recipe(cfg, recipe)
            cfg.workers_per_replica = 16
            with self.assertRaisesRegex(AssertionError, 'mismatch'):
                fleet.validate_recipe(cfg, recipe)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = State(self.tmp.name)
        atomic(self.state.path, {'tasks': {str(i): {'status': 'pending', 'attempts': []} for i in range(20)},
                                'deadline': time.time() + 600, 'last_upload': time.time(), 'halt': None})

    def claim(self, worker=0):
        return self.state.claim(str(worker), [str(i) for i in range(20)], 2, 6, 600)

    def test_concurrent_claims_are_exclusive(self):
        with ThreadPoolExecutor(max_workers=24) as pool:
            leases = list(pool.map(self.claim, range(24)))
        self.assertEqual(len({x[0] for x in leases if x}), 20)

    def test_completed_model_failure_never_retried(self):
        iid, aid = self.claim()
        self.state.finish(iid, aid, {'valid': True, 'exit_status': 'LimitsExceeded'})
        self.assertEqual(read(self.state.path)['tasks'][iid]['status'], 'valid')
        self.assertNotEqual(self.claim()[0], iid)

    def test_infrastructure_retry_is_bounded(self):
        for _ in range(2):
            iid, aid = self.claim()
            self.assertEqual(iid, '0')
            self.state.finish(iid, aid, {'valid': False, 'exit_status': 'ConnectionError'})
        self.assertEqual(self.claim()[0], '1')

    def test_crash_after_result_commit_does_not_repeat_generation(self):
        iid, aid = self.claim()
        atomic(self.state.root / 'rollouts' / iid / aid / 'done.json', {'valid': True, 'exit_status': 'Submitted'})
        self.state.recover()
        self.assertEqual(read(self.state.path)['tasks'][iid]['status'], 'valid')
        self.assertNotEqual(self.claim()[0], iid)

    def test_interrupted_attempt_retained(self):
        iid, aid = self.claim()
        self.state.recover()
        self.assertEqual(self.claim()[0], iid)
        self.assertEqual(len(read(self.state.path)['tasks'][iid]['attempts']), 2)

    def test_backup_outage_stops_claims(self):
        with self.state.edit() as data:
            data['last_upload'] = 0
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            self.claim()

    def test_explicit_resume_resets_breaker_but_retains_attempts(self):
        with self.state.edit() as data:
            for i in range(6):
                data['tasks'][str(i)] = {'status': 'invalid', 'attempts': [{'id': 'old-' + str(i), 'valid': False}]}
        self.assertIsNone(self.claim())
        with self.state.edit() as data:
            data['halt'] = None
            data['breaker_failures_baseline'] = 6
        self.assertEqual(self.claim()[0], '0')
        self.assertEqual(len(read(self.state.path)['tasks']['0']['attempts']), 2)

    def test_exit_status_classification(self):
        for status in ('Submitted', 'LimitsExceeded', 'ContextWindowExceededError'):
            self.assertTrue(classify({'info': {'exit_status': status}}, 0)['valid'])
        self.assertFalse(classify({'info': {'exit_status': 'Submitted'}}, 1)['valid'])
        self.assertFalse(classify({}, 0)['valid'])


class RentalTests(unittest.TestCase):
    def test_hf_snapshot_and_graded_publish_keep_same_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root, remote = Path(temp) / 'run', Path(temp) / 'hub'
            cfg = OmegaConf.load('scratch/swebench_lite.yaml')
            cfg.root = str(root)
            for sub in ('rollouts', 'results', 'metadata'):
                (root / sub).mkdir(parents=True)
            manifest = {'campaign': 'abc', 'repo': 'org/2026-09-23-swebench-qwen36-0-nosynth',
                        'date': '2026-09-23', 'model_key': 'qwen36_0_nosynth', 'protocol': 'test',
                        'limitations': 'test', 'config': OmegaConf.to_container(cfg)}
            atomic(root / 'metadata/manifest.json', manifest)
            atomic(root / 'metadata/state.json', {'last_upload': 0, 'tasks': {}})
            destinations = []

            def push(snapshot, repo, *a, **kw):
                destinations.append(repo)
                shutil.copytree(snapshot, remote, dirs_exist_ok=True)

            api = Mock()
            api.repo_exists.side_effect = lambda *a, **k: remote.exists()
            api.dataset_info.return_value.sha = 'immutable-commit'
            with patch.object(fleet, 'hf_api', return_value=api), \
                 patch.object(fleet, 'push_run_dir', side_effect=push), \
                 patch.object(fleet, 'hf_download', side_effect=lambda repo, name, **kw: str(remote / name)):
                fleet.publish(cfg)
                atomic(root / 'results/results.json', {'status': 'complete'})
                fleet.publish(cfg)
            self.assertEqual(destinations, [manifest['repo'], manifest['repo']])
            self.assertEqual(read(remote / 'results/results.json'), {'status': 'complete'})
            self.assertEqual(read(root / 'metadata/state.json')['hf_commit'], 'immutable-commit')

    def test_grader_cleanup_matches_full_instance_and_campaign(self):
        with tempfile.TemporaryDirectory() as temp:
            atomic(Path(temp) / 'metadata/state.json', {'tasks': {'django__django-1': {}}})
            ours = 'sweb.eval.django__django-1.lite_abc'
            with patch.object(fleet.subprocess, 'check_output', return_value=ours + '\nsweb.eval.django__django-2.lite_abc\nother\n'), \
                 patch.object(fleet.subprocess, 'run') as remove:
                fleet.cleanup_grading(temp, {'campaign': 'abc'})
            self.assertEqual(remove.call_args.args[0], ['docker', 'rm', '-f', ours])

    def test_reaper_does_not_interrupt_explicit_cpu_only_grading(self):
        with tempfile.TemporaryDirectory() as temp:
            atomic(Path(temp) / 'metadata/manifest.json', {'campaign': 'abc'})
            atomic(Path(temp) / 'metadata/state.json', {'deadline': time.time() + 600})
            with lock(Path(temp) / '.coordinator.lock'), \
                 patch.object(fleet.subprocess, 'run', return_value=Mock(returncode=3)), \
                 patch.object(runpod, 'active_pods', return_value=[]), \
                 patch.object(fleet, 'cleanup_grading') as cleanup:
                fleet.guard(OmegaConf.create({'root': temp}))
            cleanup.assert_not_called()

    def test_expiry_is_atomic_with_creation(self):
        with patch.object(runpod, 'graphql', return_value={'podFindAndDeployOnDemand': {'id': 'owned'}}) as gql:
            with patch.object(runpod, 'call', side_effect=AssertionError('No unscheduled REST rental')):
                result = runpod.provision_runpod(runpod.ProvisionSpec(terminate_at='2099-01-01T00:00:00Z'),
                                                name='nika-test', start_script='echo ok')
        self.assertEqual(result, 'owned')
        self.assertEqual(gql.call_args.args[1]['input']['terminateAfter'], '2099-01-01T00:00:00Z')

    def test_validation_cannot_allocate(self):
        with patch.object(runpod, 'graphql') as gql:
            runpod.validate_scheduled_provision()
        self.assertIn('@skip(if: true)', gql.call_args.args[0])

    def test_bootstrap_downloads_exact_weight_revisions(self):
        script = runpod._bootstrap(None, (['org/base', 'org/adapter'], None),
                                   weight_revisions={'org/base': 'a' * 40, 'org/adapter': 'b' * 40})
        self.assertIn('hf download org/base --revision ' + 'a' * 40, script)
        self.assertIn('hf download org/adapter --revision ' + 'b' * 40, script)

    def test_unrelated_pod_is_never_owned(self):
        self.assertFalse(fleet.owned({'name': 'nika-swe-lite-x', 'env': {'LASR_CAMPAIGN': 'abc'}}, {'campaign': 'abc'}))

    def test_ambiguous_allocation_reserves_whole_lifetime(self):
        self.assertEqual(fleet.reserved_cost({'pods': [{'created': 100, 'expires': 3700, 'ceiling_hourly': 5}]}), 5)

    def exercise_failure(self, watchdog_failure=False, cleanup_failure=False):
        with tempfile.TemporaryDirectory() as temp:
            cfg = OmegaConf.load('scratch/swebench_lite.yaml')
            cfg.root = temp
            manifest = {'campaign': 'only-ours'}
            atomic(Path(temp) / 'metadata/state.json', {'pods': [], 'halt': None})
            live = {'id': 'ours', 'name': 'nika-swe-lite-only-our-0', 'costPerHr': 4.59,
                    'env': {'LASR_POD_OWNER': runpod.POD_OWNER, 'LASR_CAMPAIGN': 'only-ours'}}
            guard = Mock()

            def allocate(*a, **kw):
                kw['on_provisioned']('ours')
                raise RuntimeError('SSH failed')

            with patch.dict('os.environ', {'USER_PREFIX': 'nika'}), \
                 patch.object(runpod, 'provision_eval_pod', side_effect=allocate), \
                 patch.object(runpod, 'active_pods', return_value=[live]), \
                 patch.object(runpod, 'start_watchdog', side_effect=RuntimeError('guard failed') if watchdog_failure else None,
                              return_value=guard), \
                 patch.object(runpod, 'teardown', side_effect=RuntimeError('delete failed') if cleanup_failure else None) as teardown:
                with self.assertRaises(RuntimeError):
                    fleet.replica(cfg, Path(temp) / 'config.yaml', 0, Path(temp) / 'allowed.json', time.time() + 600, manifest)
            teardown.assert_called_once_with('ours')
            if cleanup_failure or watchdog_failure:
                guard.terminate.assert_not_called()
            else:
                guard.terminate.assert_called_once()

    def test_boot_failure_terminates_owned_pod(self):
        self.exercise_failure()

    def test_watchdog_arm_failure_still_terminates_owned_pod(self):
        self.exercise_failure(watchdog_failure=True)

    def test_failed_teardown_leaves_watchdog_running(self):
        self.exercise_failure(cleanup_failure=True)


if __name__ == '__main__':
    unittest.main()
