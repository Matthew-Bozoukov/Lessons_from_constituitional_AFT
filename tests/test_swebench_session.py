# ABOUTME: GPU-free paired fleet tests cover isolation, draining, leases and publication failures.
# ABOUTME: Provider/inference boundaries are mocked; real file locks and worker threads are exercised.
import os
import unittest
if os.name != 'posix':
    raise unittest.SkipTest('Fleet uses Linux process locks')

from pathlib import Path
import tempfile
import threading
import time
import sys
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch, Mock

from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini import fleet, fleet_session as session, fleet_worker as worker
from src.eval.capabilities.swebench_mini.fleet_state import State, atomic, read


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = OmegaConf.load('configs/eval/swebench_mini/lite.yaml')
        self.cfg.root = str(self.root)
        self.cfg.target = 'org/first'
        child = OmegaConf.create(OmegaConf.to_container(self.cfg))
        child.root, child.target = str(self.root/'next-arm'), 'org/second'
        child.fleet_owner_root = self.cfg.root
        Path(child.root).mkdir()
        self.child_path = str(Path(child.root)/'launch.yaml')
        OmegaConf.save(child, self.child_path)
        self.cfg.following_configs = [self.child_path]
        self.path = self.root/'launch.yaml'
        OmegaConf.save(self.cfg, self.path)
        for arm in session.members(self.cfg):
            atomic(Path(arm.root)/'metadata/state.json', {'tasks': {'same-id': {'status': 'pending', 'attempts': []}},
                'pods': [], 'deadline': None, 'halt': None, 'last_upload': 0})
            atomic(Path(arm.root)/'metadata/manifest.json', {'campaign': arm.target, 'repo': arm.target})

    def test_duplicate_task_ids_are_separate_and_view_is_idempotent(self):
        data = session.view(self.cfg)
        self.assertEqual(set(data['tasks']), {'0:same-id', '1:same-id'})
        self.assertEqual(session.view(self.cfg, data), data)
        self.assertTrue(fleet.pending_tasks(data, ['1:same-id'], self.cfg))
        self.assertEqual(session.budget_limit(self.cfg), 360)
        with State(self.cfg.root).edit() as state:
            state['tasks']['same-id']['status'] = 'valid'
        self.assertEqual(session.view(self.cfg)['tasks']['1:same-id']['status'], 'pending')

    def test_child_halt_and_missing_initialization_cannot_look_complete(self):
        child = session.members(self.cfg)[1]
        with State(child.root).edit() as data:
            data['halt'] = 'CPU memory reserve reached'
        self.assertEqual(session.view(self.cfg)['halt'], 'CPU memory reserve reached')
        State(child.root).path.unlink()
        self.assertEqual(session.view(self.cfg)['tasks']['1:uninitialized']['status'], 'pending')

    def test_no_task_wall_clock_cap_but_emergency_cleanup_remains(self):
        now = time.time()
        self.assertEqual(worker.attempt_deadline(self.cfg, {'deadline': None}, now+21600), now+21420)
        self.assertEqual(worker.latest_admission(self.cfg, now+21600), now+14220)
        self.cfg.task_seconds = 5400
        self.assertLess(worker.attempt_deadline(self.cfg, {'deadline': None}, now+21600), now+5410)

    def test_one_ledger_and_shared_tool_directory(self):
        arms = session.members(self.cfg)
        self.assertEqual(session.owner_manifest(arms[1]), session.owner_manifest(arms[0]))
        self.assertEqual(Path(arms[1].fleet_owner_root)/'.tool-slots', self.root/'.tool-slots')

    def test_one_pod_drains_before_switching_while_other_pod_still_runs(self):
        with State(self.cfg.root).edit() as data:
            data['pods'] = [{'slot': 0}, {'slot': 1, 'ready_at': time.time(), 'status': 'working'}]
        config = OmegaConf.create({'campaign_config': str(self.path), 'replica': 0,
                    'allowed': ['0:same-id', '1:same-id'], 'expires': time.time()+21600})
        entered = threading.Barrier(5)
        release = threading.Event()
        errors, seen = [], []
        def consume(endpoint, model, cfg, worker_id, allowed, expires, unhealthy, admission=None):
            seen.append((cfg.target, worker_id, list(allowed)))
            if cfg.target == 'org/first':
                entered.wait(5)
                self.assertTrue(release.wait(5))
        def observe(*args, **kwargs):
            return SimpleNamespace(text='', raise_for_status=lambda: None)
        def target(arm):
            return SimpleNamespace(spec=SimpleNamespace(hf_path=arm.target, revision=arm.target_revision,
                base_revision=arm.base_revision, mode=arm.mode), base_url='http://synthetic/v1', model_name=arm.target,
                _server=SimpleNamespace(executor=SimpleNamespace(tail_log=lambda n: 'GPU KV cache size: 503,949 tokens')))
        def run_pair():
            try:
                for arm in session.members(self.cfg):
                    worker.runner(target(arm), config, self.root)
            except Exception as exc:
                errors.append(exc)
        with patch.object(worker, 'consume', side_effect=consume), patch.object(worker.requests, 'get', side_effect=observe):
            thread = threading.Thread(target=run_pair)
            thread.start()
            entered.wait(5)
            self.assertEqual(len(seen), 4)
            self.assertTrue(all(a[0] == 'org/first' for a in seen))
            release.set()
            thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([x[0] for x in seen], ['org/first']*4 + ['org/second']*4)
        self.assertTrue(all(x[2] == ['same-id'] for x in seen))
        self.assertEqual(read(State(self.cfg.root).path)['pods'][1]['status'], 'working')

    def test_cannot_publish_session_complete_if_one_arm_fails(self):
        from src.eval.capabilities.swebench_mini import fleet_supervisor as supervisor
        control_path = self.root/'metadata/supervisor.json'
        atomic(control_path, {'deadline': None, 'status': 'running'})
        for arm in session.members(self.cfg):
            atomic(Path(arm.root)/'results/results.json', {'status': 'complete'})
        def verify(arm, path):
            atomic(path, {'status': 'complete' if arm.target == 'org/first' else 'publication_pending'})
            return arm.target == 'org/first'
        with patch.object(session, 'fence_all'), patch.object(session, 'accounting'), \
                patch.object(supervisor, 'publish_until_verified', side_effect=verify), patch.object(fleet, 'publish'):
            self.assertFalse(session.finish(self.cfg, control_path))
        self.assertEqual(read(control_path)['status'], 'incomplete')

    def test_pair_entrypoint_preserves_both_revision_pins(self):
        from src.eval.run_eval import main
        with patch.object(fleet, 'main') as launch:
            main(['--name', 'swebench_mini', '--fleet', '--target', 'org/a', 'org/b',
                  '--target-revision', 'sha-a', '--next-target-revision', 'sha-b', '--budget-usd', '360'])
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--next-target')+1], 'org/b')
        self.assertEqual(command[command.index('--next-target-revision')+1], 'sha-b')

    def test_recovery_promotes_child_checkpoint_without_rerolling_valid_outcome(self):
        child = session.members(self.cfg)[1]
        state = State(child.root)
        iid, aid = state.claim('7-0', ['same-id'], 3, 6)
        atomic(Path(child.root)/'rollouts'/iid/aid/'done.json', {'valid': True, 'exit_status': 'LimitsExceeded'})
        with patch('subprocess.check_output', return_value=''):
            session.recover_worker(self.cfg, 7)
        self.assertEqual(read(state.path)['tasks'][iid]['status'], 'valid')
        self.assertIsNone(state.claim('8-0', [iid], 3, 6))
        self.assertEqual(read(State(self.cfg.root).path)['tasks'][iid]['status'], 'pending')

    def test_shared_phase_runs_both_queues_without_renting_twice(self):
        # Production coordinator and durable state; only the provider/agent and HF
        # boundaries are substituted. The two arms deliberately share instance IDs.
        self.cfg.replicas = 2
        self.cfg.allocation_retry_seconds = 0
        arms = session.members(self.cfg)
        for arm in arms:
            with State(arm.root).edit() as data:
                data['tasks'] = {f't{i}': {'status': 'pending', 'attempts': []} for i in range(12)}
        allocations = []
        def replica(cfg, config_path, slot, allowed_path, expires, manifest):
            allocations.append(slot)
            with State(cfg.root).edit() as data:
                data['pods'].append({'slot': slot, 'created': time.time(), 'expires': expires,
                                    'ceiling_hourly': 3.35, 'id': f'fake-{slot}', 'status': 'working'})
            candidates = session.members(cfg)
            offset = slot % 2
            for index in [offset, 1-offset]:
                arm = candidates[index]
                state = State(arm.root)
                allowed = [i.split(':', 1)[1] for i in read(allowed_path) if i.startswith(f'{index}:')]
                while lease := state.claim(f'{slot}-0', allowed, 3, 6):
                    iid, aid = lease
                    time.sleep(.001)
                    state.finish(iid, aid, {'valid': True, 'exit_status': 'Submitted',
                                 'prediction': {'model_name_or_path': arm.target, 'model_patch': ''}})
            with State(cfg.root).edit() as data:
                next(p for p in data['pods'] if p['slot'] == slot).update(ended=time.time(), status='terminated')
        with patch.object(fleet, 'replica', side_effect=replica), \
                patch.object(fleet, 'price_ceiling', return_value=3.35), \
                patch.object(fleet.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1000 * 2**30})()), \
                patch.object(fleet.psutil, 'virtual_memory', return_value=type('Memory', (), {'available': 100 * 2**30})()), \
                patch.object(session, 'checkpoints'), patch.object(fleet, 'reconcile_rejections'):
            fleet.phase(self.cfg, self.path, list(session.view(self.cfg)['tasks']), 2, 21600, {'budget_usd': 360})
        self.assertEqual(len(allocations), 2)
        for arm in arms:
            data = read(State(arm.root).path)
            self.assertTrue(all(t['status'] == 'valid' and len(t['attempts']) == 1 for t in data['tasks'].values()))
            self.assertTrue(all(t['attempts'][0]['prediction']['model_name_or_path'] == arm.target for t in data['tasks'].values()))


class CappedSessionTests(unittest.TestCase):
    def setUp(self):
        SessionTests.setUp(self)
        self.cfg.replicas = 12
        self.cfg.max_replicas_per_arm = 6

    def test_worker_main_passes_exact_pins_through_real_eval_validation(self):
        from src.eval import run_eval
        class ReachedResolution(Exception):
            pass
        self.cfg.target_revision = 'first-sha'
        child = OmegaConf.load(self.child_path)
        child.target_revision = 'second-sha'
        OmegaConf.save(child, self.child_path)
        allowed = self.root / 'allowed.json'
        atomic(allowed, ['0:same-id', '1:same-id'])
        for capped in (True, False):
            if not capped:
                del self.cfg.max_replicas_per_arm
            OmegaConf.save(self.cfg, self.path)
            for index in (0, 1):
                with self.subTest(capped=capped, primary_arm=index):
                    targets = session.worker_targets(self.cfg, index)
                    expected = {arm.target: arm.target_revision for arm in session.members(self.cfg) if arm.target in targets}
                    argv = ['worker', '--config', str(self.path), '--server', 'synthetic',
                            '--replica', '0', '--allowed', str(allowed), '--expires', '100000',
                            '--primary-arm', str(index)]
                    with patch.object(sys, 'argv', argv), patch.object(run_eval, '_preflight'), \
                            patch.object(run_eval, 'SshExec'), patch.object(run_eval, 'VllmServer'), \
                            patch.object(run_eval, 'resolve_target', side_effect=ReachedResolution) as resolve:
                        with self.assertRaises(ReachedResolution):
                            worker.main()
                    resolve.assert_called_once_with(targets[0], revision=expected[targets[0]])
                    saved = OmegaConf.load(self.root / 'metadata/replicas/0/eval.yaml')
                    self.assertEqual(dict(saved.target_revisions), expected)

    def test_fixed_lanes_do_not_switch_after_other_arm_drains(self):
        self.assertEqual(session.lane_arms(self.cfg, 12, ['0:x', '1:x']), [0, 1] * 6)
        self.assertEqual(session.lane_arms(self.cfg, 12, ['1:x']), [1] * 6)
        self.assertEqual(session.worker_targets(self.cfg, 0), ['org/first'])
        self.assertEqual(session.worker_targets(self.cfg, 1), ['org/second'])
        del self.cfg.max_replicas_per_arm
        self.assertEqual(session.worker_targets(self.cfg, 1), ['org/second', 'org/first'])

    def test_cap_validation_and_recipe_binding(self):
        self.assertEqual(fleet.recipe_settings(self.cfg)['max_replicas_per_arm'], 6)
        for invalid in (0, -1, 1.5, True):
            self.cfg.max_replicas_per_arm = invalid
            with self.assertRaises(AssertionError):
                session.replica_cap(self.cfg)
        self.cfg.max_replicas_per_arm = 5
        with self.assertRaisesRegex(AssertionError, 'Fleet exceeds'):
            session.replica_cap(self.cfg)

    def test_adopted_lanes_reserve_their_proven_arm(self):
        adopted = [{'primary_arm': 1, 'fixed_arm': True} for _ in range(6)]
        self.assertEqual(session.lane_arms(self.cfg, 12, ['0:x', '1:x'], adopted), [1] * 6 + [0] * 6)
        for records in (adopted + adopted[:1], [{'primary_arm': 0}], [{'primary_arm': 2, 'fixed_arm': True}]):
            with self.assertRaises(AssertionError):
                session.lane_arms(self.cfg, 12, ['0:x', '1:x'], records)

    def test_pending_active_draining_and_ambiguous_pods_hold_lane(self):
        for status in ('allocating', 'booting', 'working', 'draining', 'allocation-unconfirmed', 'rejected-reconciled'):
            self.assertFalse(session.lane_available({'pods': [{'allocation_lane': 2, 'status': status}]}, 2))
        for record in ({'status': 'terminated'}, {'status': 'not-requested'},
                       {'status': 'rejected-reconciled', 'reservation_released': True}):
            self.assertTrue(session.lane_available({'pods': [dict(record, allocation_lane=2)]}, 2))
        self.assertTrue(session.lane_available({'pods': [{'allocation_lane': 1, 'status': 'working'}]}, 2))

    def test_ambiguous_rental_blocks_replacement_until_reconciled(self):
        with State(self.cfg.root).edit() as data:
            data['pods'].append({'slot': 0, 'allocation_lane': 0, 'primary_arm': 0, 'fixed_arm': True,
                'created': time.time(), 'expires': time.time() + 3600, 'ceiling_hourly': 3.35,
                'id': None, 'status': 'allocation-unconfirmed'})
        with patch.object(fleet, 'replica') as rent, \
                patch.object(fleet, 'price_ceiling', return_value=3.35), patch.object(session, 'checkpoints'):
            with self.assertRaisesRegex(AssertionError, 'Reconcile existing capped rentals'):
                fleet.phase(self.cfg, self.path, ['0:same-id'], 1, 21600, {'budget_usd': 360})
        rent.assert_not_called()

    def test_failed_teardown_during_phase_cannot_start_an_extra_replica(self):
        def unresolved(cfg, config_path, slot, allowed_path, expires, manifest):
            with State(cfg.root).edit() as data:
                data['pods'].append({'slot': slot, 'allocation_lane': cfg.allocation_lane,
                    'primary_arm': cfg.primary_arm, 'fixed_arm': True, 'created': time.time(),
                    'expires': expires, 'ceiling_hourly': 3.35, 'id': 'fake', 'status': 'draining'})
            raise RuntimeError('Synthetic teardown failure')
        with patch.object(fleet, 'replica', side_effect=unresolved) as rent, \
                patch.object(fleet, 'price_ceiling', return_value=3.35), \
                patch.object(fleet.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000 * 2**30)), \
                patch.object(fleet.psutil, 'virtual_memory', return_value=SimpleNamespace(available=100 * 2**30)), \
                patch.object(session, 'checkpoints'), patch.object(fleet, 'reconcile_rejections'):
            fleet.phase(self.cfg, self.path, ['0:same-id'], 1, 21600, {'budget_usd': 360})
        self.assertEqual(rent.call_count, 1)
        self.assertIn('reconcile ownership', read(State(self.cfg.root).path)['halt'])

    def test_all_lane_scarcity_rejections_reconcile_and_reach_h100_fallback(self):
        self.cfg.gpu = 'NVIDIA H200'
        self.cfg.fallback_gpus = ['NVIDIA H100 NVL']
        self.cfg.preferred_gpus = []
        start, clock = 100000., [100000.]
        calls = {lane: [] for lane in range(12)}
        arms = session.members(self.cfg)
        def submit(fn, cfg, config_path, slot, allowed, expires, manifest):
            lane = cfg.allocation_lane
            calls[lane].append((cfg.gpu, clock[0] - start))
            with State(cfg.root).edit() as data:
                self.assertTrue(session.lane_available(data, lane))
                occupied = [p for p in data['pods'] if not session.rental_released(p)]
                self.assertLess(sum(p['primary_arm'] == cfg.primary_arm for p in occupied), 6)
                record = {'slot': slot, 'allocation_lane': lane, 'primary_arm': cfg.primary_arm,
                          'fixed_arm': True, 'name': f'fake-{slot}', 'created': clock[0], 'expires': expires,
                          'ceiling_hourly': 3.35, 'id': None, 'status': 'allocation-unconfirmed',
                          'error': 'RunPod GraphQL rejected the request; no capacity'}
                if cfg.gpu == 'NVIDIA H100 NVL':
                    record.update(status='terminated', id=f'fake-{slot}', ended=clock[0])
                data['pods'].append(record)
            future = Future()
            if cfg.gpu == 'NVIDIA H200':
                future.set_exception(RuntimeError(record['error']))
            else:
                with State(arms[cfg.primary_arm].root).edit() as data:
                    data['tasks']['same-id']['status'] = 'valid'
                future.set_result(None)
            return future
        def advance(seconds):
            clock[0] += max(seconds, 30)
            self.assertLess(clock[0] - start, 1800, 'Scarcity did not reach fallback within bounded test time')
        pool = Mock()
        pool.submit.side_effect = submit
        with patch.object(fleet, 'ThreadPoolExecutor') as executor, \
                patch.object(fleet.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(fleet.time, 'sleep', side_effect=advance), \
                patch.object(fleet, 'price_ceiling', return_value=3.35), \
                patch.object(fleet.runpod, 'active_pods', return_value=[]) as inventory, \
                patch.object(fleet.runpod, 'teardown') as teardown, \
                patch.object(fleet.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000 * 2**30)), \
                patch.object(fleet.psutil, 'virtual_memory', return_value=SimpleNamespace(available=100 * 2**30)), \
                patch.object(session, 'checkpoints'):
            executor.return_value.__enter__.return_value = pool
            fleet.phase(self.cfg, self.path, ['0:same-id', '1:same-id'], 12, 21600, {'budget_usd': 360})
        self.assertGreater(inventory.call_count, 0)
        teardown.assert_not_called()
        for attempts in calls.values():
            self.assertGreaterEqual(sum(gpu == 'NVIDIA H200' for gpu, _ in attempts), 5)
            self.assertEqual(attempts[-1][0], 'NVIDIA H100 NVL')
            self.assertGreaterEqual(attempts[-1][1], 600)
        data = read(State(self.cfg.root).path)
        self.assertIsNone(data['halt'])
        rejected = [p for p in data['pods'] if p['id'] is None]
        self.assertTrue(all(p['reservation_released'] and p['reconciliation_inventory_checks'] == 2 for p in rejected))
        for arm in arms:
            self.assertEqual(read(State(arm.root).path)['tasks']['same-id']['status'], 'valid')

    def test_twelve_independent_lanes_replace_failed_worker_without_borrowing(self):
        self.cfg.allocation_retry_seconds = 0
        arms = session.members(self.cfg)
        for arm in arms:
            with State(arm.root).edit() as data:
                data['tasks'] = {f't{i}': {'status': 'pending', 'attempts': []} for i in range(24)}
        gate = threading.Lock()
        initial_ready = threading.Event()
        replacement_ready = threading.Event()
        active, peaks, seen = [0, 0], [0, 0], []
        def replica(cfg, config_path, slot, allowed_path, expires, manifest):
            index = cfg.primary_arm
            self.assertEqual(session.worker_targets(cfg, index), [arms[index].target])
            ids = read(allowed_path)
            self.assertTrue(all(i.startswith(f'{index}:') for i in ids))
            with gate:
                active[index] += 1
                peaks[index] = max(peaks[index], active[index])
                seen.append((slot, cfg.allocation_lane, index))
                if len(seen) == 12:
                    initial_ready.set()
                if slot >= 12:
                    replacement_ready.set()
            with State(cfg.root).edit() as data:
                data['pods'].append({'slot': slot, 'allocation_lane': cfg.allocation_lane,
                    'primary_arm': index, 'fixed_arm': True, 'created': time.time(), 'expires': expires,
                    'ceiling_hourly': 3.35, 'id': f'fake-{slot}', 'status': 'booting'})
            try:
                if not initial_ready.wait(10):
                    raise RuntimeError('Workers did not start independently')
                if slot == 0:
                    raise RuntimeError('Synthetic worker failure before task claim')
                if not replacement_ready.wait(10):
                    raise RuntimeError('Failed lane was not replaced while peers were active')
                state = State(arms[index].root)
                while lease := state.claim(f'{slot}-0', [i.split(':', 1)[1] for i in ids], 3, 6):
                    iid, aid = lease
                    time.sleep(.001)
                    state.finish(iid, aid, {'valid': True, 'exit_status': 'Submitted'})
            finally:
                with State(cfg.root).edit() as data:
                    next(p for p in data['pods'] if p['slot'] == slot).update(status='terminated', ended=time.time())
                with gate:
                    active[index] -= 1
        with patch.object(fleet, 'replica', side_effect=replica), \
                patch.object(fleet, 'price_ceiling', return_value=3.35), \
                patch.object(fleet.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000 * 2**30)), \
                patch.object(fleet.psutil, 'virtual_memory', return_value=SimpleNamespace(available=100 * 2**30)), \
                patch.object(session, 'checkpoints'), patch.object(fleet, 'reconcile_rejections'):
            fleet.phase(self.cfg, self.path, list(session.view(self.cfg)['tasks']), 12, 21600, {'budget_usd': 360})
        self.assertEqual(peaks, [6, 6])
        self.assertEqual(seen[-1][1:], (0, 0))
        self.assertEqual(len(seen), 13)
        for arm in arms:
            self.assertTrue(all(t['status'] == 'valid' and len(t['attempts']) == 1
                                for t in read(State(arm.root).path)['tasks'].values()))


class ServingReuseTests(unittest.TestCase):
    def test_second_pinned_adapter_does_not_restart_base(self):
        from src.infra.endpoints.vllm import VllmServer
        with tempfile.TemporaryDirectory() as root:
            executor = Mock(endpoint_host='127.0.0.1')
            executor.fetch_adapter.side_effect = ['/cache/first', '/cache/second']
            server = VllmServer(Path(root), executor=executor)
            def start(spec, adapter):
                server.running, server.base_model, server.base_revision, server.mode = True, spec.base_model, spec.base_revision, spec.mode
                server._loaded_loras = {spec.model_key}
            def spec(name, revision):
                return SimpleNamespace(adapter=True, hf_path='org/'+name, revision=revision,
                    model_key=name, base_model='base', base_revision='pinned-base', mode='think')
            with patch.object(server, '_start', side_effect=start) as boot, \
                    patch('src.infra.endpoints.vllm.requests.post', return_value=Mock(status_code=200)) as swap:
                server.serve(spec('first', 'sha-a'))
                server.serve(spec('second', 'sha-b'))
            self.assertEqual(boot.call_count, 1)
            self.assertEqual(swap.call_count, 1)
            self.assertEqual(swap.call_args.kwargs['json'], {'lora_name': 'second', 'lora_path': '/cache/second'})
            self.assertEqual(executor.fetch_adapter.call_args_list[1].args, ('org/second', 'sha-b'))
