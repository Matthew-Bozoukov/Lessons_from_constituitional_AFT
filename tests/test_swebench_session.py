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
        self.assertEqual(worker.latest_admission(self.cfg, now+21600), now+19620)
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
                patch.object(session, 'checkpoints'), patch.object(fleet, 'reconcile_rejections'):
            fleet.phase(self.cfg, self.path, list(session.view(self.cfg)['tasks']), 2, 21600, {'budget_usd': 360})
        self.assertEqual(len(allocations), 2)
        for arm in arms:
            data = read(State(arm.root).path)
            self.assertTrue(all(t['status'] == 'valid' and len(t['attempts']) == 1 for t in data['tasks'].values()))
            self.assertTrue(all(t['attempts'][0]['prediction']['model_name_or_path'] == arm.target for t in data['tasks'].values()))


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
