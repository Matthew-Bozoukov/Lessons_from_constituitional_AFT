# ABOUTME: One serving replica consumes the shared Lite queue through the standard eval lifecycle.
# ABOUTME: The coordinator owns rentals; this worker owns bounded agent subprocesses and durable attempts.
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil

from omegaconf import OmegaConf
from src.eval.capabilities.swebench_mini.agent import AGENT_ENV, rollout_env, write_cost_registry
from scratch.swebench_lite_state import State, atomic, classify, read


def stop_process(proc):
    if proc.poll() is None:
        # Agent children have their own process groups. Killing just the replica
        # group otherwise leaves them retrying a dead endpoint after GPU teardown.
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            proc.wait(timeout=10)
            return
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)


def consume(endpoint, model, cfg, worker, allowed):
    state = State(cfg.root)
    meta = read(state.root / 'metadata/manifest.json')
    rows = {r['instance_id']: r for r in read(state.root / 'metadata/swebench_lite_test.json')}
    images = read(state.root / 'metadata/images.json')
    while lease := state.claim(worker, allowed, cfg.max_infrastructure_attempts,
                               cfg.max_infrastructure_failures, cfg.upload_stale_seconds):
        iid, aid = lease
        out = state.root / 'rollouts' / iid / aid
        out.mkdir(parents=True)
        request = {'instance': rows[iid], 'image': images[iid]['digest'], 'out': str(out),
                   'endpoint': endpoint, 'model': model, 'campaign': meta['campaign'], 'attempt': aid,
                   'cpus': cfg.agent_cpus, 'memory': cfg.agent_memory, 'pids': cfg.agent_pids,
                   'max_response_tokens': cfg.max_response_tokens, 'max_task_tokens': cfg.max_task_tokens}
        # Never persist gold solutions in rollout directories or feed them to the agent.
        request['instance'] = {k: v for k, v in request['instance'].items()
                               if k not in ('patch', 'test_patch', 'hints_text', 'FAIL_TO_PASS', 'PASS_TO_PASS')}
        atomic(out / 'request.json', request)
        registry = write_cost_registry(out, model)
        env = {k: v for k, v in os.environ.items() if not any(s in k for s in ('TOKEN', 'API_KEY', 'PASSWORD'))}
        env.update(rollout_env(registry=registry, global_config_dir=out / 'global_config'))
        env['MSWEA_SILENT_STARTUP'] = '1'
        proc = None
        rc = -1
        error = None
        try:
            with (out / 'agent.log').open('w') as log:
                proc = subprocess.Popen([str(AGENT_ENV / '.venv/bin/python'), '-m', 'scratch.swebench_lite_task',
                                         '--request', str(out / 'request.json')], env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic(out / 'process.json', {'pid': proc.pid, 'started': time.time()})
                until = min(time.time() + cfg.task_seconds, read(state.path)['deadline'] - cfg.cleanup_reserve_seconds)
                while proc.poll() is None:
                    live = read(state.path)
                    if (time.time() > until or live.get('halt') or
                            time.time() - live.get('last_upload', 0) > cfg.upload_stale_seconds):
                        raise TimeoutError('deadline, backup lag, or campaign circuit breaker')
                    time.sleep(2)
                rc = proc.returncode
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        finally:
            if proc:
                stop_process(proc)
            ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=lasr_attempt=' + aid], text=True).split()
            if ids:
                subprocess.run(['docker', 'rm', '-f', *ids], check=True, timeout=90)
        path = out / iid / (iid + '.traj.json')
        traj = read(path) if path.exists() else {}
        result = classify(traj, rc)
        if error:
            result.update(valid=False, error=error)
        if result['valid']:
            preds = read(out / 'preds.json')
            if iid not in preds or preds[iid].get('model_patch') is None:
                result.update(valid=False, error='No durable prediction')
            else:
                result['prediction'] = preds[iid]
        transcript = '\n\n'.join('### ' + str(m.get('role', 'message')) + '\n\n```json\n' +
                                   json.dumps(m, indent=2, ensure_ascii=False) + '\n```'
                                   for m in traj.get('messages', []))
        (out / 'transcript.md').write_text(transcript, encoding='utf-8')
        state.finish(iid, aid, result)
        print(f'{iid}: {result["exit_status"]}; valid={result["valid"]}', flush=True)


def runner(target, cfg, out_dir, **kwargs):
    campaign = OmegaConf.load(cfg.campaign_config)
    assert target.spec.revision == campaign.target_revision
    assert target.spec.base_revision == campaign.base_revision and target.spec.mode == campaign.mode
    endpoint = target.base_url
    with ThreadPoolExecutor(max_workers=campaign.workers_per_replica) as pool:
        futures = [pool.submit(consume, endpoint, 'hosted_vllm/' + target.model_name, campaign,
                               str(cfg.replica) + '-' + str(i), list(cfg.allowed))
                   for i in range(campaign.workers_per_replica)]
        for future in futures:
            future.result()
    (out_dir / 'metadata').mkdir(exist_ok=True)
    return {'status': 'replica finished; canonical campaign owns scoring and publication'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--server', required=True)
    parser.add_argument('--replica', type=int, required=True)
    parser.add_argument('--allowed', required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    output = Path(cfg.root) / 'metadata' / 'replicas' / str(args.replica)
    output.mkdir(parents=True, exist_ok=True)
    worker_config = {'target_revision': cfg.target_revision, 'serving': OmegaConf.to_container(cfg.serving),
                     'dataset': cfg.dataset, 'revision': cfg.dataset_revision,
                     'frozen_dataset': str(Path(cfg.root) / 'metadata/swebench_lite_test.json'),
                     'campaign_config': str(Path(args.config).resolve()), 'replica': args.replica,
                     'allowed': read(args.allowed), 'output_root': str(output / 'eval')}
    OmegaConf.save(OmegaConf.create(worker_config), output / 'eval.yaml')
    from src.eval.run_eval import main as evaluate
    evaluate(['--target', cfg.target, '--name', 'swebench_mini', '--config', str(output / 'eval.yaml'),
              '--server', args.server, '--server-bind', '127.0.0.1', '--ssh-key', cfg.ssh_key,
              '--port', str(cfg.port_base + args.replica), '--no-push'], runner=runner)


if __name__ == '__main__':
    main()
