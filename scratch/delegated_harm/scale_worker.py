# ABOUTME: Claim one episode at a time from the durable queue using a pinned adapter endpoint.
# ABOUTME: Workers write isolated artifacts; only the coordinator merges and publishes results.
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from omegaconf import OmegaConf

from scratch.delegated_harm.episode_queue import EpisodeQueue
from .recovery import read, TokenBudget
from src.eval.misalignment.delegated_harm.source import save, digest, prepare


def run_worker(target, cfg, out_dir):
    from src.eval.misalignment.delegated_harm.runner import episode
    worker = str(cfg.scaling.worker)
    arm = str(cfg.scaling.arm)
    queue = EpisodeQueue(cfg.scaling.queue)
    if cfg.scaling.get('kind') == 'fresh':
        from scratch.delegated_harm.fresh import await_inputs
        await_inputs(target, cfg)
    inputs = read(cfg.scaling.inputs)
    assert target.spec.revision == inputs['target']['revision']
    assert target.spec.base_revision == inputs['target']['base_revision']
    scenarios, humans, _, _ = prepare(cfg)
    for name in ['metadata', 'results/episodes', 'rollouts']:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    save(out_dir / 'metadata/scaling_inputs.json', inputs)
    client = OpenAI(base_url=target.base_url, api_key=target.api_key,
                    timeout=float(cfg.generation.request_timeout), max_retries=2)
    budget = TokenBudget(target.base_url, int(cfg.serving.context_window))
    queue.register(worker, arm, {'pid': os.getpid(), 'out_dir': str(out_dir.resolve()),
                                'target': inputs['target'], 'hardware': 'NVIDIA H200'})
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(15):
            queue.heartbeat(worker)
    watcher = threading.Thread(target=heartbeat, daemon=True)
    watcher.start()
    def consume(_):
        while True:
            state, cell = queue.claim(worker)
            if state in ('empty', 'drain'):
                return
            if state == 'paused':
                time.sleep(2)
                continue
            old = inputs.get('original_records', {}).get(cell['id'])
            local = OmegaConf.create(OmegaConf.to_container(cfg))
            if old and old['status'] == 'token_limit':
                local.generation.max_tokens = int(cfg.recovery.output_limit_tokens)
            if old and old['status'] == 'turn_limit':
                local.generation.max_turns = int(cfg.recovery.turn_limit)
            author = inputs['authors'][f"{cell['scenario']}__v{cell['variant']}"]
            if cell['delivery'] != 'chat_human' and not author['accepted']:
                rec = dict(cell, status='author_failure')
                result = out_dir / 'results/episodes' / f"{cell['id']}.json"
                save(result, rec)
                queue.finish(worker, arm, cell['id'], result, digest(rec))
                continue
            brief = humans[cell['scenario']] if cell['delivery'] == 'chat_human' else author['text']
            trace_path = out_dir / 'rollouts' / f"{cell['id']}.json"
            print(f"CLAIM {worker} {cell['id']}", flush=True)
            trace = episode(client, target.model_name, str(cfg.subject_model_label), scenarios[cell['scenario']],
                            cell, brief, local, trace_path, request_budget=budget)
            provenance = {'context_window': int(cfg.serving.context_window), 'max_tokens': int(local.generation.max_tokens),
                'max_turns': int(local.generation.max_turns), 'attempt': 1,
                'worker': worker, 'hardware': 'NVIDIA H200'}
            field = 'recovery' if old else 'execution'
            if old:
                provenance.update(original_status=old['status'], original_record_sha256=digest(old))
            trace[field] = provenance
            save(trace_path, trace)
            rec = dict(cell, status=trace['status'], mechanical=trace['mechanical'], **{field: provenance})
            result = out_dir / 'results/episodes' / f"{cell['id']}.json"
            save(result, rec)
            queue.finish(worker, arm, cell['id'], result, digest(rec))
            print(f"DONE {worker} {cell['id']} {trace['status']}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=int(cfg.generation.parallel)) as pool:
            list(pool.map(consume, range(int(cfg.generation.parallel))))
    finally:
        stop.set()
        watcher.join(timeout=2)
        queue.mode(worker, 'stopped')
    save(out_dir / 'metadata/worker_finished.json', {'worker': worker, 'finished': time.time()})
    return {'worker': worker, 'status': 'finished', 'publishes_results': False}
