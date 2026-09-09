# ABOUTME: Bounded owner for one overnight baseline pod; dispatches the existing eval runner.
# ABOUTME: Arms independent teardown before bootstrap and records actual rate and estimated accrual.
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

from omegaconf import OmegaConf
from src.infra import runpod
from src.infra.endpoints.vllm import resolve_target
from src.eval.run_eval import main as evaluate
from src.eval.docker import docker_preflight, require_network_capacity, require_lf_shell_scripts
from src.infra.huggingface import hf_org

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/nonmoral_overnight/20260909'
TARGET = 'dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch'
REVISION = '2225547cec8bd312a1e025f02fb6b3321c047e4a'
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
LIFETIME = 8 * 3600
CHECKPOINTS = {
    'nonmoral': (TARGET, REVISION, 'baseline'),
    'nonmoral_lf': (TARGET, REVISION, 'baseline_lf'),
    'math': ('matboz/qwen3.6-27b-lora-9284-numina-control-716-r64',
             'edfb4287c10f553c541ba28216f202d0c0f47055', 'math_baseline'),
    'table2': ('dougalldeepmind/2026-08-04-qwen36-lora-table2-only-9284-rank-64',
               '2c513ea7513baf792bd2becf0900b5c9d858c92d', 'table2_baseline'),
}


def prior_lane_spend(out: Path, prefix: str) -> float:
    """Refuse overlap/replacement; charge closed rentals by full observed lifetime."""
    total = 0.0
    for _, _, known_prefix in CHECKPOINTS.values():
        path = out / f'{known_prefix}_status.json'
        if not path.exists():
            continue
        prior = json.loads(path.read_text())
        if not prior.get('pod_id'):
            continue
        if known_prefix == prefix or not prior.get('termination_verified'):
            raise RuntimeError(f'Existing owned rental must be preserved/closed: {path}')
        elapsed = prior['terminated_at_unix'] - prior['rented_at_unix']
        total += elapsed / 3600 * (prior.get('actual_gpu_hourly_usd', 4.5) + 0.1)
    return total


def main(checkpoint='nonmoral'):
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    target, revision, prefix = CHECKPOINTS[checkpoint]
    prior_spend = prior_lane_spend(OUT, prefix)
    lifetime = min(LIFETIME, int((55 - prior_spend) / 4.6 * 3600))
    assert lifetime >= 3600, 'Less than one hour remains in the GPU lane; no rental'
    state = dict(phase='preflight', target=target, target_revision=revision,
                 base_revision=BASE_REVISION, pid=os.getpid(), pod_id=None,
                 gpu_cap_usd=60, judge_cap_usd=10, max_lifetime_s=lifetime,
                 prior_lane_gpu_storage_estimate_usd=prior_spend,
                 max_gpu_hourly_usd=4.5, storage_hourly_reserve_usd=0.1,
                 cost_note='Elapsed-rate estimates, not provider invoice; $5 teardown reserve.',
                 passes=3, expected_rollouts=240)
    shared_ledger = OUT/'baseline_judge_ledger.json'
    state['judge_ledger_start_index'] = len(json.loads(shared_ledger.read_text())) if shared_ledger.exists() else 0
    lock = threading.Lock()
    def save(**updates):
        with lock:
            state.update(updates)
            state['updated_at_unix'] = time.time()
            if state.get('rented_at_unix'):
                elapsed = (state.get('terminated_at_unix') or time.time())-state['rented_at_unix']
                state['elapsed_seconds'] = elapsed
                state['estimated_gpu_and_storage_usd'] = elapsed/3600*(state.get('actual_gpu_hourly_usd',4.5)+0.1)
            ledger = OUT/'baseline_judge_ledger.json'
            if ledger.exists():
                try:
                    entries = json.loads(ledger.read_text())
                    state['judge_charged_or_reserved_usd'] = sum(e['charged_or_reserved_usd'] for e in entries)
                except (ValueError, OSError):
                    state['judge_ledger_read_pending'] = True
            tmp=OUT/f'{prefix}_status.tmp'
            tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
            tmp.replace(OUT/f'{prefix}_status.json')
    save()
    docker_preflight()
    require_network_capacity(16,because='ODCV concurrency 8')
    keypair=runpod.default_keypair()
    assert keypair, 'SSH keypair missing; no rental'
    assert hf_org() == 'dougalldeepmind'
    spec=resolve_target(target)
    assert (spec.revision,spec.base_revision)==(revision,BASE_REVISION)
    cfg=OmegaConf.load(ROOT/'scratch/nonmoral/odcv-paired.yaml')
    cfg.passes=3
    cfg.output_root='C:/nm-eval'
    cfg.run_name=f'odcv-{checkpoint}-common-3x'
    cfg.bench_dir=str((ROOT/str(cfg.bench_dir)).resolve())
    require_lf_shell_scripts(Path(cfg.bench_dir))
    cfg.judge_budget=dict(ledger=str(OUT/'baseline_judge_ledger.json'),cap_usd=10,max_tokens=8192)
    config=OUT/f'{prefix}_config.yaml'
    OmegaConf.save(cfg,config)
    sources=['src/eval/run_eval.py','src/eval/misalignment/odcv/runner.py',
             'src/eval/docker.py','src/eval/misalignment/odcv/odcv_rollout.py',
             'src/eval/misalignment/odcv/recover.py',
             'src/eval/misalignment/odcv/odcv_judge.py','src/eval/misalignment/odcv/progress_judge.py',
             'src/infra/runpod.py','scratch/nonmoral/overnight_baseline.py']
    save(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
         git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
         shell_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT/str(cfg.bench_dir)).rglob('*.sh'))})
    watchdog=None
    finished=threading.Event()
    def monitor():
        while not finished.wait(30):
            save()
    def arm(pod_id):
        nonlocal watchdog
        save(pod_id=pod_id,rented_at_unix=time.time(),phase='provisioned')
        watchdog=runpod.start_watchdog(pod_id,lifetime,OUT/f'watchdog_{pod_id}.log')
        save(watchdog_pid=watchdog.pid)
        info=runpod.call('GET',f'/pods/{pod_id}')
        rate=float(info['costPerHr'])
        save(actual_gpu_hourly_usd=rate,provider_created_at=info.get('createdAt'),
             worst_case_gpu_storage_and_teardown_usd=prior_spend+lifetime/3600*(rate+0.1)+5)
        assert 0 < rate <= 4.5, f'Unexpected GPU rate {rate}; teardown'
        assert prior_spend+lifetime/3600*(rate+0.1)+5 <= 60
        threading.Thread(target=monitor,daemon=True).start()
        print(f'OWNED POD {pod_id}: ${rate}/h; watchdog {watchdog.pid}; cap {lifetime}s',flush=True)
    try:
        pod=runpod.provision_eval_pod([target],name=f'nikak-{checkpoint}-baseline-20260909',
                                    pubkey_path=keypair[0],identity=keypair[1],on_provisioned=arm)
        save(phase='bootstrapping',host=pod.host)
        if not runpod.wait_bootstrapped(pod.id,timeout_s=3600):
            raise TimeoutError('Baseline pod bootstrap exceeded one hour')
        save(phase='evaluating')
        evaluate(['--name','odcv','--config',str(config),'--target',target,
                  '--server',pod.host,'--ssh-key',keypair[1]])
        save(phase='evaluation_completed')
    except BaseException as exc:
        save(phase='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        raise
    finally:
        if state['pod_id']:
            gone=runpod.terminate(state['pod_id'])
            still=any(p.get('id')==state['pod_id'] for p in runpod.active_pods())
            if gone and not still:
                save(terminated_at_unix=time.time(),termination_verified=True)
                if watchdog:
                    watchdog.terminate()
            else:
                save(termination_verified=False)
                print(f"URGENT owned pod {state['pod_id']} may still bill; watchdog retained",flush=True)
        finished.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', choices=CHECKPOINTS, default='nonmoral')
    main(parser.parse_args().checkpoint)
