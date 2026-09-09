# ABOUTME: Bounded owner for one overnight baseline pod; dispatches the existing eval runner.
# ABOUTME: Arms independent teardown before bootstrap and records actual rate and estimated accrual.
from __future__ import annotations

import hashlib
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
from src.eval.docker import docker_preflight, require_network_capacity
from src.infra.huggingface import hf_org

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/nonmoral_overnight/20260909'
TARGET = 'dougalldeepmind/2026-09-02-qwen36-lora-table2-9284-nonmoral-deliberation-684-rank-64-dynbatch'
REVISION = '2225547cec8bd312a1e025f02fb6b3321c047e4a'
BASE_REVISION = '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
LIFETIME = 8 * 3600


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    state = dict(phase='preflight', target=TARGET, target_revision=REVISION,
                 base_revision=BASE_REVISION, pid=os.getpid(), pod_id=None,
                 gpu_cap_usd=60, judge_cap_usd=10, max_lifetime_s=LIFETIME,
                 max_gpu_hourly_usd=4.5, storage_hourly_reserve_usd=0.1,
                 cost_note='Elapsed-rate estimates, not provider invoice; $5 teardown reserve.',
                 passes=3, expected_rollouts=240)
    lock = threading.Lock()
    def save(**updates):
        with lock:
            state.update(updates)
            state['updated_at_unix'] = time.time()
            if state.get('rented_at_unix') and not state.get('terminated_at_unix'):
                elapsed = time.time()-state['rented_at_unix']
                state['elapsed_seconds'] = elapsed
                state['estimated_gpu_and_storage_usd'] = elapsed/3600*(state.get('actual_gpu_hourly_usd',4.5)+0.1)
            ledger = OUT/'baseline_judge_ledger.json'
            if ledger.exists():
                try:
                    entries = json.loads(ledger.read_text())
                    state['judge_charged_or_reserved_usd'] = sum(e['charged_or_reserved_usd'] for e in entries)
                except (ValueError, OSError):
                    state['judge_ledger_read_pending'] = True
            tmp=OUT/'baseline_status.tmp'
            tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
            tmp.replace(OUT/'baseline_status.json')
    save()
    docker_preflight()
    require_network_capacity(16,because='ODCV concurrency 8')
    keypair=runpod.default_keypair()
    assert keypair, 'SSH keypair missing; no rental'
    assert hf_org() == 'dougalldeepmind'
    spec=resolve_target(TARGET)
    assert (spec.revision,spec.base_revision)==(REVISION,BASE_REVISION)
    cfg=OmegaConf.load(ROOT/'configs/eval/odcv-nonmoral-paired.yaml')
    cfg.passes=3
    cfg.output_root='C:/nm-eval'
    cfg.run_name='odcv-nonmoral-common-3x'
    cfg.bench_dir=str((ROOT/str(cfg.bench_dir)).resolve())
    cfg.judge_budget=dict(ledger=str(OUT/'baseline_judge_ledger.json'),cap_usd=10,max_tokens=8192)
    config=OUT/'baseline_config.yaml'
    OmegaConf.save(cfg,config)
    sources=['src/eval/run_eval.py','src/eval/misalignment/odcv/runner.py',
             'src/eval/misalignment/odcv/odcv_judge.py','src/eval/misalignment/odcv/progress_judge.py',
             'src/infra/runpod.py','scratch/nonmoral/overnight_baseline.py']
    save(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
         git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    watchdog=None
    finished=threading.Event()
    def monitor():
        while not finished.wait(30):
            save()
    def arm(pod_id):
        nonlocal watchdog
        save(pod_id=pod_id,rented_at_unix=time.time(),phase='provisioned')
        watchdog=runpod.start_watchdog(pod_id,LIFETIME,OUT/f'watchdog_{pod_id}.log')
        save(watchdog_pid=watchdog.pid)
        info=runpod.call('GET',f'/pods/{pod_id}')
        rate=float(info['costPerHr'])
        save(actual_gpu_hourly_usd=rate,provider_created_at=info.get('createdAt'),
             worst_case_gpu_storage_and_teardown_usd=LIFETIME/3600*(rate+0.1)+5)
        assert 0 < rate <= 4.5, f'Unexpected GPU rate {rate}; teardown'
        assert LIFETIME/3600*(rate+0.1)+5 <= 60
        threading.Thread(target=monitor,daemon=True).start()
        print(f'OWNED POD {pod_id}: ${rate}/h; watchdog {watchdog.pid}; cap {LIFETIME}s',flush=True)
    try:
        pod=runpod.provision_eval_pod([TARGET],name='nikak-nonmoral-baseline-20260909',
                                    pubkey_path=keypair[0],identity=keypair[1],on_provisioned=arm)
        save(phase='bootstrapping',host=pod.host)
        if not runpod.wait_bootstrapped(pod.id,timeout_s=3600):
            raise TimeoutError('Baseline pod bootstrap exceeded one hour')
        save(phase='evaluating')
        evaluate(['--name','odcv','--config',str(config),'--target',TARGET,
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
    main()
