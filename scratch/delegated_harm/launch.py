# ABOUTME: Own one adapter's GPU, driver, spend deadline and verified cleanup.
# ABOUTME: Run twice in parallel: uv run scratch/delegated_harm/launch.py control|da.
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from omegaconf import OmegaConf

from src.infra import runpod
from src.infra.endpoints.vllm import resolve_target
from src.eval.misalignment.delegated_harm.source import prepare, save
from src.naming import eval_name

MODELS = {
    "control": "matboz/qwen3.6-27b-lora-9284-numina-control-716-r64",
    "da": "dougalldeepmind/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch",
}


def account():
    response = requests.post("https://api.runpod.io/graphql",
        headers={"Authorization": "Bearer " + runpod._key()},
        json={"query": "query { myself { clientBalance currentSpendPerHr } gpuTypes { id securePrice } }"},
        timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError("Cannot read provider balance/prices")
    return payload["data"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=MODELS)
    args = parser.parse_args()
    cfg = OmegaConf.load("configs/eval/delegated_harm.yaml")
    prepare(cfg)
    model = MODELS[args.arm]
    spec = resolve_target(model)
    assert spec.revision == cfg.expected_revisions[model]
    assert spec.base_revision == cfg.expected_base_revision
    name = eval_name("delegated_harm", spec.model_key)
    pair = runpod.default_keypair()
    if not pair:
        raise RuntimeError("No local SSH keypair")
    snapshot = account()
    gpu = "NVIDIA H100 80GB HBM3"
    quote = next(float(g["securePrice"]) for g in snapshot["gpuTypes"] if g["id"] == gpu)
    hours = min(3.75, 14.5 / (quote + 0.10))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    root = Path("output/delegated_harm") / f"{stamp}_{args.arm}"
    root.mkdir(parents=True, exist_ok=True)
    state = {"arm": args.arm, "model": model, "revision": spec.revision,
             "base_revision": spec.base_revision, "eval_name": name,
             "account_before": snapshot["myself"], "quoted_gpu_hourly_usd": quote,
             "max_hours": hours, "api_budget_usd": float(cfg.judge.budget_usd),
             "budget_usd": 30, "pid": os.getpid(), "status": "provisioning"}
    statefile = root / "controller.json"
    save(statefile, state)
    guard = None
    def owned(pod_id):
        nonlocal guard
        state.update(pod_id=pod_id, billing_started=time.time())
        save(statefile, state)
        guard = runpod.start_watchdog(pod_id, int(hours*3600), root / "watchdog.log")
        pod = runpod.call("GET", f"/pods/{pod_id}")
        rate = float(pod.get("costPerHr") or 0)
        if rate <= 0 or rate * hours > 15:
            raise RuntimeError("Actual pod hourly charge cannot fit GPU budget")
        state["actual_pod_hourly_usd"] = rate
        save(statefile, state)
    try:
        runpod.up(name=f"subagents-{args.arm}-{stamp}", eval=model, gpu=gpu,
                  max_hours=hours, on_provisioned=owned)
        pod_id = state["pod_id"]
        if not runpod.wait_bootstrapped(pod_id, timeout_s=2400):
            raise RuntimeError("GPU bootstrap did not finish")
        pod = runpod.call("GET", f"/pods/{pod_id}")
        host = f"root@{pod['publicIp']}:{pod['portMappings']['22']}"
        port = 9101 if args.arm == "control" else 9102
        state.update(server=host, local_port=port, status="evaluating")
        command = [sys.executable, "scripts/run_eval.py", "--target", model, "--name", "delegated_harm",
                   "--server", host, "--ssh-key", pair[1], "--port", str(port), "--terminate-pod",
                   f"output_root={root.as_posix()}/runs"]
        state["command"] = command
        save(statefile, state)
        completed = subprocess.run(command, check=False)
        state.update(driver_exit_code=completed.returncode,
                     status="finished" if completed.returncode == 0 else "failed")
        save(statefile, state)
        if completed.returncode:
            raise RuntimeError(f"Eval driver exited {completed.returncode}")
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:500])
        save(statefile, state)
        raise
    finally:
        if state.get("pod_id"):
            runpod.teardown(state["pod_id"])
            state.update(terminated=True, finished=time.time(), account_after=account()["myself"])
            save(statefile, state)
            if guard is not None:
                guard.terminate()


if __name__ == "__main__":
    main()
