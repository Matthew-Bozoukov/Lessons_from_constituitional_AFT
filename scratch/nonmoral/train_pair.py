# ABOUTME: Runs one or two authorized SFT conditions on one protected 2xH200 pod.
# ABOUTME: Uses the shared trainer/provisioner, durable local monitoring and verified owned-pod teardown.
import argparse
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra import runpod
from src.infra.endpoints.vllm import SshExec
from src.infra.huggingface import hf_api, hf_org
from scratch.nonmoral.result_backup import fetch_training_outputs, may_terminate_training

MAX_LIFETIME_S = int(58 / 10 * 3600)
RECOVERY_RESERVE_S = 900


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def commands(plan):
    assert 1 <= len(plan["arms"]) <= 2, "One or two conditions, one LoRA per condition"
    assert len({a["data_repo"] for a in plan["arms"]}) == len(plan["arms"])
    result = []
    for arm in plan["arms"]:
        assert re.fullmatch(r"[a-f0-9]{40}", arm["data_revision"])
        assert re.fullmatch(r"[a-f0-9]{40}", plan["base_model_revision"])
        argv = ["uv", "run", "torchrun", "--nproc_per_node=2",
                "scripts/train/train_lora.py", "--config", "configs/train/sft.yaml",
                "model=qwen36", "seed=0", "wandb=false", "constitution=none",
                "data_repo=" + arm["data_repo"], "data_revision=" + arm["data_revision"],
                "base_model_revision=" + plan["base_model_revision"]]
        result.append(shlex.join(argv))
    return result


def run(plan_path, out):
    load_dotenv(ROOT / ".env")
    assert hf_org() == "dougalldeepmind"
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    assert plan.get("approved_for_training") is True
    cmd = commands(plan)
    gpu_budget = float(plan.get('gpu_budget_usd', 60))
    assert 20 <= gpu_budget <= 60, 'Bounded SFT allocation must be $20..$60'
    max_lifetime_s = min(MAX_LIFETIME_S, int((gpu_budget - 2) / 10 * 3600))
    run_name = plan.get('run_name', 'nika-nonmoral-paired-train')
    assert re.fullmatch(r'[a-z0-9-]+', run_name)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "status.json").exists(), "Do not duplicate a paid launch"
    state = {"phase": "preflight", "time_utc": datetime.now(timezone.utc).isoformat(),
             "plan": plan, "commands": cmd, "owned_pod": None,
             "gpu_budget_usd": gpu_budget, "completed_arms": []}
    dump(out / "status.json", state)
    for arm in plan["arms"]:
        info = hf_api().dataset_info(arm["data_repo"], revision=arm["data_revision"])
        assert info.sha == arm["data_revision"] and not info.private

    dog = None
    created = None
    remote = None

    def registered(pod_id):
        nonlocal dog, created
        state["owned_pod"] = pod_id
        created = time.time()
        state["created_epoch"] = created
        dump(out / "status.json", state)
        # Price ceiling below is $10/h inclusive of a conservative disk allowance.
        # Reserve $2 for API latency and teardown. Never leave bootstrap unprotected.
        dog = runpod.start_watchdog(pod_id, max_lifetime_s, out / "watchdog.log")
        info = runpod.call("GET", "/pods/" + pod_id)
        state["gpu_hourly_usd"] = float(info["costPerHr"])
        state["budget_hourly_usd"] = state["gpu_hourly_usd"] + 0.10
        assert state["budget_hourly_usd"] <= 10, "Quoted pair exceeds reserved hourly ceiling"
        dump(out / "status.json", state)

    try:
        rendered = runpod.up(run_name, train="configs/train/sft.yaml",
                             model="qwen36", count=2, push_env=True,
                             on_provisioned=registered)
        (out / "provision.txt").write_text(rendered, encoding="utf-8")
        host = re.search(r"^host:\s+(\S+)", rendered, re.M).group(1)
        state.update(phase="bootstrap", host=host)
        dump(out / "status.json", state)
        assert runpod.wait_bootstrapped(state["owned_pod"], timeout_s=1800), "Bootstrap timed out"
        remote = SshExec(host, port=8000, workdir="/root/work")
        cuda = remote._ssh("cd /root/work && uv run python -c " + shlex.quote(
            "import json, torch; assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count()==2; "
            "print(json.dumps([torch.cuda.get_device_name(i) for i in range(2)]))"), timeout=180)
        assert "H200" in cuda
        state["cuda_check"] = cuda.strip()
        remote_dir = "/root/work/output/nonmoral-paired-supervision"
        script = ["#!/bin/bash", "set -u", "cd /root/work",
                  "export PYTHONUNBUFFERED=1", "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                  f"mkdir -p {shlex.quote(remote_dir)}"]
        for i, command in enumerate(cmd):
            script += [f"echo ARM_START_{i}", command + f" > {remote_dir}/arm_{i}.log 2>&1",
                       "rc=$?", f"printf '%s' \"$rc\" > {remote_dir}/arm_{i}.exit",
                       'if [ "$rc" -ne 0 ]; then exit "$rc"; fi']
        script += [f"touch {remote_dir}/complete"]
        script_text = "\n".join(script) + "\n"
        (out / "remote_run.sh").write_text(script_text, encoding="utf-8")
        remote._ssh(f"mkdir -p {remote_dir} && cat > {remote_dir}/run.sh", stdin_text=script_text)
        # Persist intent before SSH: a lost launch response must not cause teardown
        # of a trainer which actually started. Its process group is owned by this run.
        state['training_started'] = True
        dump(out / 'status.json', state)
        remote._ssh(f"nohup setsid bash {remote_dir}/run.sh > {remote_dir}/driver.log 2>&1 </dev/null & "
                    f"echo $! > {remote_dir}/driver.pid")
        state["phase"] = "training"
        last_sizes, last_change = {}, time.time()
        while True:
            if time.time()-created >= max_lifetime_s-RECOVERY_RESERVE_S:
                raise RuntimeError('Training window reached; preserve outputs within recovery reserve')
            probe = """
import json
from pathlib import Path
p=Path('/root/work/output/nonmoral-paired-supervision')
r={'complete':(p/'complete').exists(),'arms':[]}
for i in range(ARM_COUNT):
 f=p/f'arm_{i}.log'; e=p/f'arm_{i}.exit'
 text=f.read_text(errors='replace') if f.exists() else ''
 r['arms'].append({'index':i,'bytes':f.stat().st_size if f.exists() else 0,
                  'tail':text[-4000:], 'exit':int(e.read_text()) if e.exists() else None})
r['metadata']={str(f):json.loads(f.read_text()) for f in Path('/root/work/output/train').glob('*/run_meta.json')}
print(json.dumps(r))
""".replace('ARM_COUNT',str(len(cmd)))
            try:
                progress = json.loads(remote._ssh("python -c " + shlex.quote(probe), timeout=90))
            except Exception as exc:
                state["monitor_error"] = type(exc).__name__
                dump(out / "status.json", state)
                if time.time() - last_change > 1800:
                    raise RuntimeError("Training monitor unavailable for 30 minutes") from exc
                time.sleep(20)
                continue
            sizes = {a["index"]: a["bytes"] for a in progress["arms"]}
            if sizes != last_sizes:
                last_sizes, last_change = sizes, time.time()
            state.update(progress=progress, elapsed_s=time.time()-created,
                         estimated_gpu_usd=(time.time()-created)/3600*state["budget_hourly_usd"])
            dump(out / "status.json", state)
            for arm in progress["arms"]:
                (out / f"arm_{arm['index']}_tail.log").write_text(arm["tail"], encoding="utf-8")
                if arm["exit"] is not None and arm["exit"] != 0:
                    raise RuntimeError(f"Training arm {arm['index']} failed with exit {arm['exit']}")
                if arm["exit"] == 0 and arm["index"] not in state["completed_arms"]:
                    state["completed_arms"].append(arm["index"])
                    print(f"ARM_COMPLETE {arm['index']}", flush=True)
            if progress["complete"]:
                state["phase"] = "trained"
                break
            if time.time()-last_change > 1800:
                raise RuntimeError("Training logs stalled for 30 minutes")
            if time.time()-created >= max_lifetime_s-RECOVERY_RESERVE_S:
                raise RuntimeError("Training window reached; preserve outputs within recovery reserve")
            print(json.dumps({"phase":state["phase"], "seconds":round(state["elapsed_s"]),
                              "gpu_usd":round(state["estimated_gpu_usd"],3), "log_bytes":sizes}),flush=True)
            time.sleep(30)
        for file, meta in state["progress"]["metadata"].items():
            dump(out / (Path(file).parent.name + "_run_meta.json"), meta)
        dump(out / "status.json", state)
    except BaseException as exc:
        state["phase"] = "failed"
        state["failure"] = str(exc)
        dump(out / "status.json", state)
        raise
    finally:
        if state["owned_pod"]:
            if state.get('training_started'):
                # On failure, freeze only our process group before snapshotting saved
                # checkpoints/logs. A cleanly exited group simply no longer exists.
                recovery_deadline = min(time.time()+RECOVERY_RESERVE_S, created+max_lifetime_s-30)
                while time.time() < recovery_deadline-15:
                    try:
                        remaining = int(recovery_deadline-time.time())
                        if remaining < 15:
                            raise TimeoutError('No recovery time left before independent watchdog')
                        remote._ssh("p=/root/work/output/nonmoral-paired-supervision/driver.pid; "
                                    'test -f "$p" || exit 1; g=$(cat "$p"); '
                                    'case "$g" in ""|*[!0-9]*) exit 1;; esac; '
                                    'if kill -0 -- -"$g" 2>/dev/null; then '
                                    'kill -STOP -- -"$g"; fi', timeout=min(30,remaining))
                        expected = [dict(plan['arms'][i], base_model_revision=plan['base_model_revision'])
                                    for i in state['completed_arms']]
                        state['local_backup'] = fetch_training_outputs(
                            remote, out, expected, timeout=max(1,min(300,(remaining-35)//2)))
                        dump(out / 'local_backup.json', state['local_backup'])
                        break
                    except Exception as exc:
                        state['backup_error'] = f'{type(exc).__name__}: {exc}'
                        dump(out / 'status.json', state)
                        time.sleep(max(0,min(15,recovery_deadline-time.time())))
                if not may_terminate_training(state):
                    state['phase'] = 'artifact_recovery_required'
                    print('URGENT: local result backup unverified; ordinary teardown blocked. '
                          'Owned pod retained under its existing budget watchdog.', flush=True)
                    # The watchdog also reacts to owner death. Keep this owner alive
                    # until its existing hard ceiling, rather than exiting and causing
                    # an immediate watchdog kill after a recoverable transfer failure.
                    dump(out / 'status.json', state)
                    while time.time() < created+max_lifetime_s:
                        time.sleep(min(15,created+max_lifetime_s-time.time()))
            state["terminated"] = (runpod.terminate(state["owned_pod"])
                                   if may_terminate_training(state) else False)
            state["remaining_pods"] = [{k:p.get(k) for k in ("id","name","costPerHr")}
                                       for p in runpod.active_pods()]
            state["elapsed_s"] = time.time()-created
            state["estimated_gpu_usd"] = state["elapsed_s"]/3600*state.get("budget_hourly_usd",10)
            from account_snapshot import snapshot
            try:
                state["accounts_after"] = snapshot()
            finally:
                dump(out / "status.json", state)
            # Watchdog exits itself after observing the pod gone. Do not stop it
            # before termination is verified, including on errors in this driver.
        print(json.dumps({k:v for k,v in state.items() if k in
                          ("phase","owned_pod","terminated","estimated_gpu_usd","failure")}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--out", required=True)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(commands(json.loads(Path(args.plan).read_text(encoding='utf-8'))),indent=2))
    else:
        run(args.plan, args.out)
