# ABOUTME: Owns one bounded RunPod inference rental and its local ODCV evaluation process.
# ABOUTME: Registers independent guards immediately, monitors progress and tears down only its recorded pod.
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.docker import docker_preflight, require_lf_shell_scripts
from src.infra import runpod
from src.infra.endpoints.vllm import POD_VENV, SshExec, resolve_target


def main(key, resume=False):
    plan = OmegaConf.load("scratch/da_supervision/odcv_plan.yaml")
    arm = next(a for a in plan.arms if a.key == key)
    out = Path(plan.output_root) / key
    out.mkdir(parents=True, exist_ok=True)
    status = out / "status.json"
    assert resume or not status.exists(), f"Existing owner state at {status}; inspect before any relaunch"
    state = json.loads(status.read_text(encoding="utf-8")) if resume else {}
    state.update(phase="preflight", pid=os.getpid(), arm=OmegaConf.to_container(arm))
    pod = state.get("owned_pod") if resume else None
    guard = None
    child = None

    def save(**updates):
        state.update(updates, updated_epoch=time.time())
        temporary = status.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(status)

    def owned(pod_id):
        nonlocal pod, guard
        pod = pod_id
        save(phase="booting", owned_pod=pod, created_epoch=time.time())
        guard = runpod.start_watchdog(pod, int(plan.max_hours * 3600), out / "watchdog.log")
        info = runpod.call("GET", f"/pods/{pod}")
        price = float(info["costPerHr"])
        save(hourly_usd=price, watchdog_pid=guard.pid)
        assert price <= float(plan.hourly_ceiling_usd), f"Price {price} exceeds campaign ceiling"

    try:
        save()
        if resume:
            info = runpod.call("GET", f"/pods/{pod}")
            assert info["name"] == arm.pod_name and info["gpuCount"] == 1
            remaining = int(float(info["env"]["LASR_POD_DEADLINE"]) - time.time())
            assert remaining > 0
            guard = runpod.start_watchdog(pod, remaining, out / "recovered-watchdog.log")
            save(watchdog_pid=guard.pid, recovered_epoch=time.time(), original_deadline=float(info["env"]["LASR_POD_DEADLINE"]))
        docker_preflight()
        require_lf_shell_scripts("src/eval/misalignment/odcv/third_party/odcv-bench")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(arm.port)))
        spec = resolve_target(str(arm.target))
        assert spec.revision == arm.revision and spec.base_revision == arm.base_revision
        assert spec.mode == "think"
        result = (out / "provision.txt").read_text(encoding="utf-8") if resume else runpod.up(
            name=str(arm.pod_name), eval=str(arm.target), gpu=str(plan.gpu), count=int(plan.count),
            cloud=str(plan.cloud), max_hours=float(plan.max_hours), push_env=True, on_provisioned=owned)
        (out / "provision.txt").write_text(result, encoding="utf-8")
        host = next(line.split(None, 1)[1] for line in result.splitlines() if line.startswith("host:"))
        remote = SshExec(host, port=int(arm.port))
        if resume:
            assert (out / "resume_pass.txt").exists() or not list((out / "eval").rglob("messages_record.txt")), "Existing rollouts require an explicit same-pass resume"
            remote.stop_server()
        save(host=host)
        boot_deadline = state["created_epoch"] + int(plan.boot_timeout_s)
        while time.time() < boot_deadline:
            text = remote._ssh("tail -c 3500 /workspace/boot.log", timeout=30)
            save(boot_tail=text)
            if any(line.startswith("READY") for line in text.splitlines()):
                break
            time.sleep(20)
        else:
            raise TimeoutError("Bootstrap exceeded bounded setup window")
        cuda = remote._ssh(f"{POD_VENV}/bin/python -c " + shlex.quote(
            "import torch; assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count()==1; "
            "x=torch.ones(1,device='cuda');print(torch.cuda.get_device_name(),x.item())"), timeout=60)
        save(phase="evaluating", cuda_check=cuda)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
        cmd = [sys.executable, "-m", "scratch.da_supervision.odcv_eval", "--name", "odcv",
               "--config", str(plan.config), "--target", str(arm.target), "--server", host,
               "--port", str(arm.port), "--server-bind", "0.0.0.0", "--terminate-pod",
               f"output_root={out.as_posix()}/eval"]
        if (out / "resume_pass.txt").exists():
            existing = Path((out / "resume_pass.txt").read_text(encoding="utf-8"))
            assert existing.is_relative_to(out.absolute()) and (existing / "workspaces").is_dir()
            cmd.append(f"campaign_resume_pass={existing.as_posix()}")
        with (out / "eval.log").open("wb") as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        save(eval_pid=child.pid, command=cmd)
        last_progress = time.time()
        old_marker = None
        while child.poll() is None:
            files = list((out / "eval").rglob("messages_record.txt"))
            logs = list((out / "eval").rglob("docker_output.log"))
            marker = (len(files), sum(p.stat().st_size for p in logs), (out / "eval.log").stat().st_size)
            if marker != old_marker:
                old_marker = marker
                last_progress = time.time()
            tail = (out / "eval.log").read_text(encoding="utf-8", errors="replace")[-3500:]
            save(transcript_files=len(files), last_progress_epoch=last_progress, eval_tail=tail)
            if time.time() - last_progress > int(plan.idle_timeout_s):
                raise TimeoutError("No local evaluation progress within the health-check bound")
            if time.time() > state["created_epoch"] + float(plan.max_hours) * 3600:
                raise TimeoutError("Original campaign lifetime reached")
            time.sleep(20)
        save(phase="complete" if child.returncode == 0 else "failed", eval_exit=child.returncode)
        if child.returncode:
            raise RuntimeError(f"Evaluation exited {child.returncode}; preserve local evidence")
    except BaseException as exc:
        response = getattr(exc, "response", None)
        detail = response.text[:1000] if response is not None else None
        save(phase="failed", error=f"{type(exc).__name__}: {exc}", provider_error=detail)
        traceback.print_exc()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        if pod:
            try:
                runpod.teardown(pod)
                save(terminated=True, termination_verified_epoch=time.time())
                if guard:
                    guard.terminate()
            except BaseException as exc:
                save(teardown_error=f"{type(exc).__name__}: {exc}")
                raise


if __name__ == "__main__":
    main(sys.argv[1], resume="--resume" in sys.argv[2:])
