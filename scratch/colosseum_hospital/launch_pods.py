# ABOUTME: Provision one RunPod box per hospital arm and hand each a self-contained runner.
# ABOUTME: Pods expire provider-side, so the laptop can go off the moment this returns.
"""Launch the board-access hospital arms unattended.

One pod per arm rather than one pod running both: the September pieces put 15 seeds at
~1h35m, so 30 seeds x 2 arms is ~6h sequential and would be cut off by the 4h cap. Split
across two boxes each arm is ~3h10m and fits, for the same GPU-hours.

Credentials go to /root/.eval.env at 0600 and NEVER to /workspace, which the bootstrap
serves on a public HTTP proxy (`python3 -m http.server 8080`) -- which is also why
`--push-env` is not used here: it writes .env into exactly that directory.

Usage:
    ./.venv/bin/python scratch/colosseum_hospital/launch_pods.py
"""
import subprocess
import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values, load_dotenv

load_dotenv(ROOT / ".env")
from src.infra import runpod
from scratch.colosseum_hospital import deploy_runner

ENV = dotenv_values(ROOT / ".env")
ARMS = {
    "plain": "dougalldeepmind/2026-10-05-qwen36-0-plain",
    "da-15": "dougalldeepmind/2026-10-05-qwen36-0-da-15",
}
MAX_HOURS = 4.0
# The scheduled-provisioning path (terminateAfter) draws on a smaller host pool than the
# default one, so the profile's card can simply be unavailable -- it was for `plain` on the
# first attempt. Walk down rather than give up; H100 NVL is close to HBM3, the A100 is the
# slow last resort and risks the 4h cap truncating the second seed piece.
GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA A100 80GB PCIe"]


def ssh(ip: str, port: int, cmd: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run one command on the pod. `stdin` carries anything secret, never argv."""
    return subprocess.run(
        ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
         f"root@{ip}", cmd],
        input=stdin, text=True, capture_output=True, timeout=300,
    )


def ssh_retry(ip: str, port: int, cmd: str, stdin: str | None = None, tries: int = 20):
    """ssh, retrying while the pod's sshd is still being (re)installed by the bootstrap.

    The bootstrap apt-installs and restarts openssh-server after the pod answers, so an
    early connection is dropped mid-command and the next few are refused outright. Without
    this the runner silently never starts and the pod bills its whole lifetime for nothing.
    """
    last = None
    for _ in range(tries):
        last = ssh(ip, port, cmd, stdin)
        if last.returncode == 0:
            return last
        time.sleep(20)
    return last


def launch(arm: str, target: str, out: dict, gpus: list[str] | None = None) -> None:
    """Provision the arm's pod on the first card with capacity and start its runner."""
    ids: list[str] = []
    report = None
    for gpu in (gpus or GPUS):
        try:
            report = runpod.up(
                name=f"matboz-hosp-{arm}", eval="colosseum_hospital", target=target,
                clone_repo=True, max_hours=MAX_HOURS,
                gpu=gpu, on_provisioned=ids.append,
            )
            break
        except RuntimeError as e:
            if "resources to deploy" not in str(e):
                raise
            print(f">>> {arm}: no capacity on {gpu}, trying the next card", flush=True)
    if report is None:
        out[arm] = {"error": "no capacity on any card"}
        return
    pod_id = ids[0]
    ip, port = runpod._ssh_endpoint(pod_id)
    out[arm] = {"pod": pod_id, "host": f"{ip}:{port}", "report": report}
    print(f">>> {arm}: pod {pod_id} at {ip}:{port}, installing runner", flush=True)
    deploy_runner.main(arm, pod_id, ip, port)


def main() -> None:
    """Launch every arm in parallel and print where each one landed."""
    out: dict = {}
    wanted = sys.argv[1:] or list(ARMS)
    threads = [threading.Thread(target=launch, args=(a, ARMS[a], out)) for a in wanted]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for arm, d in out.items():
        print(f"\n=== {arm} ===")
        print(f"  pod      {d.get('pod')}   host {d.get('host')}")
        print(f"  creds    {d.get('creds')}")
        print(f"  runner   {d.get('runner')}")
        print(f"  pid      {d.get('pid')}")
        print(f"  boot log {runpod.boot_log_url(d['pod']) if d.get('pod') else '?'}")


if __name__ == "__main__":
    main()
