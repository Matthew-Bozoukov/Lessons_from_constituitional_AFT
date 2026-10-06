# ABOUTME: Install the self-terminating runner on an already-provisioned hospital pod.
# ABOUTME: Idempotent: waits for sshd, replaces creds and runner, restarts, verifies the PID.
"""Put the real teardown on a pod that is already billing.

RunPod ignores `terminateAfter`, and `up`'s watchdog dies with the laptop, so the pod
terminating ITSELF is the only cap that survives the laptop going off. That makes this
script the thing standing between an unattended run and an open-ended bill, so it verifies
rather than assumes: it confirms sshd answers, confirms the runner file landed, and
confirms a live PID before reporting success.

Usage:
    ./.venv/bin/python scratch/colosseum_hospital/deploy_runner.py <arm> <pod_id> <ip> <port>
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import dotenv_values

ENV = dotenv_values(ROOT / ".env")
TARGETS = {
    "plain": "dougalldeepmind/2026-10-05-qwen36-0-plain",
    "da-15": "dougalldeepmind/2026-10-05-qwen36-0-da-15",
}
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]


def ssh(ip, port, cmd, stdin=None, timeout=300):
    """One command on the pod; anything secret travels on stdin, never argv."""
    return subprocess.run(["ssh", "-p", str(port), *SSHO, f"root@{ip}", cmd],
                          input=stdin, text=True, capture_output=True, timeout=timeout)


def main(arm: str, pod_id: str, ip: str, port: int) -> None:
    """Wait for the pod, install the runner and credentials, start it, verify it lives."""
    for i in range(1, 121):
        r = ssh(ip, port, "echo UP", timeout=40)
        if r.returncode == 0:
            print(f">>> sshd answered on attempt {i}", flush=True)
            break
        if i % 10 == 0:
            print(f"    attempt {i}: still no sshd", flush=True)
        time.sleep(20)
    else:
        print(f"!!! {arm}: sshd never answered. Pod {pod_id} is BILLING with no runner.")
        return

    creds = "\n".join([
        f"export HF_TOKEN={ENV['HF_TOKEN_MATBOZ']}",
        f"export HF_ORG={ENV.get('HF_ORG', 'dougalldeepmind')}",
        f"export OPENROUTER_API_KEY={ENV['OPENROUTER_API_KEY']}",
        f"export RUNPOD_API_KEY={ENV['RUNPOD_API_KEY']}",
        f"export POD_ID={pod_id}", f"export ARM={arm}", f"export TARGET={TARGETS[arm]}", "",
    ])
    r = ssh(ip, port, "umask 077 && cat > /root/.eval.env && chmod 600 /root/.eval.env && echo CREDS_OK",
            stdin=creds)
    print(f"  creds:  {r.stdout.strip() or r.stderr.strip()[:160]}", flush=True)

    runner = (ROOT / "scratch/colosseum_hospital/pod_runner.sh").read_text()
    r = ssh(ip, port, "cat > /root/run_arm.sh && chmod +x /root/run_arm.sh && "
                      "grep -c self_terminate /root/run_arm.sh", stdin=runner)
    print(f"  runner: self_terminate mentions = {r.stdout.strip() or r.stderr.strip()[:160]}", flush=True)

    # Replace whatever the first launch may have started: an older runner without the
    # teardown would leave the pod billing after it finished.
    ssh(ip, port, "pkill -f run_arm.sh; sleep 2; true")
    ssh(ip, port, "cd /root && setsid nohup bash /root/run_arm.sh </dev/null "
                  ">/root/nohup.out 2>&1 & disown; true")
    time.sleep(5)
    r = ssh(ip, port, "pgrep -f run_arm.sh | tr '\\n' ' '")
    pids = r.stdout.strip()
    print(f"  PIDS:   [{pids}]  <- must be non-empty", flush=True)
    r = ssh(ip, port, "tail -3 /root/run.log 2>/dev/null")
    print(f"  log:    {r.stdout.strip()[:300]}", flush=True)
    print(f">>> {arm} {'OK' if pids else 'FAILED TO START — pod still billing'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
