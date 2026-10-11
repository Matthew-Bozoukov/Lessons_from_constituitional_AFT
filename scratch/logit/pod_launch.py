# ABOUTME: Waits for a `runpod up --train` pod to say READY, copies output/logit/turns there, checks out this branch's
# ABOUTME: head, and starts one model's scorer over the given rollout sets detached. One call per pod.
# Run: nohup uv run python -m scratch.logit.pod_launch <up.log> <models,comma,separated> <rollouts,comma,separated> [evals,comma,separated] > <launch.log> &
import re
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    up_log, model, rollouts = sys.argv[1:4]
    evals = sys.argv[4] if len(sys.argv) > 4 else "mask,odcv"
    text = ""
    while "host:" not in text and "Traceback" not in text:
        text = Path(up_log).read_text() if Path(up_log).exists() else ""
        time.sleep(15)
    if "Traceback" in text and "host:" not in text:
        sys.exit(f"rent failed: see {up_log}")
    host = re.search(r"root@([0-9.]+):([0-9]+)", text)
    ip, port = host.group(1), host.group(2)
    pod = re.search(r"down --pod ([a-z0-9]+)", text).group(1)
    print(f"pod {pod} at root@{ip}:{port}", flush=True)
    while True:
        status = subprocess.run(["uv", "run", "runpod", "status", "--pod", pod], capture_output=True, text=True).stdout
        if "READY" in status or "FAILED" in status:
            break
        time.sleep(30)
    if "FAILED" in status and "READY" not in status:
        sys.exit(f"pod {pod} boot FAILED")
    ssh = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", "-p", port, f"root@{ip}"]
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-P", port, "output/logit/turns_all.tgz",
                    f"root@{ip}:/root/work/turns_all.tgz"], check=True)
    sets = " ".join(rollouts.split(","))
    ev = " ".join(evals.split(","))
    script = "set -a; . ./.env 2>/dev/null; set +a\n" + "".join(
        f"uv run --no-sync python -m scratch.logit.score --model {m} --rollouts {sets} --eval {ev}\n"
        for m in model.split(","))
    remote = ("cd /root/work && git fetch -q origin jamie/logit && git checkout -q origin/jamie/logit && "
              "mkdir -p output/logit && tar xzf turns_all.tgz -C output/logit 2>/dev/null; "
              f"printf '%s' '{script}' > run_{model.replace(',', '_')}.sh; "
              f"nohup bash run_{model.replace(',', '_')}.sh > output/logit/{model.replace(',', '_')}_run.log 2>&1 </dev/null & disown; "
              "git log --oneline -1")
    out = subprocess.run(ssh + [remote], capture_output=True, text=True)
    print(f"{time.strftime('%H:%M')} UTC launched {model} on {sets}: {out.stdout.strip()} {out.stderr.strip()[-200:]}", flush=True)


if __name__ == "__main__":
    main()
