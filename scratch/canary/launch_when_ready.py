# ABOUTME: Reads a `runpod up --eval` log for the pod id and ssh host, waits until the pod's boot log says READY,
# ABOUTME: then starts `uv run evals ... --server <host> --terminate-pod` detached, so an eval never starts early.
# Run: nohup uv run python scratch/canary/launch_when_ready.py <up.log> <eval> <target[,target...]> <port> <eval.log> [extra evals args...] &
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    up_log, name, target, port, out_log, *extra = sys.argv[1:]
    text = ""
    while "host:" not in text and "Traceback" not in text:
        text = Path(up_log).read_text() if Path(up_log).exists() else ""
        time.sleep(10)
    if "Traceback" in text:
        sys.exit(f"rent failed: see {up_log}")
    host = re.search(r"root@[0-9.]+:[0-9]+", text).group(0)
    pod = re.search(r"down --pod ([a-z0-9]+)", text).group(1)
    print(f"pod {pod} at {host}", flush=True)
    while True:
        status = subprocess.run(
            ["uv", "run", "runpod", "status", "--pod", pod],
            capture_output=True,
            text=True,
        ).stdout
        if "READY" in status or "FAILED" in status:
            break
        time.sleep(20)
    if "FAILED" in status and "READY" not in status:
        sys.exit(f"pod {pod} boot FAILED")
    cmd = [
        "uv",
        "run",
        "evals",
        "--name",
        name,
        *extra,
        "--target",
        *target.split(","),
        "--server",
        host,
        "--port",
        port,
        "--terminate-pod",
    ]
    with open(out_log, "w") as f:
        subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(
        f"{datetime.now(timezone.utc):%H:%M} UTC started: {' '.join(cmd)}", flush=True
    )


if __name__ == "__main__":
    main()
