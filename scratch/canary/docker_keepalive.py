# ABOUTME: Pings Docker Desktop every 20 s so it never idles into Resource Saver while an ODCV run waits for
# ABOUTME: its model to load (a failed wake-up quit Docker and failed all 240 cells on 2026-10-01).
# Run: nohup uv run python scratch/canary/docker_keepalive.py [hours] &
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = Path("output/canary/logs/docker-keepalive.log")

if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    end = time.time() + hours * 3600
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while time.time() < end:
        ok = subprocess.run(["docker", "ps", "-q"], capture_output=True).returncode == 0
        if not ok:
            with open(LOG, "a") as f:
                f.write(f"{datetime.now(timezone.utc):%H:%M:%S} docker not answering\n")
        time.sleep(20)
