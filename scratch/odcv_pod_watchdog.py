# ABOUTME: Pod-SCOPED billing watchdog: destroys exactly ONE RunPod pod on deadline or stale heartbeat.
# ABOUTME: Run: uv run python scratch/odcv_pod_watchdog.py --pod <id> --heartbeat <file> [--max_minutes 300]

"""Kill one rented pod if the orchestration that rented it stops saying it is alive.

CLAUDE.md requires a GPU run to register with a watchdog before doing work and not to rely
on the orchestrating process surviving to clean up. This is that watchdog.

It takes ONE pod id and can only ever destroy that id. It NEVER lists the account and never
acts on anything it did not receive explicitly, because this RunPod account is SHARED — a
sweep would kill teammates' pods. (Two unrelated pods were live when this run started.)

Two independent triggers, whichever fires first:
  - deadline:  a hard maximum lifetime, so a wedged run still stops billing.
  - heartbeat: the driver touches a file; if it goes stale the orchestration is gone.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import fire
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

API = "https://rest.runpod.io/v1"


def _destroy(pod: str, why: str) -> None:
    """Terminate the one pod this watchdog owns, and say why."""
    key = os.environ["RUNPOD_API_KEY"]
    r = requests.delete(f"{API}/pods/{pod}", headers={"Authorization": f"Bearer {key}"},
                        timeout=60)
    print(f"[watchdog] DESTROY {pod} ({why}) -> HTTP {r.status_code} {r.text[:200]}",
          flush=True)


def main(pod: str, heartbeat: str, max_minutes: float = 300.0,
         stale_minutes: float = 45.0, poll_s: float = 60.0) -> None:
    """Watch one pod; destroy it on deadline or stale heartbeat.

    Args:
        pod: The single pod id this watchdog may destroy. Nothing else is ever touched.
        heartbeat: File whose mtime the orchestration refreshes while it is alive.
        max_minutes: Hard lifetime cap from watchdog start.
        stale_minutes: Destroy if the heartbeat has not been touched in this long.
        poll_s: Seconds between checks.
    """
    hb = Path(heartbeat)
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.touch()
    t0 = time.time()
    deadline = t0 + max_minutes * 60
    print(f"[watchdog] armed for pod {pod} ONLY | deadline {max_minutes:.0f}m | "
          f"stale {stale_minutes:.0f}m | heartbeat {hb}", flush=True)
    while True:
        time.sleep(poll_s)
        now = time.time()
        if now > deadline:
            _destroy(pod, f"hard deadline {max_minutes:.0f}m reached")
            return
        age_m = (now - hb.stat().st_mtime) / 60 if hb.exists() else 1e9
        if age_m > stale_minutes:
            _destroy(pod, f"heartbeat stale {age_m:.1f}m > {stale_minutes:.0f}m")
            return
        print(f"[watchdog] ok  elapsed {(now-t0)/60:>5.1f}m  heartbeat age {age_m:>4.1f}m",
              flush=True)


if __name__ == "__main__":
    fire.Fire(main)
