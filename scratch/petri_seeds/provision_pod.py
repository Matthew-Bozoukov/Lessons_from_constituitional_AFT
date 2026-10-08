# ABOUTME: Rent one inference pod for a Petri constitution audit and record its ssh endpoint.
# ABOUTME: A real file, not a stdin heredoc: find_dotenv() walks caller frames and dies on `<string>`.
import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# Explicit path rather than find_dotenv(), which needs a real caller frame.
load_dotenv(REPO / ".env", override=True)

from src.infra.runpod import _ssh_endpoint, up  # noqa: E402


def main(name: str, target: str, out: str = "/tmp/petri_pod.json",
         max_hours: float = 6.0) -> None:
    """Provision a vLLM inference pod for `target` and write its endpoint to `out`.

    Args:
        name: Pod name; prefix it so the owner is obvious in the RunPod console.
        target: HF adapter the audit will serve.
        out: Where to record pod id, ip and ssh port.
        max_hours: Watchdog ceiling.
    """
    def record(pod_id: str) -> None:
        ip, port = _ssh_endpoint(pod_id)
        Path(out).write_text(json.dumps({"pod_id": pod_id, "ip": ip, "port": port}))
        print(f">>> recorded {pod_id} {ip}:{port}", flush=True)

    up(name=name, eval="mask", target=target, max_hours=max_hours, on_provisioned=record)
    print(">>> POD READY", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
