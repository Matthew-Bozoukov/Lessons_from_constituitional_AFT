# ABOUTME: Stop exactly the receipted Vast CPU VM at its absolute expiry, retaining disk.
# ABOUTME: Runs from Windows Task Scheduler or a guest systemd timer; never destroys instances.
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.eval.capabilities.swebench_mini.fleet_host import watchdog_deadline, deadline_value
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from vastai import VastAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    client = VastAI(api_key=dotenv_values(args.env)["VAST_API_KEY"].strip())
    instance = client.show_instance(int(receipt["instance_id"]))
    if not instance:
        print("Receipted VM no longer exists.")
        return
    if instance["label"] != receipt["label"]:
        raise RuntimeError("Ownership label changed; refusing to operate on VM")
    now = datetime.now(timezone.utc)
    deadline = watchdog_deadline(receipt)
    due = now.timestamp() >= deadline_value(deadline)
    if args.check or not due:
        print(json.dumps({"id": instance["id"], "status": instance["actual_status"],
                          "stop_at": deadline, "due": due, "lifetime": receipt.get("lifetime", "expiring")}))
        return
    if instance["actual_status"] == "stopped":
        print("Provider confirms VM stopped; retained storage still bills.")
        return
    result = client.stop_instance(instance["id"])
    if not result.get("success"):
        raise RuntimeError("Provider did not accept STOP")
    print("Provider accepted STOP; next timer tick verifies stopped state.")


if __name__ == "__main__":
    main()
