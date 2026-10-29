# ABOUTME: Read-only provider spend and pod inventory for the authorized overnight work.
# ABOUTME: Records selected billing fields only, never authentication headers or credentials.
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.infra import runpod


def snapshot():
    load_dotenv()
    out = {"time_utc": datetime.now(timezone.utc).isoformat()}
    out["runpod"] = runpod.account_spend()  # v2 has no balance route: spend + burn
    response = requests.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        timeout=30,
    )
    out["openrouter_http_status"] = response.status_code
    if response.ok:
        data = response.json().get("data", {})
        out["openrouter"] = {k: data.get(k) for k in ("total_credits", "total_usage")}
    out["pods"] = [
        {k: p.get(k) for k in ("id", "name", "gpu", "cost", "createdAt", "status")}
        for p in runpod.active_pods()
    ]
    return out


if __name__ == "__main__":
    data = snapshot()
    dest = Path(sys.argv[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
