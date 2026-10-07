# ABOUTME: Read-only remote monitor preserving small training evidence before owner teardown.
# ABOUTME: Run: uv run python scratch/plain_dose/preserve_evidence.py --seconds 7500
import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.infra.endpoints.vllm import SshExec

CAMPAIGN = ROOT / "output/plain_dose_campaign"
PROBE = r'''
import json
from pathlib import Path
root=Path('/root/work')
paths=list((root/'output/da-supervision').glob('*.log'))
paths+=list((root/'output/da-supervision').glob('*.exit'))
for pattern in ('*/run_meta.json','*/checkpoint-*/trainer_state.json','*/adapter/training_meta.json','*/adapter/train_config.yaml'):
 paths+=list((root/'output/train').glob(pattern))
out={}
for p in paths:
 if p.is_file() and not p.is_symlink():
  out[p.relative_to(root).as_posix()]=p.read_text(errors='replace')
print(json.dumps(out))
'''


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def collect(status_path):
    state = json.loads(status_path.read_text(encoding="utf-8"))
    destination = status_path.parent / "publication_evidence"
    destination.mkdir(exist_ok=True)
    write(destination / "owner_status.json", state)
    if not state.get("host") or state.get("terminated"):
        return {"arm": state["arm"]["key"], "phase": state["phase"], "copied": 0}
    remote = SshExec(state["host"], port=8000, workdir="/root/work")
    files = json.loads(remote._ssh("python3 -c " + shlex.quote(PROBE), timeout=35))
    inventory = []
    validations = []
    for relative, content in files.items():
        relative_path = Path(relative)
        assert not relative_path.is_absolute() and ".." not in relative_path.parts
        path = destination / "remote" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        inventory.append({"path": relative, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if relative.endswith("/run_meta.json") and "/smoke_" not in relative:
            meta = json.loads(content)
            expected = state["arm"]
            history = meta.get("log_history", [])
            expected_steps = expected.get("expected_steps") or {"da5": 123, "da25": 154}[expected["key"]]
            expected_rows = expected.get("expected_rows") or {"da5": 1960, "da25": 2458}[expected["key"]]
            finite = all(math.isfinite(float(value)) for row in history for key, value in row.items()
                         if key in ("loss", "grad_norm", "train_loss"))
            checks = {
                "dataset_pin": meta["dataset"]["revision"] == expected["data_revision"],
                "base_pin": meta["base_model_revision"] == state["plan"]["base_model_revision"],
                "finite_metrics": finite,
                "expected_steps": bool(history) and history[-1].get("step") == expected_steps,
                "epoch_one": bool(history) and history[-1].get("epoch") == 1,
                "expected_rows": meta["n_examples"] == expected_rows,
                "single_gpu": meta["world_size"] == 1,
            }
            validations.append({"path": relative, "checks": checks,
                                "verified": all(checks.values()), "final_metrics": history[-1] if history else None})
    receipt = {"time_utc": datetime.now(timezone.utc).isoformat(), "host": state["host"],
               "owned_pod": state["owned_pod"], "files": inventory, "validations": validations}
    write(destination / "receipt.json", receipt)
    return {"arm": state["arm"]["key"], "phase": state["phase"], "copied": len(files),
            "validations": validations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=7500)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        for path in sorted(CAMPAIGN.glob("da*_retry1/status.json")):
            try:
                result = collect(path)
            except Exception as exc:
                result = {"owner": str(path.parent), "error": type(exc).__name__, "message": str(exc)[:300]}
            print(json.dumps(result), flush=True)
        if args.once:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
