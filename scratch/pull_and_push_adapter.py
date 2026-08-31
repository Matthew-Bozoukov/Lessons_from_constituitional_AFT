# ABOUTME: Pulls a finished LoRA adapter off the training pod and pushes it to HF with the
# ABOUTME: thinking stamp, so the pod can be destroyed without losing the run.

"""Get the adapter off the pod, then to the Hub.

`push_to_hub` was off for the run so no HF token sat on a rented box, which means the pod is
the only copy until this runs. Verifies the stamp is present before pushing: run_eval hard
errors on an adapter without training_meta.json, and backfilling it later has cost this
project two manual retrievals.

Run: uv run python scratch/pull_and_push_adapter.py <pod-id> <ssh-ip> <ssh-port>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import hf_api  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

load_dotenv()

POD, IP, PORT = sys.argv[1], sys.argv[2], sys.argv[3]
REMOTE_ROOT = "/workspace/out/train_t2_9000_synthdoc_1000_dynbatch"
PRIMARY_REPO = "LASR-Callum/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64-dynbatch"
# The HF token is fine-grained and scoped to the matboz user: org repo creation 403s (hit
# on 2026-08-09 pushing the SURF results). Fall back rather than fail at the last step.
FALLBACK_REPO = "matboz/qwen3.6-27b-lora-t2-9000-synthdoc-1000-r64-dynbatch"
LOCAL = Path(f"output/adapters/t2_9000_synthdoc_1000_dynbatch_{timestamp()}")

SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-p", PORT, f"root@{IP}"]


def sh(cmd, **kw):
    """Run a command, returning stdout; raises on failure."""
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw).stdout


def main():
    """Pull the adapter, verify the stamp, push, and report the URL."""
    run_dir = sh(SSH + [f"ls -d {REMOTE_ROOT}/*/ | tail -1"]).strip().rstrip("/")
    print(f"remote run dir: {run_dir}")
    listing = sh(SSH + [f"ls {run_dir}"])
    print(f"contents:\n{listing}")

    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o",
                    "UserKnownHostsFile=/dev/null", "-P", PORT, "-r",
                    f"root@{IP}:{run_dir}", str(LOCAL)], check=True)
    print(f"pulled to {LOCAL}")

    # The adapter may sit at the run dir root or in an `adapter/` subdir depending on how
    # the trainer finished; find the one that actually holds weights.
    cands = [p for p in [LOCAL, LOCAL / "adapter"] if (p / "adapter_model.safetensors").exists()]
    if not cands:
        cands = [p.parent for p in LOCAL.rglob("adapter_model.safetensors")]
    assert cands, f"no adapter_model.safetensors under {LOCAL}"
    adapter = cands[0]
    print(f"adapter dir: {adapter}")

    stamp = adapter / "training_meta.json"
    assert stamp.exists(), f"NO training_meta.json in {adapter} — run_eval will refuse it"
    meta = json.loads(stamp.read_text())
    print(f"stamp: {meta}")
    assert meta.get("thinking") is True, "thinking stamp is not true"

    api = hf_api()
    for repo in (PRIMARY_REPO, FALLBACK_REPO):
        try:
            api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(folder_path=str(adapter), repo_id=repo, repo_type="model")
            print(f"\nPUSHED: https://huggingface.co/{repo}")
            (LOCAL / "push_meta.json").write_text(json.dumps({
                "git_sha": git_sha(), "pod": POD, "repo": repo,
                "remote_dir": run_dir, "training_meta": meta,
            }, indent=2))
            return
        except Exception as e:  # noqa: BLE001 - try the fallback namespace
            print(f"push to {repo} failed: {type(e).__name__}: {str(e)[:120]}")
    raise SystemExit("both namespaces failed")


if __name__ == "__main__":
    main()
