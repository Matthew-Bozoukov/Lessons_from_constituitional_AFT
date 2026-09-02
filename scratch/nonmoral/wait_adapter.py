# ABOUTME: Block until the trained adapter's real files are on HF, so teardown never races the
# ABOUTME: upload. Run: uv run python scratch/nonmoral/wait_adapter.py --repo <org/name>
"""Poll an HF model repo until it holds actual adapter weights, not just .gitattributes.

A pod's container disk is not storage: everything on it dies with it. The repo existing is NOT
evidence the adapter arrived -- `create_repo` runs before the upload, so a half-finished push
looks exactly like a finished one from the repo listing. This waits for the files that matter.
"""
import time

import fire

from huggingface_hub import HfApi
from src.huggingface import hf_token

WANT = ("adapter_model.safetensors", "adapter_config.json", "training_meta.json")


def main(repo: str, timeout_s: int = 3600, every_s: int = 60) -> None:
    api = HfApi(token=hf_token())
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            files = {s.rfilename for s in api.model_info(repo).siblings}
        except Exception as exc:                      # repo may not exist yet
            print(f"  not readable yet: {type(exc).__name__}", flush=True)
            files = set()
        missing = [w for w in WANT if w not in files]
        if not missing:
            print(f"ADAPTER COMPLETE on {repo}")
            for f in sorted(files):
                print("  ", f)
            return
        print(f"  waiting — have {len(files)} file(s), missing {missing}", flush=True)
        time.sleep(every_s)
    raise SystemExit(f"TIMEOUT after {timeout_s}s — adapter still incomplete on {repo}. "
                     "DO NOT tear the pod down; inspect /root/work/train.log.")


if __name__ == "__main__":
    fire.Fire(main)
