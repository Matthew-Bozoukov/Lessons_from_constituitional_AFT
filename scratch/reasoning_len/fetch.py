# ABOUTME: Download matched sycophancy + debate-speeches eval rollouts for all five arms from HF.
# ABOUTME: One-off analysis input for the inference-time reasoning-length measurement.
import dotenv, os, time

dotenv.load_dotenv(".env")
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor

ARMS = {
    "MO_DA": "qwen3-6-27b-lora-t2-9284-da716-r64-dynbatch",
    "MO_CR": "qwen3-6-27b-lora-t2-9284-courtroom716-r64-dynbatch",
    "MO_PC": "qwen3-6-27b-lora-t2-9284-peercritique716-r64-dynbatch",
    "CTRL": "qwen3-6-27b-lora-table2-only-9284-r64",
    "BASE": "Qwen3-6-27B",
}
EVALS = ["sycophancy", "debate-speeches"]
jobs = [
    (e, a, f"LASR-Callum/2026-08-17-{e}-{s}") for e in EVALS for a, s in ARMS.items()
]


def go(j):
    e, a, r = j
    last = None
    for attempt in range(8):
        try:
            p = snapshot_download(
                r,
                repo_type="dataset",
                local_dir=f"data/reasoning_len/{e}/{a}",
                max_workers=4,
                allow_patterns=[
                    "rollouts/*",
                    "records.jsonl",
                    "results.json",
                    "run_meta.json",
                ],
            )
            n = len(os.listdir(os.path.join(p, "rollouts")))
            return f"{e} {a} -> {p} ({n} rollouts)"
        except Exception as exc:  # transient DNS / connect flakes
            last = exc
            time.sleep(5 * (attempt + 1))
    return f"{e} {a} FAILED {type(last).__name__}: {last}"


for j in jobs:
    print(go(j), flush=True)
