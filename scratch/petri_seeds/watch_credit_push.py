# ABOUTME: Watches a running Petri driver: stops the audit when OpenRouter credit runs out (the driver
# ABOUTME: then tears its pod down), and once the driver exits pushes every scored seed to HF.
"""Credit watchdog + publisher for a laptop-driven Petri audit.

    uv run python scratch/petri_seeds/watch_credit_push.py --driver_log <..._petri_plainconst.log> \\
        --run_dir output/petri/<run> --target <hf> --repo <name> [--preamble <file>] [--dry]

The pod is torn down by the driver script itself (it runs `runpod down` whenever the audit exits);
this only ends the audit early, so a run with no credit does not keep the H200 billing. The push
reads the local .eval log, so it needs no pod. `--dry` checks once and prints what it would push.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import fire
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scratch" / "petri_seeds"))
from push_results import SEEDS, valid_scores  # noqa: E402
from push_results import main as push  # noqa: E402

CREDIT_ERRORS = re.compile(r"Error code: 402|insufficient credits|requires more credits", re.I)


def credit_left() -> float:
    """OpenRouter balance in dollars, from the .env key (never the shell's stale export)."""
    key = dotenv_values(ROOT / ".env")["OPENROUTER_API_KEY"]
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    return d["total_credits"] - d["total_usage"]


def pids(pattern: str) -> list[int]:
    """Process ids whose command line matches `pattern`."""
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split() if int(p) != os.getpid()]


def main(driver_log: str, run_dir: str, target: str, repo: str, preamble: str = "",
         context_window: int = 0, min_credit: float = 0.5, every: int = 60,
         driver_pattern: str = "bash scratch/petri_seeds/run_plainconst_petri.sh",
         audit_pattern: str = "run_audit.py --server", dry: bool = False) -> None:
    """Stop the audit on credit exhaustion; push the scored seeds once the driver has exited.

    Args:
        driver_log: The driver's tee'd log; its `_audit.txt` sibling is scanned for 402s.
        run_dir: The audit's run directory (holds logs/*.eval).
        target: HF model audited, recorded in the card.
        repo: Dataset name under HF_ORG.
        preamble: Constitution file prepended to target system messages, recorded in the card.
        context_window: Served context window, recorded in the card.
        min_credit: Balance in dollars below which the audit is stopped.
        every: Seconds between checks.
        driver_pattern: pgrep pattern for the driver script.
        audit_pattern: pgrep pattern for THIS run's audit process, so a second audit running
            alongside is never interrupted.
        dry: Check once, print the decision and what would be pushed, change nothing.
    """
    audit_txt = Path(str(driver_log).removesuffix(".log") + "_audit.txt")
    stopped = False
    while True:
        alive = bool(pids(driver_pattern))
        left = credit_left()
        hit = audit_txt.exists() and bool(CREDIT_ERRORS.search(audit_txt.read_text(errors="ignore")))
        print(f">>> {time.strftime('%H:%M:%S')} driver={'up' if alive else 'exited'} "
              f"credit=${left:.2f} credit_errors={hit}", flush=True)
        if alive and not stopped and (left < min_credit or hit):
            audit = pids(audit_pattern)
            print(f"!!! OpenRouter credit exhausted -- SIGINT to the audit {audit}; "
                  f"the driver then tears its pod down", flush=True)
            if not dry:
                for p in audit:
                    os.kill(p, signal.SIGINT)
                stopped = True
        if dry or not alive:
            break
        time.sleep(every)

    scores, _ = valid_scores(list(Path(run_dir, "logs").glob("*.eval")))
    missing = sorted({p.stem for p in SEEDS.glob("*.md")} - set(scores))
    print(f">>> scored {len(scores)}/90, missing {len(missing)}", flush=True)
    if dry:
        print(f">>> dry: would push {run_dir} -> {repo}", flush=True)
        return
    assert scores, "nothing scored; nothing to push"
    push(run_dirs=run_dir, target=target, repo=repo, allow_missing=missing,
         preamble=preamble, context_window=context_window)
    from huggingface_hub import HfApi
    api = HfApi(token=dotenv_values(ROOT / ".env")["HF_TOKEN"])
    full = f"{dotenv_values(ROOT / '.env')['HF_ORG']}/{repo}"
    for f in (Path(driver_log), audit_txt):
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"logs/{f.name}",
                            repo_id=full, repo_type="dataset")
    print(f">>> driver logs pushed to {full}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
