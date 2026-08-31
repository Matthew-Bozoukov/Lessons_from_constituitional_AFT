# ABOUTME: Measures the true OpenRouter cost of ONE Petri seed at max_turns=8 with a Kimi
# ABOUTME: auditor and deepseek-v4-flash judge/realism, by reading /credits either side.

"""One seed, measured rather than estimated.

Every role runs on OpenRouter, so the /credits delta is the whole cost of the audit —
auditor, judge, realism and target together. The endpoint lags, so the final read waits
before sampling (the convention in docs/EXPENDITURE.md).

Run: uv run python scratch/petri_cost_probe.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, timestamp  # noqa: E402

# The SURF nested .env, not the repo root one: this task's spend is all on the key supplied
# for it, and the two are different accounts (root key usage ~433, supplied key ~1091), so
# mixing them would make the /credits delta meaningless.
load_dotenv("src/eval/audits/surf/third_party/SURF/.env", override=True)

ENV = Path("/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/"
           "6acf579c-f9ed-4a99-85f6-3aecd744034e/scratchpad/petri-env")
SEED_DIR = Path("src/eval/audits/petri/petri-subscription/seed-one").resolve()
OUT = Path(f"output/petri/cost_probe_{timestamp()}")

AUDITOR = sys.argv[1] if len(sys.argv) > 1 else "openrouter/moonshotai/kimi-k2.5"
JUDGE = "openrouter/deepseek/deepseek-v4-flash"
REALISM = "openrouter/deepseek/deepseek-v4-flash"
# The target is a stand-in, not our adapter: this probe measures the AUDIT's cost, and a
# self-hosted target would add GPU time without changing what is being measured. Same
# family and size as the adapter's base so transcript length is representative.
TARGET = "openrouter/qwen/qwen3.6-27b"
MAX_TURNS = 8


def credits():
    """Account-level (total_credits, total_usage) from OpenRouter."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    return d["total_credits"], d["total_usage"]


def main():
    """Run one seed and report measured spend."""
    OUT.mkdir(parents=True, exist_ok=True)
    _, before = credits()
    print(f"usage before: {before:.4f}")

    cmd = [
        str(ENV / ".venv/bin/inspect"), "eval", "inspect_petri/audit",
        "-T", f"seed_instructions={SEED_DIR}",
        "-T", f"max_turns={MAX_TURNS}",
        "-T", "realism_filter=0.6",
        "-T", "enable_rollback=True",
        "-T", "target_tools=synthetic",
        "--model-role", f"auditor={AUDITOR}",
        "--model-role", f"judge={JUDGE}",
        "--model-role", f"realism={REALISM}",
        "--model-role", f"target={TARGET}",
        "--model", TARGET,
        "--epochs", "1",
        "--max-connections", "1",
        "--log-dir", str(OUT / "logs"),
        "--log-format", "eval",
        "--temperature", "0.7",
    ]
    print(" ".join(cmd[1:]) + "\n")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"]})
    elapsed = time.time() - t0
    (OUT / "stdout.log").write_text(proc.stdout)
    (OUT / "stderr.log").write_text(proc.stderr)
    print(proc.stdout[-2500:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2500:])

    print(f"\nexit {proc.returncode}, {elapsed:.0f}s — waiting 45s for /credits to settle")
    time.sleep(45)
    _, after = credits()
    spent = after - before

    (OUT / "cost.json").write_text(json.dumps({
        "git_sha": git_sha(), "auditor": AUDITOR, "judge": JUDGE, "realism": REALISM,
        "target": TARGET, "max_turns": MAX_TURNS, "epochs": 1, "seeds": 1,
        "usage_before": before, "usage_after": after, "measured_cost_usd": round(spent, 4),
        "wall_clock_s": round(elapsed, 1), "returncode": proc.returncode,
    }, indent=2))

    print(f"usage after:  {after:.4f}")
    print(f"MEASURED COST FOR ONE SEED: ${spent:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
