# ABOUTME: Runs one Petri seed for N epochs and prices it from the eval log's per-model token
# ABOUTME: usage against live OpenRouter rates. Supersedes the /credits probe.

"""Run a Petri seed and cost it properly.

The earlier probe read OpenRouter's /credits either side of a run. That is unreliable here:
the endpoint lags by minutes so charges land in the next run's window, and the key is a
shared account whose other traffic lands in the window too — measured deltas came out ~10x
the true cost. Pricing the eval log's own token counts is exact and immune to both.

One accounting subtlety: Anthropic reports cache reads INSIDE input_tokens, other providers
report them alongside. Double-counting Sonnet's input is what made GLM look cheap by
comparison, so the split is handled per family below.

Run: uv run python scratch/petri_run.py <seed.md|seed_dir> [epochs] [auditor] [max_turns]
"""

import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, timestamp  # noqa: E402

load_dotenv("src/eval/audits/surf/third_party/SURF/.env", override=True)

ENV = Path("/tmp/claude-1000/-home-matthewb-git-repos-teaching-claude-why-replication/"
           "6acf579c-f9ed-4a99-85f6-3aecd744034e/scratchpad/petri-env")
SEED_ARG = Path(sys.argv[1])
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
AUDITOR = sys.argv[3] if len(sys.argv) > 3 else "openrouter/z-ai/glm-5.2"
MAX_TURNS = int(sys.argv[4]) if len(sys.argv) > 4 else 8

JUDGE = "openrouter/deepseek/deepseek-v4-flash"
REALISM = "openrouter/deepseek/deepseek-v4-flash"
# Target: an OpenRouter id, or "pod:<pod-id>:<served-name>" for an adapter served on a
# RunPod box. The latter becomes inspect's openai-api provider, which reads the endpoint
# from <NAME>_BASE_URL / <NAME>_API_KEY — here VLLM_*.
TARGET_ARG = sys.argv[5] if len(sys.argv) > 5 else "openrouter/qwen/qwen3.6-27b"
if TARGET_ARG.startswith("pod:"):
    _, pod_id, served = TARGET_ARG.split(":")
    TARGET = f"openai-api/vllm/{served}"
    TARGET_ENV = {"VLLM_BASE_URL": f"https://{pod_id}-8000.proxy.runpod.net/v1",
                  "VLLM_API_KEY": "dummy"}
else:
    TARGET, TARGET_ENV = TARGET_ARG, {}

OUT = Path(f"output/petri/run_{SEED_ARG.stem}_{timestamp()}")


def rates():
    """Live OpenRouter $/token for input, output and cache-read, keyed by model id."""
    out = {}
    for m in requests.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]:
        p = m["pricing"]
        out[m["id"]] = (float(p["prompt"]), float(p["completion"]),
                        float(p.get("input_cache_read") or 0))
    return out


def price(log, rate):
    """Cost of one eval log, per model, from its recorded token usage."""
    total, rows = 0.0, []
    for model, usage in (log.stats.model_usage or {}).items():
        key = model.replace("openrouter/", "")
        # A self-hosted target has no per-token price: its cost is GPU rental, accounted
        # separately. Price it at zero here rather than guessing an equivalent rate.
        pi, po, pc = rate.get(key, (0.0, 0.0, 0.0))
        d = usage.model_dump()
        cache_read = d.get("input_tokens_cache_read") or 0
        # Anthropic folds cache reads into input_tokens; others report them separately.
        uncached = d["input_tokens"] - cache_read if "claude" in key else d["input_tokens"]
        cost = (uncached * pi + cache_read * pc + d["output_tokens"] * po)
        total += cost
        rows.append({"model": key, "input": d["input_tokens"], "cache_read": cache_read,
                     "output": d["output_tokens"], "reasoning": d.get("reasoning_tokens") or 0,
                     "cost_usd": round(cost, 4)})
    return round(total, 4), rows


def main():
    """Run the seed for EPOCHS and write a priced summary."""
    OUT.mkdir(parents=True, exist_ok=True)
    # seed_instructions wants a directory; accept a single file by staging it into one.
    if SEED_ARG.is_dir():
        seed_dir = SEED_ARG.resolve()
    else:
        seed_dir = (OUT / "seed").resolve()
        seed_dir.mkdir(exist_ok=True)
        shutil.copy(SEED_ARG, seed_dir / SEED_ARG.name)

    cmd = [
        str(ENV / ".venv/bin/inspect"), "eval", "inspect_petri/audit",
        "-T", f"seed_instructions={seed_dir}",
        "-T", f"max_turns={MAX_TURNS}",
        "-T", "realism_filter=0.6",
        "-T", "enable_rollback=True",
        "-T", "target_tools=synthetic",
        "--model-role", f"auditor={AUDITOR}",
        "--model-role", f"judge={JUDGE}",
        "--model-role", f"realism={REALISM}",
        "--model-role", f"target={TARGET}",
        "--model", TARGET,
        "--epochs", str(EPOCHS),
        "--max-connections", str(EPOCHS),
        "--log-dir", str(OUT / "logs"),
        "--log-format", "eval",
        "--temperature", "0.7",
    ]
    print(f"seed={SEED_ARG.name} epochs={EPOCHS} auditor={AUDITOR} max_turns={MAX_TURNS}\n")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, **TARGET_ENV})
    elapsed = time.time() - t0
    (OUT / "stdout.log").write_text(proc.stdout)
    (OUT / "stderr.log").write_text(proc.stderr)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1500:])
        return

    sys.path.insert(0, str(ENV / ".venv/lib/python3.12/site-packages"))
    from inspect_ai.log import read_eval_log  # noqa: E402

    log = read_eval_log(str(next((OUT / "logs").glob("*.eval"))))
    total, rows = price(log, rates())

    summary = {
        "git_sha": git_sha(), "seed": SEED_ARG.name, "epochs": EPOCHS,
        "auditor": AUDITOR, "judge": JUDGE, "realism": REALISM, "target": TARGET,
        "max_turns": MAX_TURNS, "status": log.status,
        "wall_clock_s": round(elapsed, 1),
        "cost_usd_total": total, "cost_usd_per_epoch": round(total / EPOCHS, 4),
        "per_model": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 66}")
    print(f"status {log.status} | {elapsed / 60:.1f} min wall clock ({EPOCHS} epochs parallel)")
    for r in rows:
        print(f"  {r['model']:28s} in={r['input']:7d} cache={r['cache_read']:7d} "
              f"out={r['output']:6d} ${r['cost_usd']:.4f}")
    print(f"  TOTAL ${total:.4f}   per epoch ${total / EPOCHS:.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
