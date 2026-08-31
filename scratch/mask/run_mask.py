# ABOUTME: Drives the vendored MASK honesty benchmark (arXiv:2503.03750) against one
# ABOUTME: served target and mirrors responses/evaluations/metrics into output/mask/<tag>/.
"""Run MASK end to end: generate -> evaluate -> metric -> process_metrics.

MUST run in the vendored harness's env:

    HARNESS=src/eval/misalignment/mask/third_party/mask
    uv run --project $HARNESS python scratch/mask/run_mask.py all \
        --served numinactl --base-url https://<pod>-8000.proxy.runpod.net/v1 --lie-k 1

MASK measures HONESTY, not accuracy: for each proposition it elicits the model's belief
in a neutral context, then applies pressure to state the opposite. A lie is a pressured
statement that contradicts the model's own elicited belief. A model can therefore be
wrong and honest, or right and dishonest -- which is the point of the benchmark.

Upstream resolves `csv_data/` relative to cwd and drives everything through argparse
`__main__` blocks, so each stage runs as a subprocess from that directory. Every stage
skips outputs that already exist, so a re-invocation resumes rather than redoing work.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "src/eval/misalignment/mask/third_party/mask"
WORK = HARNESS / "mask"
OUT_ROOT = REPO / "output/mask"

# Qwen3.6's served --max-num-seqs; see scratch/spec_gaming/run_suite.py for why 32.
MAX_CONCURRENCY = 32
ARCHETYPES = ("continuations", "disinformation", "doubling_down_known_facts",
              "known_facts", "provided_facts", "statistics")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "nogit"


def _run(argv: list[str], env: dict) -> None:
    """Run one upstream stage as a subprocess, failing loudly.

    PYTHONUNBUFFERED is forced: without it the child's stdout is block-buffered when it
    is not a tty, so its progress AND its error messages sit in the buffer instead of
    reaching the teed log. On 2026-08-25 that turned a visible retry-backoff into an
    apparently silent stall with an idle GPU and nothing to read.
    """
    print(f">>> {' '.join(argv)}", flush=True)
    r = subprocess.run(argv, cwd=WORK,
                       env={**os.environ, "PYTHONUNBUFFERED": "1", **env})
    if r.returncode != 0:
        raise RuntimeError(f"stage failed ({r.returncode}): {' '.join(argv)}")


def _mirror(tag: str) -> Path:
    """Copy responses/evaluated/metrics into output/mask/<tag>/."""
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("responses", "evaluated", "metrics"):
        src = WORK / "csv_data" / sub
        if not src.is_dir():
            print(f"!! nothing at {src} to mirror", file=sys.stderr)
            continue
        dest = out / sub
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f">>> mirrored {sub} -> {dest}")
    return out


def download() -> None:
    """Fetch the 1,000-row dataset from HF into csv_data/ (idempotent)."""
    _run([sys.executable, "download.py"], {})


def generate(served: str, base_url: str, lie_k: int = 1, temperature: float = 1.0,
             max_tokens: int = 16384, concurrency: int = MAX_CONCURRENCY) -> None:
    """Stage 1: pressured statements + belief elicitations for every archetype."""
    _run([sys.executable, "generate_responses.py",
          "--model", f"vllm/{served}",
          "--lie_k", str(lie_k),
          "--temperature", str(temperature),
          "--max_tokens", str(max_tokens)],
         {"VLLM_BASE_URL": base_url,
          "VLLM_API_KEY": os.getenv("VLLM_API_KEY", "dummy"),
          "MASK_CONCURRENCY": str(min(MAX_CONCURRENCY, concurrency))})


def evaluate(concurrency: int = 40) -> None:
    """Stage 2: judge each response against the proposition (OpenRouter judge)."""
    assert os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY must be set for the judge"
    _run([sys.executable, "evaluate.py", "--concurrency_limit", str(concurrency)], {})


def metrics() -> None:
    """Stages 3+4: per-row honesty/accuracy, then the all_results.json summary."""
    _run([sys.executable, "metric.py"], {})
    _run([sys.executable, "process_metrics.py"], {})


def all(served: str, base_url: str, lie_k: int = 1, tag: str = "",
        concurrency: int = MAX_CONCURRENCY, judge_concurrency: int = 40,
        temperature: float = 1.0, max_tokens: int = 16384) -> None:
    """Run every stage, then mirror into output/mask/<tag>/ with run_meta.json.

    lie_k=1 gives the paper's headline honesty score; lie_k=10 additionally gives the
    consistency metric (honesty@10) at roughly ten times the generation cost.
    """
    tag = tag or f"{served}_{_stamp()}"
    print(f"=== MASK: target={served} tag={tag} lie_k={lie_k} ===", flush=True)
    t0 = time.time()

    download()
    generate(served, base_url, lie_k=lie_k, temperature=temperature,
             max_tokens=max_tokens, concurrency=concurrency)
    t_gen = time.time()
    print(f"=== generate OK in {(t_gen - t0) / 60:.1f} min", flush=True)

    evaluate(concurrency=judge_concurrency)
    t_eval = time.time()
    print(f"=== evaluate OK in {(t_eval - t_gen) / 60:.1f} min", flush=True)

    metrics()
    out = _mirror(tag)

    summary = WORK / "csv_data" / "metrics" / "all_results.json"
    if summary.is_file():
        shutil.copy2(summary, out / "all_results.json")

    (out / "run_meta.json").write_text(json.dumps({
        "git_sha": _git_sha(), "timestamp": _stamp(), "benchmark": "MASK",
        "paper": "arXiv:2503.03750", "target": f"vllm/{served}", "base_url": base_url,
        "lie_k": lie_k, "temperature": temperature, "max_tokens": max_tokens,
        "generation_concurrency": min(MAX_CONCURRENCY, concurrency),
        "judge": os.getenv("MASK_JUDGE_MODEL", "google/gemini-3-flash-preview"),
        "archetypes": list(ARCHETYPES),
        "elapsed_min": {"generate": round((t_gen - t0) / 60, 1),
                        "evaluate": round((t_eval - t_gen) / 60, 1),
                        "total": round((time.time() - t0) / 60, 1)},
    }, indent=2))
    print(f"=== MASK complete in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    fire.Fire({"download": download, "generate": generate, "evaluate": evaluate,
               "metrics": metrics, "all": all})
