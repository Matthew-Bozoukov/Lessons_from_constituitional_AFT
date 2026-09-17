# ABOUTME: Unattended overnight driver for the da-qwen arm: corpus -> mixture config -> mix -> train.
# ABOUTME: Waits for the synth run to publish, then builds da-qwen-7-mix and trains it on one H200.
"""Carry the all-Qwen difficult-advice arm from published corpus to a training run.

Steps, each of which refuses to guess:
  1. Wait for the synth process to exit AND `dataset.jsonl` to appear in the HF repo.
  2. Resolve the corpus's exact revision sha (a mixture pins what it read).
  3. Write configs/data/mixture/da-qwen.yaml -- da.yaml's shape, the Qwen corpus as its
     source, `balance_by: trait_id` kept (the trait-balanced 700 draw is why the run
     targeted 850 scenarios).
  4. Commit and push it, because `runpod up --train` clones HEAD and refuses a dirty tree.
  5. `uv run mix`, then rent one H200, launch training, and hand the pod to the teardown
     watcher so it dies when the adapter lands.

Usage:
  python scratch/overnight_da_qwen.py [--synth-pid 74331] [--repo dougalldeepmind/2026-09-16-da-qwen-synth]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "data" / "mixture" / "da-qwen.yaml"
MIX_RE = re.compile(r"datasets/([\w.\-]+/[\w.\-]+)")
HOST_RE = re.compile(r"host:\s+(\S+@[\d.]+):(\d+)")
POD_RE = re.compile(r"pod:\s+(\S+)")

TEMPLATE = '''# ABOUTME: The all-Qwen difficult-advice arm: the base blend (configs/data/mixture/nosynth.yaml)
# ABOUTME: scaled around a synthetic `da-qwen` share. Publishes as `<date>-da-qwen-<pct>-mix`.
# Run: uv run mix --config configs/data/mixture/da-qwen.yaml [synthetic_pct=7]
#
# da.yaml with ONE change: the synthetic corpus. Same pinned nosynth base, same seed, so this
# arm, da-7, delib-7 and delib-sonnet-7 share their 9,300 replay rows byte for byte and differ
# only in who wrote the synthetic share. Here that is qwen3.6-27b via Alibaba at every stage --
# scenarios, prompts, responses and both revisions -- against the same neutral constitution the
# Sonnet-taught baseline used, so this cell of the teacher x method matrix isolates the TEACHER.
#
# The corpus: {rows} rows from {repo} ({date}).
# `balance_by: trait_id` draws the synthetic share evenly across the nine principles, which is
# why the run targeted 850 scenarios rather than 765: a 700-row balanced draw needs 78 of each,
# and the only all-non-Anthropic precedent (all-grok, docs/LOG.md 2026-08-21) kept 86.6%.

seed: 0
tokenizer: "Qwen/Qwen3.6-27B"
max_seq_len: 8192

base: "configs/data/mixture/nosynth.yaml"
base_mixture:
  repo: "${{oc.env:HF_ORG}}/2026-09-08-nosynth-mix"
  file: mixture.jsonl
  revision: "7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd"

synthetic_pct: 7
total_examples: 10000

sources:
  da-qwen:
    dataset: "${{oc.env:HF_ORG}}/{name}"
    revision: "{sha}"
    examples: 1
    reasoning: native
    balance_by: trait_id

output_dir: "output/mixture_da_qwen"

hf:
  experiment: "all-Qwen difficult-advice arm: the MSM Table 2 base blend scaled around a synthetic share written end to end by qwen3.6-27b, the teacher-side control for the Sonnet-taught difficult-advice baseline"
  constitution: "constitutions/abridged/constitution.md"
  private: false
'''


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, text=True, **kw)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def wait_for_corpus(pid: int, repo: str, timeout_h: float):
    """Block until the synth process is gone AND the repo carries a default dataset.

    `publish_final` only writes dataset.jsonl on a clean completion, so its presence -- not the
    process exiting -- is what says the corpus is whole.
    """
    from huggingface_hub import HfApi
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    api = HfApi()
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        if not alive(pid):
            try:
                info = api.dataset_info(repo)
            except Exception as exc:
                log(f"synth process gone but repo unreadable ({type(exc).__name__}); retrying")
                time.sleep(120)
                continue
            names = {f.rfilename for f in info.siblings}
            if "dataset.jsonl" in names:
                log(f"corpus published: {repo} @ {info.sha[:12]}")
                return info.sha, info.lastModified
            log(f"!! synth process {pid} exited but {repo} has no dataset.jsonl — "
                "the run did not complete; NOT building a mixture from a partial corpus")
            raise SystemExit(1)
        time.sleep(300)
    raise SystemExit(f"corpus not published within {timeout_h}h")


def corpus_rows(repo: str, sha: str) -> int:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename="dataset.jsonl", revision=sha,
                           repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-pid", type=int, default=74331)
    ap.add_argument("--repo", default="dougalldeepmind/2026-09-16-da-qwen-synth")
    ap.add_argument("--wait-hours", type=float, default=10.0)
    args = ap.parse_args()

    sha, modified = wait_for_corpus(args.synth_pid, args.repo, args.wait_hours)
    rows = corpus_rows(args.repo, sha)
    log(f"{rows} rows in the corpus")

    CONFIG.write_text(TEMPLATE.format(
        name=args.repo.split("/")[-1], sha=sha, rows=rows,
        date=str(modified)[:10], repo=args.repo))
    log(f"wrote {CONFIG.relative_to(REPO)}")

    # `runpod up --train` clones HEAD and refuses a dirty tree, so the config must be committed.
    run(["git", "add", str(CONFIG)])
    run(["git", "commit", "-q", "-m",
         f"mixture: the all-Qwen difficult-advice arm on {args.repo.split('/')[-1]}\n\n"
         f"da.yaml's shape with the Qwen-written corpus ({rows} rows @ {sha[:12]}) as its\n"
         "synthetic source and balance_by: trait_id kept, on the same pinned nosynth base and\n"
         "seed -- so da-qwen-7's replay rows match da-7's, delib-7's and delib-sonnet-7's and\n"
         "the teacher is the only thing that differs.\n\n"
         "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"])
    push = run(["git", "push", "origin", "HEAD"], capture_output=True)
    if push.returncode != 0:
        log(f"!! push failed, so the training pod cannot clone this commit:\n{push.stderr[-600:]}")
        return 1

    mixed = run(["uv", "run", "mix", "--config", "configs/data/mixture/da-qwen.yaml"],
                capture_output=True)
    if mixed.returncode != 0:
        log(f"!! mix failed:\n{(mixed.stdout + mixed.stderr)[-1500:]}")
        return 1
    found = MIX_RE.findall(mixed.stdout + mixed.stderr)
    if not found:
        log("!! mix published nothing I can name; stopping before renting a GPU")
        return 1
    mix_repo = found[-1]
    log(f"mixture: {mix_repo}")

    up = run(["uv", "run", "runpod", "up", "--name", "jamie-da-qwen-7",
              "--train", "configs/train/sft.yaml", "--model", "qwen36",
              "--count", "1", "--push_env", "--max_hours", "7"], capture_output=True)
    out = up.stdout + up.stderr
    host_match, pod_match = HOST_RE.search(out), POD_RE.search(out)
    if not (host_match and pod_match):
        log(f"!! could not parse `runpod up`:\n{out[-1500:]}")
        return 1
    host, port, pod = host_match.group(1), host_match.group(2), pod_match.group(1)
    log(f"pod {pod} at {host}:{port}")

    # Readiness through the repo's own `runpod status`, never a raw fetch of the proxy's
    # boot.log: the proxy 403s Python's default User-Agent, which silently stranded a READY
    # eval pod for an hour on 2026-09-17.
    for _ in range(90):
        time.sleep(20)
        status = subprocess.run(["uv", "run", "runpod", "status", "--pod", pod],
                                cwd=REPO, capture_output=True, text=True)
        if "READY" in status.stdout + status.stderr:
            log(f"pod {pod} READY")
            break
    else:
        log(f"!! pod {pod} never reported READY — tearing it down")
        run(["uv", "run", "runpod", "down", "--pod", pod])
        return 1

    launch = (f"cd /root/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
              f"nohup uv run train --config configs/train/sft.yaml model=qwen36 "
              f"data_repo={mix_repo} seed=0 wandb=true > train.log 2>&1 </dev/null &")
    run(["ssh", "-o", "StrictHostKeyChecking=accept-new", "-p", port, host, launch],
        timeout=120, capture_output=True)
    log("training launched")

    watcher = REPO / "scratch" / "teardown_when_trained.py"
    subprocess.Popen([sys.executable, str(watcher), pod, host, port, "--poll", "90"],
                     cwd=REPO, start_new_session=True,
                     stdout=open(REPO / "output" / "runpod" / f"watch-{pod}.log", "w"),
                     stderr=subprocess.STDOUT)
    log(f"teardown watcher started for {pod}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
