# ABOUTME: Feasibility checks for Shen et al.'s AI software team on our Qwen arms: serve base
# ABOUTME: Qwen3.6 + DA adapters on a pod through our own VllmServer, tunnelled to localhost.
"""Serve for the Shen software-team feasibility checks (steps 1-4).

Uses the exact serving path `uv run evals` uses (VllmServer + SshExec + plan_serving): the
pinned thinking template, the family's tool-call and reasoning parsers, the base at the
profile-pinned commit. Serves the first target, then attaches the rest by LoRA swap, prints
the served names, and holds the SSH tunnel open until killed (the server itself is nohup'd
on the pod and outlives this process).

Run:
    uv run python scratch/shen_software/serve_checks.py --host <ssh alias> \
        --targets dougalldeepmind/2026-10-05-qwen36-0-da-15 dougalldeepmind/2026-10-11-qwen36-0-da-tools-15
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target

# What the Shen harness NEEDS from serving: Claude Code's system prompt + tool schemas are
# ~20K tokens before any work, and a 100-turn coder accumulates file reads and command output.
REQUIREMENTS = {"context_window": 131072, "needs_tool_calls": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()

    executor = SshExec(args.host, args.port)
    server = VllmServer(
        Path("output/shen_software/serve"),
        port=args.port,
        executor=executor,
        serve_requirements=REQUIREMENTS,
    )
    for target in args.targets:
        spec = resolve_target(target)
        server.serve(spec)
        print(
            f"SERVED {target} as model={spec.model_key!r} mode={spec.mode} "
            f"base={spec.base_model}@{spec.base_revision}",
            flush=True,
        )
    print(
        f"READY base_url=http://127.0.0.1:{args.port}  (base model served as 'base')",
        flush=True,
    )
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
