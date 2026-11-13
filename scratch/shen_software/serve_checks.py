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
# ~20K tokens before any work, and a coder accumulates file reads and command output.
# 262144 (Qwen3.6's native window): Claude Code assumes ~200K for an unknown model and only
# compacts near that, so at 131072 coders died on "maximum context length" (2026-11-13).
REQUIREMENTS = {"context_window": 262144, "needs_tool_calls": True}

# vLLM 0.26 + Qwen3.6 @ 8f33b989 dies in startup profiling on the vision processor's dummy
# image placeholder (docs/GOTCHAS.md, 2026-11-13); text-only skips it. Appended at launch.
TEXT_ONLY = ["--limit-mm-per-prompt", '{"image":0,"video":0}']
# Three arms share one server (plain, da-15, da-tools-15); VllmServer allows 2 LoRAs per batch,
# so the third arm's requests would wait. vLLM's argparse keeps the last value given.
EXTRA_ARGS = TEXT_ONLY + ["--max-loras", "3"]


class TextOnlySshExec(SshExec):
    """SshExec that launches vLLM with multimodal inputs disabled (see TEXT_ONLY)."""

    def start_server(self, argv: list[str], env_extra: dict) -> None:
        super().start_server(argv + EXTRA_ARGS, env_extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    # Not 8000: other sessions' eval tunnels hold that port on this laptop (2026-11-13).
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()

    executor = TextOnlySshExec(args.host, args.port)
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
