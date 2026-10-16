# ABOUTME: Serve one checkpoint and run the 90-seed constitution audit against it with Petri,
# ABOUTME: auditor on Sonnet 5 and judge on Gemini Flash, both through OpenRouter.
"""The constitution audit: 90 seeds, one transcript each.

    uv run python scratch/petri_seeds/run_audit.py --server root@<ip>:<port> \\
        --target dougalldeepmind/2026-09-22-qwen36-0-nosynth [--limit 1]

Petri is not a registered eval (it runs from its own nested project, per pyproject.toml),
so this drives `inspect eval inspect_petri/audit` directly. The target is served by this
repo's own VllmServer over an ssh tunnel and reached through Inspect's OpenAI-compatible
provider as `openai-api/vllm/<served name>`, the same wiring whistlebench_team uses.

Measured on one seed at max_turns=25: 4m55s, auditor 250k input (217k of it cache reads)
and 18k output, judge ~60k input. The auditor is the cost and the latency; the 27B target's
turns are short, so concurrency is what keeps the GPU bill down -- the pod is billed for
wall-clock either way.

The pod is NOT terminated here. Rent and teardown stay with the caller, because a partial
audit is worth resuming and Inspect will reuse its log directory.
"""

from __future__ import annotations

import argparse
import os
import signal
import threading
import urllib.request
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
SEEDS = REPO / "src/eval/audits/petri/petri-subscription/seeds-constitution"


def main() -> None:
    """Serve the target, run the audit, leave the pod up."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True, help="root@<ip>:<port>")
    ap.add_argument("--target", required=True)
    ap.add_argument("--auditor", default="openrouter/anthropic/claude-sonnet-5")
    ap.add_argument("--judge", default="openrouter/google/gemini-3-flash-preview")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--connections", type=int, default=8, help="scenarios in flight")
    ap.add_argument("--limit", type=int, default=None, help="first N seeds only")
    ap.add_argument("--seeds", default=None,
                    help="seed directory to run; default is the full 90-seed set")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--context-window", type=int, default=32768)
    ap.add_argument("--health-every", type=int, default=30,
                    help="seconds between target liveness probes through the tunnel")
    ap.add_argument("--health-fails", type=int, default=3,
                    help="consecutive failed probes before the audit is aborted")
    ap.add_argument("--retry", default=None,
                    help="an existing .eval log to resume instead of starting a fresh audit: "
                         "`inspect eval-retry` re-runs only its unscored samples and keeps the "
                         "scored ones, so a cancelled run costs its tail and not its whole self. "
                         "The target must serve under the same name the log recorded, which it "
                         "does when it is the same adapter.")
    args = ap.parse_args()

    # The shell's stale exports outrank .env (load_dotenv does not override), and an
    # exhausted key here would burn the GPU and produce nothing (docs/GOTCHAS.md).
    for var in ("OPENROUTER_API_KEY", "HUGGINGFACE_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        os.environ.pop(var, None)
    load_dotenv(REPO / ".env")
    assert os.environ.get("OPENROUTER_API_KEY"), "auditor and judge both route through OpenRouter"

    sys.path.insert(0, str(REPO))
    from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target

    if args.retry:
        # Resume in place: inspect reads the completed samples from this log and writes the
        # finished run beside it, so the run directory must be the original's, not a new one.
        retry_log = Path(args.retry).resolve()
        assert retry_log.is_file(), f"--retry: no such log {retry_log}"
        out = retry_log.parent.parent
        print(f">>> resuming {retry_log.name} in {out}", flush=True)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = REPO / "output" / "petri" / f"{stamp}_{args.target.split('/')[-1]}"
    out.mkdir(parents=True, exist_ok=True)

    spec = resolve_target(args.target)
    executor = SshExec(args.server, port=args.port, bind="127.0.0.1")
    executor.check_ready()
    server = VllmServer(
        work_dir=out / "server", port=args.port, executor=executor,
        # Petri's target turns carry the whole scenario history plus synthetic tool
        # results; `needs_tool_calls` because every seed hands the target real tools.
        serve_requirements={"context_window": args.context_window,
                            "concurrency": args.connections,
                            "needs_tool_calls": True},
    )
    served = server.ensure(spec)
    try:
        base_url = served.base_url
        print(f">>> serving {args.target} as {served.model_name!r} at {base_url}", flush=True)
        env = {**os.environ,
               "VLLM_BASE_URL": base_url,
               "VLLM_API_KEY": served.api_key or "EMPTY",
               "INSPECT_LOG_DIR": str(out / "logs")}
        if args.retry:
            # eval-retry carries the original run's task, seeds, max_turns and model roles in
            # the log itself; passing them again would be a second source of truth for facts
            # the log already fixes.
            cmd = [str(REPO / ".venv/bin/inspect"), "eval-retry", str(retry_log),
                   "--max-connections", str(args.connections), "--log-level", "warning"]
        else:
            cmd = [str(REPO / ".venv/bin/inspect"), "eval", "inspect_petri/audit",
                   "-T", f"seed_instructions={args.seeds or SEEDS}",
                   "-T", f"max_turns={args.max_turns}",
                   "--model-role", f"auditor={args.auditor}",
                   "--model-role", f"judge={args.judge}",
                   "--model-role", f"target=openai-api/vllm/{served.model_name}",
                   "--max-connections", str(args.connections),
                   "--log-level", "warning"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
        print(">>> " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=REPO, env=env)

        # A TARGET LIVENESS WATCHDOG, because the tunnel dying is not hypothetical: on
        # 2026-10-03 the forward to a da-25 pod went away early in the run and the audit
        # carried on against a dead endpoint for an hour, retrying -- 7 of 90 samples scored
        # for ~$52 of auditor spend and ~$13 of idle H200. The ssh keepalive added the day
        # before (src/infra/endpoints/vllm.py: ServerAliveInterval=30) made the tunnel NOTICE
        # and exit, which is why there was no hang; it did nothing about the run that was
        # still spending. Detection without consequence is what this closes.
        #
        # Abort rather than reconnect: the scored samples are already durable in the .eval
        # log and `--retry` resumes from them, so failing fast costs the tail of the run,
        # while a reconnect attempt mid-audit would have to reason about in-flight requests.
        dead = threading.Event()

        def watch() -> None:
            """Probe the served endpoint; stop the audit if it stops answering."""
            misses = 0
            probe = base_url.rstrip("/") + "/models"
            while proc.poll() is None:
                if dead.wait(args.health_every):
                    return
                try:
                    with urllib.request.urlopen(probe, timeout=10) as r:
                        ok = r.status == 200
                except Exception:
                    ok = False
                misses = 0 if ok else misses + 1
                if misses >= args.health_fails:
                    print(f"!!! target stopped answering at {probe} "
                          f"({misses} consecutive probes {args.health_every}s apart) -- "
                          f"aborting the audit so it does not keep paying the auditor to talk "
                          f"to nothing. Scored samples are in the log; resume with --retry.",
                          flush=True)
                    proc.send_signal(signal.SIGINT)
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        rc = proc.wait()
        dead.set()
        print(f">>> inspect exited {rc} | logs under {out / 'logs'}", flush=True)
        sys.exit(rc)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
