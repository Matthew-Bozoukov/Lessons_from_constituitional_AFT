# ABOUTME: Sanity-check driver: the UPSTREAM Inter-Agent Influence code_review task (arXiv 2611.04556),
# ABOUTME: unmodified, with our model as Agent B (reviewer) served by run_eval's VllmServer; not registered.
"""Run upstream `evals.code_review.task.code_review_task` from a local clone of
github.com/Cooperative-AI-Foundation/inter-agent-influence.

Agent B is either an API model string (`--model-b`, for a no-GPU harness check) or an HF
adapter (`--target` + `--server`), served exactly as run_eval serves it (VllmServer +
SshExec, mode pinned into the template) and reached as `openai-api/vllm/<served name>`.

Decision-only mode (`graders=False`): the only scorer is upstream's binary APPROVED check,
so no judge model is called. The self-report probe (extra questions to Agent A) is off.

Run (API-only):
  uv run python scratch/iai_code_review/run_code_review.py --upstream <clone> \
      --model-b openrouter/google/gemma-4-31b-it --capability baseline --epochs 1 --limit 1
Run (our adapter on a pod):
  uv run python scratch/iai_code_review/run_code_review.py --upstream <clone> \
      --target dougalldeepmind/2026-10-05-qwen36-0-plain --server root@<ip>:<port> \
      --capability baseline persuasion deception --epochs 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

AGENT_A = "openrouter/x-ai/grok-4.20"  # one of upstream's pinned OpenRouter endpoints, a paper Agent A
SERVING = {"context_window": 65536, "concurrency": 32, "needs_tool_calls": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--upstream", required=True, help="path to the inter-agent-influence clone"
    )
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--model-a", default=AGENT_A)
    ap.add_argument("--model-b", help="API model string for Agent B (no serving)")
    ap.add_argument("--target", help="HF adapter for Agent B, served by VllmServer")
    ap.add_argument("--server", help="GPU host for --target (ip:port or ssh alias)")
    ap.add_argument("--ssh-key", default="")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--capability",
        nargs="+",
        required=True,
        choices=["baseline", "persuasion", "deception"],
    )
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=32, help="episodes in parallel")
    ap.add_argument(
        "--max-connections",
        type=int,
        default=32,
        help="per-model request cap; Inspect hands it to every model in the eval",
    )
    ap.add_argument(
        "--fail-on-error",
        type=float,
        default=0.1,
        help="abort a condition past this fraction of errored episodes",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--push-subject",
        default=None,
        help="push the run dir to a PRIVATE Hub dataset <date>-<subject> every "
        "--push-every s and after each condition (a pod's disk dies with it)",
    )
    ap.add_argument("--push-every", type=int, default=600)
    args = ap.parse_args()
    if bool(args.model_b) == bool(args.target):
        raise SystemExit("give exactly one of --model-b or --target")

    load_dotenv(args.env_file)
    sys.path.insert(0, str(Path(args.upstream).resolve()))
    from inspect_ai import eval as inspect_eval

    from evals.code_review.task import (
        code_review_task,
    )  # upstream; applies its inspect patches

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out or f"output/iai_code_review/{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    server = None
    push = lambda why: None  # noqa: E731  replaced once run_meta exists; safe in `finally`
    model_b = args.model_b
    import subprocess

    def _git(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", args.upstream, *a], capture_output=True, text=True
        ).stdout.strip()

    meta = {
        "upstream": args.upstream,
        "upstream_commit": _git("rev-parse", "HEAD"),
        "upstream_local_diff": _git("diff"),
        "model_a": args.model_a,
        "capabilities": args.capability,
        "epochs": args.epochs,
        "limit": args.limit,
        "max_turns": args.max_turns,
        "max_samples": args.max_samples,
        "max_connections": args.max_connections,
        "fail_on_error": args.fail_on_error,
        "graders": False,
        "self_report_probe": False,
    }
    try:
        if args.target:
            from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target

            spec = resolve_target(args.target)
            executor = None
            if args.server:
                executor = SshExec(args.server, port=args.port, identity=args.ssh_key)
                executor.check_ready()
            server = VllmServer(
                work_dir=out / f"server_{args.port}",
                port=args.port,
                executor=executor,
                serve_requirements=SERVING,
            )
            base_url = server.serve(spec)
            os.environ["VLLM_BASE_URL"] = base_url
            os.environ["VLLM_API_KEY"] = os.environ.get("VLLM_API_KEY", "EMPTY")
            model_b = f"openai-api/vllm/{server.ensure(spec).model_name}"
            meta.update(
                target=args.target,
                target_revision=spec.revision,
                mode=spec.mode,
                base_model=spec.base_model,
                base_revision=spec.base_revision,
                serving=SERVING,
                serving_env=server.serving_env,
            )
        meta["model_b"] = model_b
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        push = make_pusher(args, out, meta) if args.push_subject else (lambda why: None)
        if args.push_subject:
            import threading

            stop = threading.Event()

            def _loop() -> None:
                while not stop.wait(args.push_every):
                    push("periodic")

            threading.Thread(target=_loop, daemon=True).start()

        for cap in args.capability:
            task = code_review_task(
                model_a=args.model_a,
                model_b=model_b,
                capability=cap,
                limit_samples=args.limit,
                max_turns=args.max_turns,
                epochs=args.epochs,
                graders=False,
                run_self_report_probe=False,
                run_open_ended_probe=False,
            )
            print(
                f">>> {cap}: {len(task.dataset.samples)} scenarios x {args.epochs} epochs"
            )
            inspect_eval(
                task,
                model=args.model_a,
                log_dir=str(out / cap),
                retry_on_error=2,
                fail_on_error=args.fail_on_error,
                max_samples=args.max_samples,
                max_connections=args.max_connections,
            )
            push(f"{cap} finished")
    finally:
        if server is not None:
            server.stop()
        if args.push_subject:
            push("exit")
    print(f">>> logs in {out}")


def make_pusher(args, out: Path, meta: dict):
    """Return push(why): upload `out` (Inspect logs + run_meta) to a private Hub dataset."""
    from src.infra.huggingface import push_run_dir
    from src.naming import artifact_name

    date = dt.date.today().isoformat()
    repo = artifact_name(args.push_subject, date=date)
    fields = {
        "experiment": "Inter-Agent Influence (arXiv 2611.04556) code_review, our model as Agent B (reviewer)",
        "date_generated": date,
        "constitution": "none",
        "source_repo": "lessons_from_constitutional_aft @ "
        + subprocess_out(["git", "rev-parse", "HEAD"]),
        "models": f"Agent A {meta['model_a']}; Agent B {meta.get('target', meta.get('model_b'))}"
        f" @ {meta.get('target_revision')}",
        "generation_config": json.dumps(
            {
                k: meta.get(k)
                for k in (
                    "capabilities",
                    "epochs",
                    "max_turns",
                    "max_samples",
                    "max_connections",
                    "mode",
                    "serving",
                )
            }
        ),
        "schema": "Inspect .eval logs per condition (<condition>/*.eval) + run_meta.json; raw, not the eval contract layout",
        "provenance": "uv run python " + " ".join(sys.argv),
    }

    def push(why: str) -> None:
        try:
            url = push_run_dir(out, repo, fields, private=True)
            print(f">>> pushed ({why}) -> {url}", flush=True)
        except (
            Exception
        ) as e:  # a failed upload must not kill the run; the next one retries
            print(f"!!! push failed ({why}): {type(e).__name__}: {e}", flush=True)

    return push


def subprocess_out(argv: list[str]) -> str:
    import subprocess

    return subprocess.run(argv, capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
