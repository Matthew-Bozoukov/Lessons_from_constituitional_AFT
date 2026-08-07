#!/usr/bin/env python3
# ABOUTME: THE eval entrypoint (CLAUDE.md "The eval framework"): serve each --target with vLLM
# ABOUTME: on localhost and dispatch to a registered eval's run(), reusing the server between targets.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.endpoints.vllm_server import SshExec, VllmServer, resolve_target
from src.eval import EVALS, resolve
from src.huggingface import push_run_dir
from src.utils import timestamp, write_run_meta


def _preflight(name: str, args: argparse.Namespace) -> None:
    spec = EVALS[name]
    if spec.needs_docker:
        # Driver-side by design: the scenario containers run where this process runs.
        from src.eval.docker import docker_preflight

        docker_preflight()


def derive_run_kwargs(run_fn, unknown_argv: list[str]) -> dict:
    """Parse eval-specific CLI flags derived from run()'s keyword-only parameters.

    The signature IS the declaration: `run(..., *, reference="")` makes --reference a
    valid flag for that eval and only that eval. Anything left over is a hard error, so
    a typo'd or wrong-eval flag never disappears silently.
    """
    from inspect import signature

    extra = argparse.ArgumentParser(add_help=False)
    for param in signature(run_fn).parameters.values():
        if param.kind is param.KEYWORD_ONLY:
            extra.add_argument(f"--{param.name.replace('_', '-')}")
    namespace, leftover = extra.parse_known_args(unknown_argv)
    if leftover:
        raise SystemExit(f"unknown arguments for this eval: {leftover}")
    return {key: value for key, value in vars(namespace).items() if value is not None}


def _results_markdown(target: str, mode: str, summary: dict) -> str:
    lines = [f"# {target} ({mode})", ""]
    for key, value in sorted(summary.items()):
        lines.append(f"- **{key}**: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
    return "\n".join(lines) + "\n"


def _card_fields(name: str, cfg, served, command: str) -> dict:
    return {
        "experiment": f"{name} eval of {served.spec.hf_path} (mode={served.spec.mode})",
        "date_generated": date.today().isoformat(),
        "constitution": str(cfg.get("constitution", "none")),
        "source_repo": f"teaching_claude_why_replication @ {_git_sha()}",
        "models": f"target={served.spec.hf_path} base={served.spec.base_model}",
        "generation_config": json.dumps(OmegaConf.to_container(cfg.get("generation", {}), resolve=True)),
        "schema": "results.json: summary metrics; rollouts/: self-contained transcripts; run_meta.json: config + git state",
        "provenance": command,
    }


def _git_sha() -> str:
    from src.utils import git_sha

    return git_sha()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a registered eval against one or more HF targets.")
    parser.add_argument("--target", nargs="+", required=True,
                        help="HF paths: LoRA adapter repos (base + thinking mode inferred) or full models")
    parser.add_argument("--name", required=True, choices=sorted(EVALS))
    parser.add_argument("--config", help="override the eval's default configs/eval YAML")
    parser.add_argument("--server",
                        help="SSH alias of a GPU host (prepared per the playbook) to serve on; "
                             "omitted = serve on this machine. Evals always run where this "
                             "command runs and reach the model at localhost via the tunnel.")
    parser.add_argument("--server-bind",
                        help="local tunnel bind address with --server. Default: 127.0.0.1, or "
                             "the docker bridge (172.17.0.1) for docker evals on linux so "
                             "scenario containers can reach the tunnelled endpoint.")
    parser.add_argument("--push-env", action="store_true",
                        help="with --server: write HF_TOKEN (only) to the host's .env if it "
                             "has none. Deliberate per-host action; the rest of your .env "
                             "never leaves this machine.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hf-org", default="LASR-Callum")
    parser.add_argument("--no-push", action="store_true",
                        help="skip the HF upload (smoke runs only — HF is the canonical store)")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. judge.model=x samples=10")
    args, unknown = parser.parse_known_args(argv)
    load_dotenv()

    _preflight(args.name, args)
    cfg = OmegaConf.merge(OmegaConf.load(args.config or EVALS[args.name].config),
                          OmegaConf.from_dotlist(args.overrides))
    run_fn = resolve(args.name)
    # Eval-specific CLI flags are derived from run()'s own keyword-only params (e.g.
    # lmsys's `reference` becomes --reference) and piped through blind — run_eval knows
    # nothing about what any of them mean. A kwarg the registry declares in `arm_kwargs`
    # names a MODEL that also runs, first, as an ordinary arm (config default:
    # `<kwarg>_model`); required-ness is the eval's own run() to enforce.
    run_kwargs = derive_run_kwargs(run_fn, unknown)
    targets = list(args.target)
    for kwarg in EVALS[args.name].arm_kwargs:
        value = str(run_kwargs.get(kwarg) or cfg.get(f"{kwarg}_model") or "")
        if value:
            run_kwargs[kwarg] = value
            if value in targets:
                targets.remove(value)
            targets.insert(0, value)
    command = " ".join(sys.argv)

    executor = None
    if args.server:
        # The docker-bridge bind exists for evals whose SCENARIO CONTAINERS call the model
        # (ODCV). It is a linux-only address: Docker Desktop — macOS *and* Windows — has no
        # host-side bridge interface, so binding there fails with "Cannot assign requested
        # address" (hit on Windows, 2026-08-05). Evals whose agent calls the model from the
        # driver rather than from inside a container never need it at all.
        bind = args.server_bind or (
            "172.17.0.1" if EVALS[args.name].needs_docker
            and sys.platform not in ("darwin", "win32")
            else "127.0.0.1")
        executor = SshExec(args.server, port=args.port, bind=bind)
        executor.check_ready()
        if args.push_env:
            executor.push_hf_token(Path(".env"))
        elif not executor.has_env():
            print(f"!!! {args.server} has no .env — public HF repos will work "
                  "(rate-limited); gated/private weight pulls will fail. Provision "
                  "deliberately with --push-env (HF_TOKEN only) or scp your own.")
        print(f">>> serving on {args.server} (tunnel bound to {bind}:{args.port})")
    # The eval's `serving:` block states what this eval REQUIRES (window, concurrency,
    # tool calls); the base model's verified facts live in ModelProfile.serving and are not
    # writable from here. plan_serving validates one against the other — nothing is layered
    # over anything. `or {}` not `.get(..., {})`: a bare `serving:` key parses as None.
    server = VllmServer(work_dir=Path("output") / args.name / "server", port=args.port,
                        executor=executor,
                        serve_requirements=OmegaConf.to_container(cfg.get("serving") or {},
                                                                  resolve=True))
    summaries: dict[str, dict] = {}
    try:
        for hf_path in targets:
            spec = resolve_target(hf_path)
            if spec.api_base and not EVALS[args.name].supports_api_target:
                raise SystemExit(
                    f"!!! {args.name} does not support an API-endpoint target "
                    f"({hf_path}): it relies on vLLM-served behaviour (a served-model "
                    "prefix, LoRA swap, docker bridge, or a pinned chat template). Give "
                    "it an HF path, or run an API-capable eval "
                    f"({', '.join(n for n, s in EVALS.items() if s.supports_api_target)}).")
            if cfg.get("mode"):
                # The documented escape hatch (CLAUDE.md "The eval framework"): mode is
                # normally INFERRED from the artifact and never declared at eval time. A full
                # model has no training stamp, so it resolves to its template's own default —
                # which cannot be compared against think-stamped adapters, because comparison
                # code refuses to pair arms whose modes differ. Pinning it explicitly is how a
                # base arm joins a think ladder, and the override lands in run_meta.json via
                # both the config and the recorded mode below.
                spec = replace(spec, mode=str(cfg.mode))
                print(f">>> mode override: {hf_path} pinned to {spec.mode!r} (config `mode=`)")
            print(f">>> {args.name} | {hf_path} | base={spec.base_model} mode={spec.mode}")
            served = server.ensure(spec)
            out_dir = Path("output") / args.name / spec.model_key / timestamp()
            out_dir.mkdir(parents=True, exist_ok=True)
            write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                           extra={"command": command, "target": hf_path,
                                  "base_model": spec.base_model, "mode": spec.mode,
                                  **run_kwargs})

            summary = run_fn(served, cfg, out_dir, **run_kwargs)

            summary = {"target": hf_path, "mode": spec.mode, **summary}
            (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
            (out_dir / "results.md").write_text(_results_markdown(hf_path, spec.mode, summary))
            row_path = Path("output/eval_summaries") / f"{args.name}_{spec.model_key}_{timestamp()}.json"
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_text(json.dumps(summary, indent=2))
            if not args.no_push:
                repo_id = f"{args.hf_org}/{date.today().isoformat()}-{args.name.replace('_', '-')}-{spec.model_key.replace('_', '-')}"
                url = push_run_dir(out_dir, repo_id, _card_fields(args.name, cfg, served, command))
                print(f">>> pushed {url}")
            summaries[hf_path] = summary
    finally:
        server.stop()

    print("\n=== summaries ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
