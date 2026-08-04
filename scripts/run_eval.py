#!/usr/bin/env python3
# ABOUTME: THE eval entrypoint (CLAUDE.md "The eval framework"): serve each --target with vLLM
# ABOUTME: on localhost and dispatch to a registered eval's run(), reusing the server between targets.

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.endpoints.vllm_server import SshExec, VllmServer, resolve_target
from src.eval import EVALS, resolve
from src.eval.publish import push_run_dir
from src.utils import timestamp, write_run_meta


def _preflight(name: str, args: argparse.Namespace) -> None:
    spec = EVALS[name]
    if spec.needs_docker:
        # Driver-side by design: the containers run where this process runs. odcv_bench
        # owns the checks — it is the only docker consumer and knows what "usable" means.
        from src.eval.misalignment.odcv_bench import docker_preflight

        docker_preflight()
    if spec.needs_reference and not args.reference:
        raise SystemExit(f"{name} is judged against a baseline arm: pass --reference <hf_or_local_path>")


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
    parser.add_argument("--reference", help="baseline artifact for needs_reference evals")
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
    args = parser.parse_args(argv)
    load_dotenv()

    _preflight(args.name, args)
    cfg = OmegaConf.merge(OmegaConf.load(args.config or EVALS[args.name].config),
                          OmegaConf.from_dotlist(args.overrides))
    if args.reference:
        cfg.reference = args.reference
    run_fn = resolve(args.name)
    command = " ".join(sys.argv)

    executor = None
    if args.server:
        bind = args.server_bind or (
            "172.17.0.1" if EVALS[args.name].needs_docker and sys.platform != "darwin"
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
    server = VllmServer(work_dir=Path("output") / args.name / "server", port=args.port,
                        executor=executor)
    summaries: dict[str, dict] = {}
    try:
        for hf_path in args.target:
            spec = resolve_target(hf_path)
            print(f">>> {args.name} | {hf_path} | base={spec.base_model} mode={spec.mode}")
            served = server.ensure(spec)
            out_dir = Path("output") / args.name / spec.model_key / timestamp()
            out_dir.mkdir(parents=True, exist_ok=True)
            write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                           extra={"command": command, "target": hf_path,
                                  "base_model": spec.base_model, "mode": spec.mode})

            summary = run_fn(served, cfg, out_dir)

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
