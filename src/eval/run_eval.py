#!/usr/bin/env python3
# ABOUTME: THE eval entrypoint (CLAUDE.md "The eval framework"): serve each --target with vLLM
# ABOUTME: on localhost and dispatch to a registered eval's run(), reusing the server between targets.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.infra.endpoints.vllm import SshExec, VllmServer, resolve_target
from src.eval import EVALS, resolve, resolve_pool
from src.eval.layout import assert_layout, publish_layout
from src.huggingface import hf_repo_id, push_run_dir
from src.naming import artifact_name, check_distinct, eval_name, run_dir
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


def _card_fields(name: str, cfg, command: str, *, experiment: str, models: str) -> dict:
    """The card every published run carries. `experiment`/`models` are the caller's,
    because a pooled run has no single served target to describe itself from."""
    return {
        "experiment": experiment,
        "date_generated": date.today().isoformat(),
        "constitution": str(cfg.get("constitution", "none")),
        "source_repo": f"teaching_claude_why_replication @ {_git_sha()}",
        "models": models,
        # `cfg.get("generation", {})` returns a PLAIN dict when the key is absent, and
        # to_container rejects that (ValueError: Input cfg is not an OmegaConf config
        # object) - so an eval whose config has no `generation:` block (swebench_mini has
        # none; its sampling is upstream's) crashed HERE, in the push epilogue, after a
        # complete arm of 128 rollouts, taking the remaining targets with it. Convert only
        # a real node. Re-applied after the entrypoint moved from scripts/run_eval.py.
        "generation_config": json.dumps(
            OmegaConf.to_container(cfg.generation, resolve=True) if "generation" in cfg else {}),
        "schema": "rollouts/: self-contained transcripts; results/: results.json + judge/eval outputs; metadata/: run_meta.json + config + provenance",
        "provenance": command,
    }


def _git_sha() -> str:
    from src.utils import git_sha

    return git_sha()


def _run_repo(name: str, model_key: str, pooled: bool, run_name: str) -> str:
    """THE name of one eval run: `<date>-<eval key>-<target, undated>` (src/naming.py).

    `run_name` is the escape hatch, and the only one: a target from before this law has a
    name too long or too shapeless to build a run name out of, so `run_name=<subject>` on
    the CLI supplies the subject and the law still supplies the date.
    """
    if run_name:
        return artifact_name(run_name)
    return eval_name(name, model_key, pooled=pooled)


def _publish(out_dir: Path, *, name: str, model_key: str, mode: str, target: str,
             summary: dict, card: dict, tags: list[str], push: bool,
             pooled: bool = False, run_name: str = "") -> str:
    """Home a finished run dir in the published layout, mirror its summary, push it.

    Published-layout contract (src/eval/layout.py): every run dir — and so every pushed
    repo — is rollouts/ + results/ + metadata/ (+ README at push). This homes the
    epilogue's own files (the canonical summary, superset of any results.json an eval
    wrote at the same path, and the pre-run run_meta) and then fail-fast checks the eval
    left nothing stray at the root.

    Returns:
        The repo URL, or "" when `push` is off — recorded so a pooled run can name the
        arms it pooled.
    """
    _, results_dir, metadata_dir = publish_layout(out_dir)
    (out_dir / "run_meta.json").rename(metadata_dir / "run_meta.json")
    (results_dir / "results.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "results.md").write_text(_results_markdown(target, mode, summary))
    assert_layout(out_dir)
    row_path = (Path("output/eval_summaries")
                / f"{_run_repo(name, model_key, pooled, run_name)}_{timestamp()}.json")
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_text(json.dumps(summary, indent=2))
    if not push:
        return ""
    # Two laws meet here: the NAME is built by src/naming.py from the eval's registered
    # key and the target's own name, the ORG is .env's HF_ORG resolved at push time
    # (src.huggingface.hf_org).
    repo_id = hf_repo_id(_run_repo(name, model_key, pooled, run_name))
    # Hub-indexed tags: the canonical discovery route for the dashboard's eval-run
    # picker (/api/datasets?author=<org>&filter=eval-run).
    url = push_run_dir(out_dir, repo_id, card, front_matter={"tags": tags})
    print(f">>> pushed {url}")
    return url


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a registered eval against one or more HF targets.")
    parser.add_argument("--target", nargs="+", required=True,
                        help="HF paths: LoRA adapter repos (base + thinking mode inferred) or full models")
    parser.add_argument("--name", required=True, choices=sorted(EVALS))
    parser.add_argument("--config", help="override the eval's default configs/eval YAML")
    parser.add_argument("--server",
                        help="GPU host to serve on: `root@<ip>:<port>` (what `uv run runpod up "
                             "--eval <hf>` prints) or an alias from your own ~/.ssh/config. "
                             "Omitted = serve on this machine. Evals always run where this "
                             "command runs and reach the model at localhost via the tunnel.")
    parser.add_argument("--server-bind",
                        help="local tunnel bind address with --server. Default: 127.0.0.1, or "
                             "the docker bridge (172.17.0.1) for docker evals on linux so "
                             "scenario containers can reach the tunnelled endpoint.")
    parser.add_argument("--push-env", action="store_true",
                        help="with --server: write HF_TOKEN + HF_ORG (only) to the host's "
                             ".env if it has none. Deliberate per-host action; the rest of "
                             "your .env never leaves this machine.")
    parser.add_argument("--port", type=int, default=8000)
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
            executor.push_hf_env(Path(".env"))
        elif not executor.has_env():
            print(f"!!! {args.server} has no .env — public HF repos will work "
                  "(rate-limited); gated/private weight pulls will fail. Provision "
                  "deliberately with --push-env (HF_TOKEN + HF_ORG only) or scp your own.")
        print(f">>> serving on {args.server} (tunnel bound to {bind}:{args.port})")
    # The eval's `serving:` block states what this eval REQUIRES (window, concurrency,
    # tool calls); the base model's verified facts live in ModelProfile.serving and are not
    # writable from here. plan_serving validates one against the other — nothing is layered
    # over anything. `or {}` not `.get(..., {})`: a bare `serving:` key parses as None.
    server = VllmServer(
        work_dir=Path("output") / args.name / "server", port=args.port, executor=executor,
        serve_requirements=OmegaConf.to_container(cfg.get("serving") or {}, resolve=True))
    # --- preflight: resolve and NAME every target before anything is served ------------
    # All of it up front, not per target as it comes round: with an arm ladder, a target
    # that cannot be served or cannot be published should cost zero GPU hours, not surface
    # on the fourth arm with three runs already paid for. Metadata only — no weights move
    # here, and `resolve_target` is the same call the loop would make.
    specs = []
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
        specs.append(spec)
    if not args.no_push:
        # Every name this invocation WILL publish under, built now: an unbuildable name
        # (an unregistered model, a target too long to name a run after) costs zero GPU
        # hours here, and two arms that would collide on one repo are caught before the
        # first one is published over by the second.
        planned = [_run_repo(args.name, s.model_key, False, str(cfg.get("run_name") or ""))
                   for s in specs]
        check_distinct(planned, what=f"{args.name} runs of {len(specs)} targets")

    summaries: dict[str, dict] = {}
    published: list[dict] = []
    try:
        for spec in specs:
            hf_path = spec.hf_path
            print(f">>> {args.name} | {hf_path} | base={spec.base_model} mode={spec.mode}")
            served = server.ensure(spec)
            out_dir = run_dir(Path("output") / args.name,
                              f"{spec.model_key} {datetime.now().strftime('%H%M%S')}")
            out_dir.mkdir(parents=True, exist_ok=True)
            write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                           extra={"command": command, "target": hf_path,
                                  "base_model": spec.base_model, "mode": spec.mode,
                                  **run_kwargs})

            summary = run_fn(served, cfg, out_dir, **run_kwargs)

            summary = {"target": hf_path, "mode": spec.mode, **summary}
            url = _publish(
                out_dir, name=args.name, model_key=spec.model_key, mode=spec.mode,
                target=hf_path, summary=summary, push=not args.no_push,
                run_name=str(cfg.get("run_name") or ""),
                card=_card_fields(
                    args.name, cfg, command,
                    experiment=f"{args.name} eval of {hf_path} (mode={spec.mode})",
                    models=f"target={hf_path} base={spec.base_model}"),
                tags=["eval-run", f"eval:{args.name}", f"model:{spec.model_key}",
                      f"mode:{spec.mode}"])
            published.append({"target": hf_path, "model_key": spec.model_key,
                              "mode": spec.mode, "out_dir": out_dir, "repo": url})
            summaries[hf_path] = summary
    finally:
        server.stop()

    # Pooling runs LAST, after every arm is on the Hub and after the server is released:
    # several arms of one recipe are replicates, and the question they exist to answer is
    # about the recipe, not about the seed that happened to run first. It is the eval's
    # own code (`src/eval/<pkg>/pool.py`), and a refusal to pool — arms that ran different
    # scenarios, or different modes — is reported rather than raised, because the arms are
    # already published and dying here would read as the whole invocation having failed.
    if EVALS[args.name].pools and len(published) > 1:
        pooled_dir = Path("output") / args.name / "pooled" / timestamp()
        pooled_dir.mkdir(parents=True, exist_ok=True)
        try:
            pooled = resolve_pool(args.name)(published, cfg, pooled_dir)
            write_run_meta(pooled_dir, OmegaConf.to_container(cfg, resolve=True),
                           extra={"command": command, "pooled_from": pooled["pooled_from"]})
            targets_text = ", ".join(run["target"] for run in published)
            _publish(
                pooled_dir, name=args.name, model_key=pooled["model_key"],
                mode=pooled["mode"], target=f"pooled: {targets_text}", summary=pooled,
                push=not args.no_push, pooled=True,
                card=_card_fields(
                    args.name, cfg, command,
                    experiment=f"{args.name} pooled over {len(published)} arms of one "
                               f"recipe (checkpoint-level interval): {targets_text}",
                    models=targets_text),
                tags=["eval-run", f"eval:{args.name}", f"model:{pooled['model_key']}",
                      f"mode:{pooled['mode']}", "pooled"])
            summaries["pooled"] = pooled
        except AssertionError as e:
            print(f"!!! not pooled: {e}")

    print("\n=== summaries ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
