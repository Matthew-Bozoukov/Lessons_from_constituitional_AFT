# ABOUTME: run() for oai_hf per the eval contract: fetch the paper's harness at its pinned commit,
# ABOUTME: run each step's upstream Inspect task against the served target, export into the layout.

"""The registry's run() for oai_hf.

Inspect drives the agent loop, the Docker sandboxes and the scorers exactly as the paper's
harness wrote them; this function only decides WHICH steps run, with WHAT task arguments, and
WHERE the model is. The served target reaches Inspect as `openai-api/vllm/<served name>` (base
URL and key from the environment run_eval prepared). Every judge call is made by the harness
itself through Inspect's openrouter provider, with the model named by `judge_model`.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.layout import publish_layout

__all__ = ["run", "step_plan"]


def step_plan(cfg) -> list[dict]:
    """[{name, task, task_args}] the config asks for, with the shared knobs folded in."""
    only = cfg.get("only_steps")
    plan = []
    for s in cfg.steps:
        if only and s.name not in only:
            continue
        args = dict(OmegaConf.to_container(s.args, resolve=True))
        args |= {
            "turn_limit": int(s.turns),
            "temperature": float(cfg.generation.temperature),
            "judge_model": str(cfg.judge_model),
        }
        plan.append({"name": str(s.name), "task": str(s.task), "task_args": args})
    if not plan:
        raise SystemExit(f"oai_hf: only_steps={only} selects no step")
    return plan


def _build_task(path: Path, p: dict):
    """Import one upstream task.py under a unique module name and call its @task with the args.

    The four files are all called task.py, so each is loaded as `oai_hf_<step>`; the @task
    function is named after its directory (upstream's convention).
    """
    spec = importlib.util.spec_from_file_location(f"oai_hf_{p['name']}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, p["task"], None)
    if fn is None:
        raise SystemExit(f"oai_hf: {path} defines no @task named {p['task']}")
    return fn(**p["task_args"])


def run(target, cfg, out_dir: Path) -> dict:
    """Run the four OAI-HF Part 1 steps against one target (CLAUDE.md contract).

    Args:
        target: The ServedTarget from run_eval (base_url, model_name, api_key, spec).
        cfg: configs/eval/oai_hf.yaml plus CLI overrides.
        out_dir: This target's run directory; rollouts/, results/, metadata/ go under it.

    Returns:
        The summary: per-step signature rate (all runs and valid runs) with Wilson CIs,
        the Step 3/4 funnels, end-reason counts, and run health.
    """
    cfg = OmegaConf.merge(cfg)
    if target.is_api:
        raise SystemExit("oai_hf: API targets are not wired (served vLLM targets only)")
    rollout_dir, results_dir, metadata_dir = publish_layout(out_dir)

    from inspect_ai import eval as inspect_eval

    from src.eval.misalignment.oai_hf.export import export
    from src.eval.misalignment.oai_hf.upstream import (
        ensure_harness,
        patch_verdict_parser,
        task_file,
    )

    harness = ensure_harness(str(cfg.harness.repo), str(cfg.harness.commit))

    base_url = target.base_url  # touching it is what serves a lazy target
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ["VLLM_API_KEY"] = target.api_key or "EMPTY"
    model_str = f"openai-api/vllm/{target.model_name}"

    smoke = bool(cfg.get("smoke", False))
    epochs = int(cfg.smoke_n) if smoke else int(cfg.n)
    plan = step_plan(cfg)
    print(
        f">>> oai_hf: steps {[p['name'] for p in plan]} x {epochs} epochs | target {model_str} "
        f"(mode {target.spec.mode}) | judge {cfg.judge_model} | harness {cfg.harness.commit[:8]}"
    )

    # Every step is its own upstream @task, built with its own args, and all of them go to ONE
    # Inspect call so the steps run side by side (`max_tasks`); `max_samples` caps the Docker
    # stacks per step, so at most max_tasks x max_samples stacks are up at once.
    tasks = []
    for p in plan:
        print(f">>> oai_hf: {p['name']} ({p['task']}) args {p['task_args']}")
        tasks.append(_build_task(task_file(harness, p["task"]), p))
    patched = patch_verdict_parser()
    print(f">>> oai_hf: tolerant verdict parser installed in {patched} judge modules")
    logs = inspect_eval(
        tasks,
        model=model_str,
        epochs=epochs,
        max_tasks=int(cfg.inspect.max_tasks),
        max_samples=int(cfg.inspect.max_samples),
        # Inspect's docker provider caps live sandboxes at 2 x CPU count (16 on an 8-core Mac) unless told
        # otherwise; that cap, not max_samples, set the concurrency of the first full run.
        max_sandboxes=int(cfg.inspect.max_sandboxes),
        max_connections=int(cfg.inspect.max_connections),
        retry_on_error=int(cfg.inspect.retry_on_error),
        log_dir=str(metadata_dir / "inspect"),
        log_format=str(cfg.inspect.log_format),
        fail_on_error=False,
        display="plain",
    )
    by_task = {p["task"]: p["name"] for p in plan}
    step_logs: dict = {p["name"]: [] for p in plan}
    for log in logs:
        step_logs[by_task[log.eval.task.split("/")[-1]]].append(log)

    run_info = {
        "target": model_str,
        "mode": target.spec.mode,
        "judge_model": str(cfg.judge_model),
        "harness": {"repo": str(cfg.harness.repo), "commit": str(cfg.harness.commit)},
        "epochs": epochs,
        "smoke": smoke,
        "steps": plan,
        "inspect_logs": str((metadata_dir / "inspect").relative_to(out_dir)),
        "inspect_version": importlib.metadata.version("inspect-ai"),
        "verdict_parser_override": "trailing-comma repair (upstream.patch_verdict_parser)",
    }
    return export(step_logs, rollout_dir, results_dir, run_info)
