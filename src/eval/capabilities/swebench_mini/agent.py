# ABOUTME: Rollout phase of the standardized baseline: drive the PINNED mini-SWE-agent against
# ABOUTME: our served endpoint, changing nothing about the official config except the endpoint.

"""Running mini-SWE-agent without becoming a fork of it.

The scientific value of this baseline is that it is *stock*. Everything here is therefore
built around one rule: the upstream config file is never edited. It is passed through
untouched and layered under a tiny overlay, because `mini-extra swebench` accepts `-c`
repeatedly and deep-merges the specs in order (`recursive_merge` in its runner). So the
official file keeps its identity — `config_sha256` below is computed on the installed bytes
and can be compared against upstream's — while our two changes stay visible, reviewable and
recorded in the run directory.

Exactly two things are overlaid, and both were approved deliberately:

1. `model.model_kwargs.api_base` — the endpoint swap. This is the whole point.
2. `environment.run_args` — `--network none` added to the container. A documented deviation:
   network isolation is not expressible in the official config, so it cannot be done without
   one. SWE-bench eval images are self-contained, so this should not change what a task can
   do; it removes the possibility of a rollout reaching the internet for a fix.

Two environment variables matter as much as the overlay:

- `MSWEA_COST_TRACKING=ignore_errors` — litellm has no price for a locally-served model, and
  the default mode errors on missing cost info. NOTE the consequence: with cost tracking off,
  the config's `cost_limit: 3.0` can never fire, so the effective per-instance bound is
  `step_limit: 250` alone. That must be stated wherever the baseline is reported.
- `MSWEA_GLOBAL_CONFIG_DIR` — pointed at an empty directory we own. mini-swe-agent otherwise
  auto-loads a machine-global `.env` (observed on this laptop at
  `%LOCALAPPDATA%/mini-swe-agent/mini-swe-agent/.env`), which would let one developer's
  leftover settings silently change the baseline on their machine only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

# Both nested environments are pinned by committed lockfiles; see their pyproject.toml for
# why they are separate from the main project and from each other.
# Absolute, not repo-relative: these paths end up in `uv run --project` argv for subprocesses
# that may run with a different cwd (grading runs inside its own output directory). A relative
# path silently becomes "project directory does not exist" there.
ENVS = (Path(__file__).resolve().parent / "envs")
AGENT_ENV = ENVS / "agent"


def _agent_python(*code: str) -> str:
    """Ask the pinned agent environment about itself (never this repo's interpreter)."""
    out = subprocess.run(
        ["uv", "run", "--project", str(AGENT_ENV), "python", "-c", "\n".join(code)],
        capture_output=True, text=True,
        # The package prints a startup banner containing emoji; on a cp1252 console that
        # crashes the child before it reaches our print.
        env=os.environ | {"PYTHONIOENCODING": "utf-8", "MSWEA_SILENT_STARTUP": "1"})
    if out.returncode != 0:
        raise RuntimeError(f"pinned agent env is not usable: {out.stderr[-600:]}")
    return out.stdout.strip().splitlines()[-1]


def official_config_path() -> Path:
    """Absolute path of the INSTALLED official swebench.yaml (the file we must not edit)."""
    return Path(_agent_python(
        "import minisweagent, pathlib",
        "print(pathlib.Path(minisweagent.__file__).parent / 'config' / 'benchmarks' / 'swebench.yaml')"))


def agent_version() -> str:
    return _agent_python("import importlib.metadata as m", "print(m.version('mini-swe-agent'))")


def config_sha256(path: Path) -> str:
    """Identity of the scaffold's behaviour: prompts, step limit, truncation, timeouts.

    Reported alongside the version because a version number is a claim and this is evidence —
    it is what actually ran, and it is comparable against upstream's file byte for byte.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_overlay(base_url: str, out_dir: Path, *, disable_network: bool = True,
                  pull_timeout: int | None = None) -> Path:
    """Write the minimal `-c` overlay layered on top of the untouched official config."""
    # `--rm` is upstream's own default run_args and is restated because a deep merge REPLACES
    # a list rather than extending it; dropping it would leak a container per instance.
    run_args = ["--rm"] + (["--network", "none"] if disable_network else [])
    overlay = {"model": {"model_kwargs": {"api_base": base_url}},
               "environment": {"run_args": run_args}}
    if pull_timeout:
        # Third documented deviation, and only meaningful when images are pulled in the
        # background: upstream's 120s covers a warm start, not a cold multi-GB download, so
        # an instance that races ahead of the pre-pull would die with TimeoutExpired and an
        # empty patch. Raising the ceiling makes it wait instead of failing.
        overlay["environment"]["pull_timeout"] = int(pull_timeout)
    path = out_dir / "mini_overlay.yaml"
    OmegaConf.save(OmegaConf.create(overlay), path)
    return path


def write_cost_registry(out_dir: Path, model_name: str) -> Path:
    """Zero-cost litellm registry for the locally served model (the documented local path)."""
    path = out_dir / "litellm_model_registry.json"
    path.write_text(json.dumps({model_name: {"input_cost_per_token": 0.0,
                                             "output_cost_per_token": 0.0,
                                             "litellm_provider": "hosted_vllm"}}, indent=2))
    return path


def rollout_command(*, dataset: str, split: str, filter_regex: str, workers: int,
                    model_name: str, rollouts_dir: Path, overlay: Path,
                    official_config: Path) -> list[str]:
    """The exact argv for the rollout, official config first and overlay second."""
    return ["uv", "run", "--project", str(AGENT_ENV), "mini-extra", "swebench",
            # The resolved dataset path, not the `verified` alias: the rollout, the subset
            # draw and the grading harness must all name the SAME dataset or "resolved" is
            # measured against tests the agent never saw.
            "--subset", dataset, "--split", split,
            "--filter", filter_regex,
            "-w", str(workers),
            "-o", str(rollouts_dir),
            "-m", model_name,
            "-c", str(official_config), "-c", str(overlay)]


def rollout_env(*, registry: Path, global_config_dir: Path, api_key: str = "EMPTY") -> dict:
    """Environment for the rollout subprocess (see the module docstring for each variable)."""
    global_config_dir.mkdir(parents=True, exist_ok=True)
    return {
        "MSWEA_COST_TRACKING": "ignore_errors",
        "LITELLM_MODEL_REGISTRY_PATH": str(registry),
        "MSWEA_GLOBAL_CONFIG_DIR": str(global_config_dir),
        "HOSTED_VLLM_API_KEY": api_key,   # vLLM ignores the value; litellm demands one
        "OPENAI_API_KEY": api_key,
        "PYTHONIOENCODING": "utf-8",
    }


def run_rollouts(argv: list[str], env_extra: dict, log_path: Path) -> int:
    """Run the rollout to completion, teeing its output to a log beside the rollouts.

    Returns the exit code rather than raising: a batch that fails some instances still
    produced predictions for the rest, and throwing away completed rollouts because the
    process exited non-zero would be the expensive kind of wrong.
    """
    print(">>> " + " ".join(shlex.quote(a) for a in argv))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(argv, env=os.environ | env_extra, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def provenance(*, official_config: Path, overlay: Path, target, serve_params: dict) -> dict:
    """Everything the report line has to be able to cite, resolved rather than assumed."""
    return {
        "scaffold": "mini-swe-agent",
        "scaffold_version": agent_version(),
        "scaffold_config": str(official_config.name),
        "scaffold_config_sha256": config_sha256(official_config),
        "scaffold_overlay": OmegaConf.to_container(OmegaConf.load(overlay), resolve=True),
        "scaffold_lockfile": str(AGENT_ENV / "uv.lock"),
        "model": target.spec.hf_path,
        "base_model": target.spec.base_model,
        "thinking_mode": target.spec.mode,
        "serving": serve_params,
        # Stated explicitly because it is a real property of the baseline, not an oversight:
        # cost tracking is off for a local endpoint, so the config's cost_limit never fires.
        "effective_instance_budget": "step_limit only (cost_limit inert: local model, "
                                     "MSWEA_COST_TRACKING=ignore_errors)",
    }
