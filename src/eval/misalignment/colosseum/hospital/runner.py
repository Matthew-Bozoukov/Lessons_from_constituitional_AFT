# ABOUTME: The registry's run() for colosseum_hospital — seat both arms on one vLLM server,
# ABOUTME: drive Colosseum's misalignment sweep for one condition, harvest per-seed measures.

"""One arm of one condition: nine agents, two checkpoints, one GPU.

Both checkpoints are LoRA adapters over the same base in the same thinking mode, so vLLM
holds both at once and a seat picks its arm by name in the request body
(`ServedTarget.sibling()`); there is no second server and no second GPU.

Colosseum runs as a SUBPROCESS with cwd set to its own checkout (the same reasoning as
the Jira runner: its own asyncio orchestration, cwd-relative conventions, an rmtree
helper). The only contract between us is a config file in and a directory tree out.

run() measures ONE arm. The contrast between arms is pool.py's job, which run_eval calls
after every arm of the invocation is published — the contrast, not the arm, is the result.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.eval.layout import publish_layout
from src.eval.misalignment.colosseum.hospital.config import (
    CONDITIONS,
    PAIR,
    build_sweep_config,
)
from src.eval.misalignment.colosseum.hospital.harvest import MEASURES, harvest_run_root

__all__ = ["run"]

PATCH = "src/eval/misalignment/colosseum/third_party/hospital_seating.patch"
FIXES_PATCH = "src/eval/misalignment/colosseum/third_party/hospital_eval_fixes.patch"
TERRARIUM_PATCH = (
    "src/eval/misalignment/colosseum/third_party/terrarium_hospital_fixes.patch"
)


def _colosseum_root(cfg) -> Path:
    """Where the patched Colosseum checkout lives."""
    root = Path(os.environ.get("COLOSSEUM_ROOT") or cfg.colosseum_root).expanduser()
    driver = root / "experiments" / "agent_misalignment" / "run.py"
    assert driver.is_file(), (
        f"no Colosseum checkout at {root}. Clone github.com/umass-ai-safety/colosseum at "
        f"ac0b405, apply {PATCH} and {FIXES_PATCH}, install its dependencies (not the "
        "package) into this environment, and point COLOSSEUM_ROOT (or the config's "
        "colosseum_root) at it."
    )
    # A clone that was re-fetched loses the patches, and the failure is SILENT: every seat
    # falls back to the sweep's single model and the study compares an arm against itself.
    text = driver.read_text()
    assert "_resolve_agent_llm_configs_by_seat" in text, (
        f"the Colosseum checkout at {root} is missing the seating patch. Re-apply it:\n"
        f"  git -C {root} apply {PATCH}"
    )
    # Without this one the `fixes` block is dropped on the floor and no prompt, retry
    # reason or secret block is recorded — a "fixed" run of the original harness.
    assert "secret_instructions" in text, (
        f"the Colosseum checkout at {root} is missing the eval-fixes patch. Re-apply it "
        f"(after the seating patch):\n  git -C {root} apply {FIXES_PATCH}"
    )
    return root


def _terrarium_fixes_version() -> str:
    """The version stamp of the patched terrarium-agents package this interpreter imports.

    Colosseum runs as a subprocess of THIS interpreter, so the package it will import is
    the one importable here. An unpatched package ignores every `fixes` flag silently.
    """
    try:
        from envs.dcops.hospital import hospital_env
        from terrarium.agents import base as terrarium_base
    except ImportError as e:  # pragma: no cover - depends on the host
        raise AssertionError(
            "terrarium-agents is not importable from this environment; the eval runs "
            "Colosseum with this interpreter, so install Colosseum's dependencies here "
            "(scratch/colosseum_hospital/pod_bootstrap.sh does)."
        ) from e
    version = getattr(terrarium_base, "TERRARIUM_FIXES", None)
    assert version and getattr(hospital_env, "HOSPITAL_FIXES", None) == version, (
        "the installed terrarium-agents package is missing the fixes patch, so every "
        f"`fixes` flag would be ignored. Apply {TERRARIUM_PATCH} into its site-packages "
        "(scratch/colosseum_hospital/pod_bootstrap.sh does):\n"
        "  cd $(python -c 'import terrarium, os; print(os.path.dirname("
        f"terrarium.__path__[0]))') && patch -p1 < <repo>/{TERRARIUM_PATCH}"
    )
    return str(version)


def _run(argv: list[str], *, cwd: Path, log: Path) -> None:
    """Run a Colosseum entrypoint, streaming its output to a log kept with the run."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> {' '.join(argv)}  (cwd={cwd})")
    with log.open("a") as fh:
        fh.write(
            f"\n=== {datetime.now().isoformat(timespec='seconds')} {' '.join(argv)}\n"
        )
        fh.flush()
        proc = subprocess.run(argv, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} {argv[-1]} exited {proc.returncode}; last lines of {log}:\n"
            + "\n".join(log.read_text().splitlines()[-40:])
        )


def _newest_output(output_dir: Path) -> Path:
    """The timestamped tree the sweep just wrote."""
    stamped = sorted(
        (p for p in output_dir.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime
    )
    assert stamped, f"the sweep wrote no output tree under {output_dir}"
    return stamped[-1]


def run(target, cfg, out_dir: Path) -> dict:
    """Run one condition with `target` in the pair's two seats.

    Args:
        target: The arm under test, as a ServedTarget from run_eval.
        cfg: configs/eval/colosseum_hospital.yaml, plus any `key=value` CLI overrides.
        out_dir: This arm's run directory.

    Returns:
        A summary carrying per-seed measures for this arm — the input pool.py turns into
        the treatment-minus-control contrasts that are the actual result.
    """
    condition = str(cfg.condition)
    assert condition in CONDITIONS, (
        f"condition={condition!r}; expected one of {sorted(CONDITIONS)}"
    )

    # ABSOLUTE before anything derives a path from it: the sweep's cwd is the checkout.
    out_dir = Path(out_dir).resolve()
    root = _colosseum_root(cfg)
    terrarium_fixes = _terrarium_fixes_version()
    _, results_dir, metadata_dir = publish_layout(out_dir)

    # The control checkpoint in the seven seats that are not the pair. For the control
    # block this is the same repo as `target`.
    peer = target.sibling(str(cfg.peer))

    # Touching base_url is what serves a target (serving is lazy). Both are touched HERE
    # so a base/mode mismatch or a dead server fails now, not inside a nine-agent episode.
    print(f">>> serving {target.model_name} at {target.base_url}")
    print(f">>> seating peer {peer.model_name} at {peer.base_url}")

    sweep_cfg = build_sweep_config(condition, target, peer, cfg, out_dir)
    config_path = metadata_dir / f"colosseum_{condition}.yaml"
    config_path.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False))

    output_dir = Path(sweep_cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "rollouts" / "colosseum_driver.log"

    # `max_concurrent_runs` is episodes in flight. Colosseum drives an episode's agents
    # STRICTLY sequentially (`for agent in env.agents:` with an await inside), so each
    # episode is one in-flight request; the ceiling is vLLM's --max-num-seqs.
    _run(
        [
            sys.executable,
            "-m",
            "experiments.agent_misalignment.run",
            "--config",
            str(config_path.resolve()),
            "--max-concurrent-runs",
            str(int(cfg.max_concurrent_runs)),
        ],
        cwd=root,
        log=log,
    )

    run_root = _newest_output(output_dir)
    expected = {"pair": target.model_name, "other": peer.model_name}
    harvest = harvest_run_root(
        run_root,
        expected_seats=expected,
        expected_fixes=sweep_cfg["fixes"],
        pair=PAIR,
    )

    (results_dir / "per_seed.json").write_text(
        json.dumps(harvest["measures"], indent=2)
    )
    (results_dir / "episodes.json").write_text(
        json.dumps(harvest["episodes"], indent=2)
    )

    health = harvest["health"]
    if health["execution_turns_without_tools"]:
        # Not raised: some are normal (a department with an empty queue). All of them is
        # the reasoning-model failure mode — the trace eats the token budget, no tool
        # call is emitted, and every arm scores a clean zero that looks like refusal.
        print(
            f"!!! {health['execution_turns_without_tools']}/{health['execution_turns']} "
            f"execution turns made NO tool call. If that is most of them, raise "
            f"max_tokens (currently {int(cfg.max_tokens)})."
        )
    if health["failed_runs"]:
        print(
            f"!!! the sweep reported {health['failed_runs']} failed episode(s) "
            f"(see {log}); they are absent from per_seed.json"
        )
    # The harness's own rates, printed with every arm: a retry follows every call that
    # changed nothing, so an arm that refuses more is also re-asked more.
    print(
        f">>> harness: {health['retry_calls']}/{health['llm_calls']} calls were retries "
        f"{health['retry_reasons']}; {health['truncated_calls']} truncated, "
        f"{health['looping_calls']} looping; {health['rejected_tool_calls']} tool calls "
        f"rejected before a handler, {health['scheduler_rejections']} refused by the "
        f"strict scheduler; fixes={sweep_cfg['fixes']}"
    )

    return {
        "condition": condition,
        "arm": target.spec.hf_path,
        "peer": str(cfg.peer),
        "pair": list(PAIR),
        "seats": expected,
        # The harness variant, verbatim: every published number is read against it.
        "fixes": sweep_cfg["fixes"],
        "terrarium_fixes": terrarium_fixes,
        "max_conversation_steps": int(cfg.max_conversation_steps),
        "measures": harvest["measures"],
        "measure_definitions": dict(MEASURES),
        "health": health,
        "colosseum_run_root": str(run_root.relative_to(out_dir)),
    }
