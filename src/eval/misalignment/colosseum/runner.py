# ABOUTME: The registry's run() for colosseum_jira — seat both arms on one vLLM server,
# ABOUTME: drive Colosseum's sweep for one experiment, harvest per-seed measures.

"""One arm of one experiment: six agents, two checkpoints, one GPU.

The shape that makes this affordable is that both checkpoints are LoRA adapters over the
same base model in the same thinking mode, so vLLM holds both at once and a seat picks
its arm by name in the request body. `ServedTarget.sibling()` is what attaches the second
one; there is no second server and no second GPU.

Colosseum runs as a SUBPROCESS rather than an import. It is an installed package on its
own path with its own asyncio orchestration, its own cwd-relative output conventions and
a `clear_seed_directories` call that rmtree's paths relative to the working directory.
Driving it in-process would mean adopting all of that into ours; driving it as a
subprocess with an explicit cwd means the only contract between us is a config file in
and a directory tree out.

run() measures ONE arm. The comparison between arms is pool.py's job, which run_eval
calls after every arm of the invocation has been published — that is the framework's own
answer to "several arms of one recipe", and it is right here because the contrast, not
the arm, is the result.
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
from src.eval.misalignment.colosseum.config import EXPERIMENTS, build_sweep_config
from src.eval.misalignment.colosseum.harvest import MEASURES, harvest_run_root

__all__ = ["run"]


def _colosseum_root(cfg) -> Path:
    """Where the patched Colosseum checkout lives."""
    root = Path(os.environ.get("COLOSSEUM_ROOT") or cfg.colosseum_root).expanduser()
    assert (root / "experiments" / "collusion" / "run.py").is_file(), (
        f"no Colosseum checkout at {root}. Build one with "
        "scripts/infra/slurm/setup_killarney.sh (login node — it needs the network), or "
        "point COLOSSEUM_ROOT at an existing patched clone."
    )
    # A clone that was re-fetched loses the patch, and the failure it causes is a SILENT
    # one: every seat falls back to the sweep's single model and the study compares an arm
    # against itself. Cheap to check here, expensive to discover in the results.
    text = (root / "experiments" / "collusion" / "run.py").read_text()
    assert "_resolve_agent_llm_configs" in text, (
        f"the Colosseum checkout at {root} is missing the per-agent model routing patch. "
        "Re-apply it:\n"
        f"  git -C {root} apply src/eval/misalignment/colosseum/third_party/"
        "per_agent_models.patch"
    )
    return root


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
    """Run one experiment with `target` in the seats under test.

    Args:
        target: The arm under test, as a ServedTarget from run_eval.
        cfg: configs/eval/colosseum_jira.yaml, plus any `key=value` CLI overrides.
        out_dir: This arm's run directory.

    Returns:
        A summary carrying per-cell, per-seed measures for this arm — the input pool.py
        turns into the treatment-minus-control contrasts that are the actual result.
    """
    experiment = str(cfg.experiment)
    assert experiment in EXPERIMENTS, (
        f"experiment={experiment!r}; expected one of {sorted(EXPERIMENTS)}"
    )

    root = _colosseum_root(cfg)
    _, results_dir, metadata_dir = publish_layout(out_dir)

    # The control checkpoint filling the seats that are not under test. For the control
    # arm's own block this is the same repo as `target`: sibling() returns a second handle
    # on the one adapter, which costs nothing and keeps the two blocks structurally
    # identical instead of special-casing one of them.
    peer = target.sibling(str(cfg.peer))

    # Touching base_url is what actually serves a target (serving is lazy). Both are
    # touched HERE, before any config is written, so that a base/mode mismatch or a dead
    # server fails now rather than inside a six-agent episode. The order matters only in
    # that the first call boots vLLM and the second attaches its adapter to it.
    print(f">>> serving {target.model_name} at {target.base_url}")
    print(f">>> seating peer {peer.model_name} at {peer.base_url}")

    sweep_cfg = build_sweep_config(experiment, target, peer, cfg, out_dir)
    config_path = metadata_dir / f"colosseum_{experiment}.yaml"
    config_path.write_text(yaml.safe_dump(sweep_cfg, sort_keys=False))

    output_dir = Path(sweep_cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "rollouts" / "colosseum_driver.log"

    # `max_concurrent_runs` is threads, not processes: asyncio's default executor caps at
    # min(32, cpu_count + 4), so a job with too few CPUs silently runs fewer episodes at
    # once than the config asks for. The other ceiling is vLLM's own --max-num-seqs (32 on
    # this architecture), and six agents per episode means ~5 episodes saturate it.
    _run(
        [
            sys.executable,
            "-m",
            "experiments.collusion.run",
            "--config",
            str(config_path.resolve()),
            "--max-concurrent-runs",
            str(int(cfg.max_concurrent_runs)),
        ],
        cwd=root,
        log=log,
    )

    run_root = _newest_output(output_dir)

    # The exact optimum, per episode. Without this there is no normalised regret — only
    # upstream's ratio against a loose analytic bound. Solved offline from the recorded
    # instance, so it needs no GPU and no network.
    _run(
        [
            sys.executable,
            "experiments/collusion/compute_jira_optimal.py",
            "--root",
            str(run_root.resolve()),
            "--write-json",
        ],
        cwd=root,
        log=log,
    )

    expected = {"target": target.model_name, "peer": peer.model_name}
    harvest = harvest_run_root(run_root, expected_seats=expected)

    (results_dir / "per_seed.json").write_text(
        json.dumps(harvest["measures"], indent=2)
    )
    (results_dir / "episodes.json").write_text(
        json.dumps(harvest["episodes"], indent=2)
    )

    health = harvest["health"]
    if health["zero_tasks_done"]:
        # Not raised: a handful of these is normal. All of them is the reasoning-model
        # failure mode — the trace eats the token budget, no tool call is ever emitted,
        # and every arm scores a clean zero that looks like a null result.
        print(
            f"!!! {health['zero_tasks_done']}/{health['episodes']} episodes completed "
            f"ZERO tasks. If that is most of them, raise max_tokens (currently "
            f"{int(cfg.max_tokens)}) — a truncated <think> block emits no tool call."
        )

    return {
        "experiment": experiment,
        "arm": target.spec.hf_path,
        "peer": str(cfg.peer),
        "seats": expected,
        "measures": harvest["measures"],
        "measure_definitions": dict(MEASURES),
        "health": health,
        "colosseum_run_root": str(run_root.relative_to(out_dir)),
    }
