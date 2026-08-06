# ABOUTME: Grading phase: run saved patches through the PINNED official SWE-bench harness in
# ABOUTME: its own environment — no GPU, no model, runnable long after the pod is destroyed.

"""Turning saved patches into a resolved rate.

Grading is deliberately a separate phase with its own pinned environment. It needs docker and
CPU, not a GPU, so keeping an H100 rented while test suites run would be pure waste; and it
must stay reproducible on its own, so that regrading a year-old `preds.json` does not depend
on the agent's stack still resolving.

What the harness actually does is worth stating, because "grading" sounds cheap and is not:
for each instance it starts the task's prebuilt container, applies the candidate patch, runs
the repository's real test suite, and parses the result against that instance's
`FAIL_TO_PASS`/`PASS_TO_FAIL` lists. The predictions file is kilobytes; the environments
behind it are tens of gigabytes of images.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

# Absolute: `grade()` runs the harness with cwd=grade_dir so its report lands in the run
# directory, which makes any repo-relative `--project` path unresolvable (the first smoke run
# died exactly here).
ENVS = (Path(__file__).resolve().parent / "envs")
HARNESS_ENV = ENVS / "harness"


# Official uv image: ships uv + CPython, so the bridge container needs no apt step.
_BRIDGE_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
# The bridge builds the harness venv INSIDE the container. The repo is bind-mounted, and its
# envs/harness/.venv was created for the host OS — a Linux container cannot execute Windows
# binaries, and uv would otherwise find and try to reuse it. Redirecting the environment keeps
# the committed uv.lock as the source of truth (same pinned versions) while giving Linux its
# own venv.
_BRIDGE_VENV = "/tmp/harness-venv"


def needs_bridge() -> bool:
    """True when the harness cannot run natively and must go through a Linux container.

    `swebench.harness.prepare_images` imports `resource`, a Unix-only stdlib module, at
    package-import time, so on Windows every entrypoint dies before doing anything. Docker
    Desktop does not fix that by itself — the harness PROCESS needs Linux, not just the
    containers it launches. But the host daemon can run a Linux container that mounts the
    docker socket and drives that same daemon, which is what this does.

    The rollout phase has no such constraint and runs fine on Windows.
    """
    import sys

    return sys.platform == "win32"


def check_platform() -> None:
    """Kept for callers that want a hard stop; the bridge means Windows is now supported."""
    return None


def _bridge_wrap(argv: list[str], repo_root: Path, cwd_in_repo: Path) -> list[str]:
    """Wrap a harness invocation so it runs inside a Linux container on the host daemon.

    The container gets three things: the docker socket (so the test containers it starts are
    siblings on the SAME daemon, reusing images already pulled for the rollouts), the repo
    (so the PINNED lockfile is what installs, not a loose `pip install swebench`), and a
    writable venv path outside the mount.
    """
    rel = cwd_in_repo.resolve().relative_to(repo_root.resolve()).as_posix()
    project = "/repo/src/eval/capabilities/swebench_mini/envs/harness"
    # Sync explicitly: `uv run --project` does NOT auto-sync a non-package project (these env
    # projects set `package = false`), and without it uv silently falls back to the system
    # interpreter — which then fails with "No package metadata was found for swebench" rather
    # than anything pointing at the real cause. Verified 2026-08-05.
    inner = (f"uv sync -q --project {project} && "
             + " ".join(shlex.quote(a) for a in argv))
    return [
        "docker", "run", "--rm",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{repo_root.resolve()}:/repo",
        # Named volume, so the harness venv survives between bridge invocations instead of
        # being rebuilt from the lockfile on every call (~1 min each time).
        "-v", f"swebench-harness-venv:{_BRIDGE_VENV}",
        "-e", f"UV_PROJECT_ENVIRONMENT={_BRIDGE_VENV}",
        "-w", f"/repo/{rel}",
        _BRIDGE_IMAGE,
        "bash", "-lc", inner,
    ]


def harness_version() -> str:
    """Version of the pinned harness, asked of the environment that will actually grade."""
    inner = ["uv", "run", "--project", str(HARNESS_ENV), "python", "-c",
             "import importlib.metadata as m; print(m.version('swebench'))"]
    if needs_bridge():
        repo_root = ENVS.parents[4]
        inner = ["uv", "run", "--project",
                 "/repo/src/eval/capabilities/swebench_mini/envs/harness", "python", "-c",
                 "import importlib.metadata as m; print(m.version('swebench'))"]
        cmd = _bridge_wrap(inner, repo_root, repo_root)
    else:
        cmd = inner
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", env=os.environ | {"PYTHONIOENCODING": "utf-8"})
    if out.returncode != 0:
        raise RuntimeError(f"pinned harness env is not usable: {(out.stderr or '')[-600:]}")
    return out.stdout.strip().splitlines()[-1]


def _harness_argv(*, dataset: str, predictions: str, run_id: str, max_workers: int,
                  cache_level: str, namespace: str, split: str = "test",
                  instance_ids: list[str] | None = None,
                  project: str | None = None) -> list[str]:
    """Build the harness invocation. `--report_dir .` with cwd set keeps every artifact
    (report, per-instance test logs) inside the run directory instead of the driver's cwd."""
    argv = ["uv", "run", "--project", project or str(HARNESS_ENV),
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset, "--split", split,
            "--predictions_path", predictions,
            "--run_id", run_id,
            "--max_workers", str(max_workers),
            "--cache_level", cache_level,
            "--namespace", namespace,
            "--report_dir", "."]
    if instance_ids:
        argv += ["--instance_ids", *instance_ids]
    return argv


def verify_environment(*, dataset: str, instance_ids: list[str], out_dir: Path,
                       split: str = "test", max_workers: int = 2,
                       cache_level: str = "env", namespace: str = "swebench",
                       timeout: int = 3600) -> dict:
    """Prove this machine can grade, using gold patches and no model at all.

    The harness accepts `--predictions_path gold`, which submits each instance's own
    reference patch. A correctly configured host resolves 100% of them; anything less means
    the environment is broken — wrong images, docker misconfigured, tests not running — and
    every model number produced on that host would be wrong in the same direction, silently.

    Run this on any fresh grading host BEFORE trusting a single pass@1. It costs a few
    minutes of CPU and no API credit.

    Returns:
        The gold report plus a `passed` flag (all requested instances resolved).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"gold_check_{len(instance_ids)}"
    repo_root = ENVS.parents[4]
    bridge = needs_bridge()
    argv = _harness_argv(dataset=dataset, predictions="gold", run_id=run_id,
                         max_workers=max_workers, cache_level=cache_level,
                         namespace=namespace, split=split, instance_ids=instance_ids,
                         project=("/repo/src/eval/capabilities/swebench_mini/envs/harness"
                                  if bridge else None))
    if bridge:
        argv = _bridge_wrap(argv, repo_root, out_dir)
        print(">>> gold check via Linux bridge container")
    print(">>> " + " ".join(shlex.quote(a) for a in argv))
    log = out_dir / "gold_check.log"
    with log.open("w", encoding="utf-8") as fh:
        # Under the bridge, -w already points at the mounted out_dir.
        proc = subprocess.run(argv, cwd=None if bridge else out_dir,
                              stdout=fh, stderr=subprocess.STDOUT,
                              text=True, env=os.environ | {"PYTHONIOENCODING": "utf-8"},
                              timeout=timeout)
    path = report_path(out_dir, run_id)
    if path is None:
        tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise SystemExit(f"gold check produced no report (exit {proc.returncode}).\n"
                         f"This machine cannot grade. Last log lines:\n{tail}")
    report = json.loads(path.read_text(encoding="utf-8"))
    resolved = set(report.get("resolved_ids") or [])
    missing = sorted(set(instance_ids) - resolved)
    return {"passed": not missing, "n_requested": len(instance_ids),
            "n_resolved": len(resolved & set(instance_ids)), "unresolved_gold": missing,
            "harness_version": harness_version(), "dataset": dataset, "report_file": path.name}


def to_harness_predictions(preds: dict[str, dict], selected_ids: list[str],
                           out_path: Path) -> int:
    """Convert mini-SWE-agent's id-keyed preds.json into the harness's JSONL format.

    Restricted to the selected ids on purpose: a rollout directory that was extended to a
    deeper subset holds predictions for instances outside this run, and grading those would
    silently change the denominator of a result that claims a specific subset hash.

    Returns:
        How many predictions were written.
    """
    rows = [preds[i] | {"instance_id": i} for i in selected_ids if i in preds]
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def report_path(grade_dir: Path, run_id: str) -> Path | None:
    """Locate the harness's run report, which it names after the model and run id."""
    matches = sorted(grade_dir.glob(f"*.{run_id}.json"))
    return matches[-1] if matches else None


def grade(*, preds_path: Path, selected_ids: list[str], dataset: str, revision: str,
          run_id: str, grade_dir: Path, max_workers: int = 8,
          cache_level: str = "env", namespace: str = "swebench",
          timeout: int = 1800) -> dict:
    """Run the pinned harness over saved predictions and return its report.

    Args:
        preds_path: mini-SWE-agent's preds.json.
        selected_ids: The subset this run is defined over (the grading denominator).
        dataset: The SAME dataset string the rollout used — not the friendly alias.
        revision: Dataset revision, recorded so a regrade cannot silently drift.
        run_id: Names the harness's containers and its report file.
        grade_dir: Where predictions, logs and the report land.
        max_workers: Parallel instances. Bounded by CPU and by docker's own limits.
        cache_level: `env` keeps base+environment images between runs (~100GB for the full
            benchmark) and rebuilds thin instance layers; `instance` is faster and needs
            ~2TB. `env` is upstream's default and the right trade here.
        namespace: Pull prebuilt images from this Docker Hub namespace instead of building
            locally — building the environments from source takes hours.

    Returns:
        The harness's parsed report, plus the provenance of the harness itself.
    """
    check_platform()
    grade_dir.mkdir(parents=True, exist_ok=True)
    jsonl = grade_dir / "preds.jsonl"
    n = to_harness_predictions(json.loads(preds_path.read_text(encoding="utf-8")),
                               selected_ids, jsonl)
    if not n:
        raise SystemExit(f"no predictions for the selected subset in {preds_path} — "
                         "nothing to grade (did the rollout phase produce anything?)")

    repo_root = ENVS.parents[4]  # <repo>/src/eval/capabilities/swebench_mini/envs -> <repo>
    bridge = needs_bridge()
    argv = _harness_argv(
        dataset=dataset,
        # Inside the bridge the repo is mounted at /repo, so the predictions path and the
        # harness project must be named in CONTAINER terms, not host terms.
        predictions=(f"/repo/{jsonl.resolve().relative_to(repo_root.resolve()).as_posix()}"
                     if bridge else str(jsonl)),
        run_id=run_id, max_workers=max_workers, cache_level=cache_level, namespace=namespace,
        project=("/repo/src/eval/capabilities/swebench_mini/envs/harness" if bridge else None))
    if bridge:
        argv = _bridge_wrap(argv, repo_root, grade_dir)
        print(">>> grading via Linux bridge container (host harness cannot import on Windows)")
    print(">>> " + " ".join(shlex.quote(a) for a in argv))
    log = grade_dir / "harness.log"
    with log.open("w", encoding="utf-8") as fh:
        # cwd=grade_dir so the harness's report and run artifacts land inside the run
        # directory rather than wherever the driver happened to be invoked from. Under the
        # bridge the same is achieved with -w on the mounted path.
        proc = subprocess.run(argv, cwd=None if bridge else grade_dir, stdout=fh,
                              stderr=subprocess.STDOUT, text=True,
                              env=os.environ | {"PYTHONIOENCODING": "utf-8"},
                              timeout=timeout)

    path = report_path(grade_dir, run_id)
    if path is None:
        tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise SystemExit(f"harness produced no report (exit {proc.returncode}).\n"
                         f"Last log lines:\n{tail}")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_harness"] = {"version": harness_version(), "dataset": dataset,
                          "dataset_revision": revision, "run_id": run_id,
                          "cache_level": cache_level, "namespace": namespace,
                          "n_predictions": n, "report_file": path.name,
                          "lockfile": str(HARNESS_ENV / "uv.lock")}
    return report
