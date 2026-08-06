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


def check_platform() -> None:
    """Refuse Windows up front: the official harness cannot import there.

    `swebench.harness.prepare_images` imports `resource`, a Unix-only stdlib module, at
    package-import time — so every entrypoint dies before it does anything, with a
    `ModuleNotFoundError` buried in a subprocess log. Docker Desktop does not help: the
    harness process itself is what needs to be on Linux, not just the containers.

    The rollout phase has no such constraint and runs fine on Windows.
    """
    import sys

    if sys.platform == "win32":
        raise SystemExit(
            "\nSWE-bench grading cannot run on Windows.\n"
            "  Why: the official harness imports `resource` (Unix-only) at import time;\n"
            "       Docker Desktop does not change that — the harness itself must be on Linux.\n"
            "  Fix: run this phase on Linux against the saved run directory. Either\n"
            "       - a cheap vast.ai CPU instance (docker + ~$0.01/hr), or\n"
            "       - a WSL2 distro with Docker Desktop's WSL integration enabled.\n"
            "  Rollouts are unaffected: produce them anywhere, grade them on Linux.")


def harness_version() -> str:
    out = subprocess.run(
        ["uv", "run", "--project", str(HARNESS_ENV), "python", "-c",
         "import importlib.metadata as m; print(m.version('swebench'))"],
        capture_output=True, text=True, env=os.environ | {"PYTHONIOENCODING": "utf-8"})
    if out.returncode != 0:
        raise RuntimeError(f"pinned harness env is not usable: {out.stderr[-600:]}")
    return out.stdout.strip().splitlines()[-1]


def _harness_argv(*, dataset: str, predictions: str, run_id: str, max_workers: int,
                  cache_level: str, namespace: str, split: str = "test",
                  instance_ids: list[str] | None = None) -> list[str]:
    """Build the harness invocation. `--report_dir .` with cwd set keeps every artifact
    (report, per-instance test logs) inside the run directory instead of the driver's cwd."""
    argv = ["uv", "run", "--project", str(HARNESS_ENV),
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
    check_platform()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"gold_check_{len(instance_ids)}"
    argv = _harness_argv(dataset=dataset, predictions="gold", run_id=run_id,
                         max_workers=max_workers, cache_level=cache_level,
                         namespace=namespace, split=split, instance_ids=instance_ids)
    print(">>> " + " ".join(shlex.quote(a) for a in argv))
    log = out_dir / "gold_check.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(argv, cwd=out_dir, stdout=fh, stderr=subprocess.STDOUT,
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

    argv = _harness_argv(dataset=dataset, predictions=str(jsonl), run_id=run_id,
                         max_workers=max_workers, cache_level=cache_level,
                         namespace=namespace)
    print(">>> " + " ".join(shlex.quote(a) for a in argv))
    log = grade_dir / "harness.log"
    with log.open("w", encoding="utf-8") as fh:
        # cwd=grade_dir so the harness's report and run artifacts land inside the run
        # directory rather than wherever the driver happened to be invoked from.
        proc = subprocess.run(argv, cwd=grade_dir, stdout=fh, stderr=subprocess.STDOUT,
                              text=True, env=os.environ | {"PYTHONIOENCODING": "utf-8"},
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
